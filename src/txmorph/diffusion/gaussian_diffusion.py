"""Gaussian diffusion process (DDPM + Improved DDPM learned variance).

Matches TxMorph-WSI_Equations_and_Notation.md exactly:
  forward:   z_t = sqrt(abar_t) z0 + sqrt(1-abar_t) eps
  mean:      mu_theta = (1/sqrt(alpha_t)) (z_t - beta_t/sqrt(1-abar_t) eps_theta)
  variance:  Sigma_theta = exp(v log beta_t + (1-v) log beta_tilde_t),  v=sigmoid(head)
  loss:      L = ||eps - eps_theta||^2 + lambda * L_vlb   (mean stop-gradded in L_vlb)
The denoiser returns (eps, v_raw); v = sigmoid(v_raw) in [0,1].
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .schedule import NoiseSchedule, extract


def normal_kl(mean_q, logvar_q, mean_p, logvar_p):
    """KL(N(mean_q, e^logvar_q) || N(mean_p, e^logvar_p)) elementwise."""
    return 0.5 * (
        logvar_p - logvar_q
        + torch.exp(logvar_q - logvar_p)
        + (mean_q - mean_p) ** 2 * torch.exp(-logvar_p)
        - 1.0
    )


class GaussianDiffusion:
    def __init__(self, schedule: NoiseSchedule, lambda_vlb: float = 1e-3):
        self.sch = schedule
        self.T = schedule.T
        self.lambda_vlb = lambda_vlb

    def to(self, device):
        self.sch.to(device)
        return self

    # ---- forward ----
    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = extract(self.sch.sqrt_alphas_cumprod, t, z0.shape)
        b = extract(self.sch.sqrt_one_minus_acp, t, z0.shape)
        return a * z0 + b * noise

    # ---- tractable posterior q(z_{t-1}|z_t, z0) ----
    def q_posterior(self, z0, zt, t):
        mean = (
            extract(self.sch.posterior_mean_coef0, t, zt.shape) * z0
            + extract(self.sch.posterior_mean_coeft, t, zt.shape) * zt
        )
        var = extract(self.sch.posterior_variance, t, zt.shape)
        logvar = extract(self.sch.posterior_log_variance, t, zt.shape)
        return mean, var, logvar

    def predict_x0_from_eps(self, zt, t, eps):
        a = extract(self.sch.sqrt_alphas_cumprod, t, zt.shape)
        b = extract(self.sch.sqrt_one_minus_acp, t, zt.shape)
        return (zt - b * eps) / a

    # ---- learned reverse p(z_{t-1}|z_t, c) ----
    def p_mean_variance(self, model, zt, t, c=None, learned_var=True, clip_x0=None):
        eps, v_raw = model(zt, t, c)
        x0 = self.predict_x0_from_eps(zt, t, eps)
        if clip_x0 is not None:                       # stabilize unbounded latents
            x0 = x0.clamp(-clip_x0, clip_x0)
        mean, _, _ = self.q_posterior(x0, zt, t)
        if learned_var:
            v = torch.sigmoid(v_raw)
            log_beta = extract(self.sch.log_betas, t, zt.shape)
            log_btilde = extract(self.sch.posterior_log_variance, t, zt.shape)
            logvar = v * log_beta + (1.0 - v) * log_btilde
        else:
            logvar = extract(self.sch.posterior_log_variance, t, zt.shape)
        return mean, logvar, x0

    # ---- losses ----
    def loss(self, model, z0, c=None, t=None):
        B = z0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)
        eps, v_raw = model(zt, t, c)

        # L_simple: noise MSE -> trains the mean
        l_simple = F.mse_loss(eps, noise)

        # L_vlb: KL(q_posterior || p_theta), variance only (mean stop-gradded)
        x0_pred = self.predict_x0_from_eps(zt, t, eps.detach())
        mean_p, _, _ = self.q_posterior(x0_pred, zt, t)        # mean detached via eps.detach()
        v = torch.sigmoid(v_raw)
        log_beta = extract(self.sch.log_betas, t, zt.shape)
        log_btilde = extract(self.sch.posterior_log_variance, t, zt.shape)
        logvar_p = v * log_beta + (1.0 - v) * log_btilde
        mean_q, _, logvar_q = self.q_posterior(z0, zt, t)
        l_vlb = normal_kl(mean_q, logvar_q, mean_p, logvar_p).mean()

        total = l_simple + self.lambda_vlb * l_vlb
        return {"total": total, "simple": l_simple.detach(), "vlb": l_vlb.detach()}

    # ---- sampling ----
    @torch.no_grad()
    def p_sample(self, model, zt, t, c=None, learned_var=True, clip_x0=None):
        mean, logvar, _ = self.p_mean_variance(model, zt, t, c, learned_var, clip_x0)
        noise = torch.randn_like(zt)
        nonzero = (t != 0).float().reshape(-1, *([1] * (zt.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * logvar) * noise

    @torch.no_grad()
    def sample(self, model, n, dim, c=None, device="cpu", learned_var=True, clip_x0=None):
        """Draw n samples. c is (1,dc) or (n,dc) context, or None."""
        zt = torch.randn(n, dim, device=device)
        if c is not None and c.shape[0] == 1:
            c = c.expand(n, -1)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            zt = self.p_sample(model, zt, t, c, learned_var, clip_x0)
        return zt

    @torch.no_grad()
    def estimate_nll(self, model, z0, c, n_mc: int = 50):
        """Monte Carlo NLL proxy: mean denoising loss over random timesteps.

        Lower value → z0 is more likely under the model conditioned on c.

        **Do not difference two independent calls of this** to compare hypotheses —
        each call carries MC noise far larger than the gap between conditions, so
        the difference is dominated by sampling noise. Use :meth:`nll_contrast`,
        which shares the random draws across hypotheses. Returns (B,).
        """
        return self.nll_contrast(model, z0, [c], n_mc=n_mc)[0]

    @torch.no_grad()
    def nll_contrast(self, model, z0, contexts, n_mc: int = 50, generator=None):
        """NLL proxy for several contexts under **common random numbers**.

        Every hypothesis in ``contexts`` is scored on the identical timesteps and
        the identical noise, so the shared randomness cancels when the results are
        differenced. This is what makes a likelihood-ratio P(pCR) usable: with
        independent draws per hypothesis the MC noise swamps the signal (measured
        on weakly-separated toy data: AUROC 0.504 independent vs 0.598 paired,
        with ~120x the spread), which would silently reduce the biomarker to
        chance rather than fail loudly.

        Returns ``(len(contexts), B)`` of per-sample NLL estimates.
        """
        acc = [torch.zeros(z0.shape[0], device=z0.device) for _ in contexts]
        for _ in range(n_mc):
            t = torch.randint(0, self.T, (z0.shape[0],), device=z0.device,
                              generator=generator)
            noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype,
                                generator=generator)
            zt = self.q_sample(z0, t, noise)           # shared corruption
            for j, c in enumerate(contexts):
                eps, _ = model(zt, t, c)
                acc[j] += F.mse_loss(eps, noise, reduction="none").mean(dim=-1)
        return torch.stack([a / n_mc for a in acc])
