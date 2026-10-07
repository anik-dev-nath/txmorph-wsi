"""Two-sample distributional metrics for evaluating generated ``z_post``.

WHY THIS MODULE EXISTS
    Under the unpaired reframing the model predicts a *distribution* over
    post-treatment morphology, and no per-patient ground-truth pair exists to
    score against. Reconstruction error is therefore not measurable and must not
    be reported. What is measurable is whether the generated post-treatment
    latents match the held-out real post-treatment latents *as a distribution*.

    All three metrics below are two-sample statistics between a generated set and
    a real held-out set. MMD carries a permutation test, so it yields a p-value
    rather than an uncalibrated number -- the difference between "our samples look
    close" and evidence.

    Sample sizes here are small (the post cohort is 53 patients / 96 slides), so
    always report the permutation p-value alongside the statistic and never read a
    bare MMD as an effect size.
"""
from __future__ import annotations

import numpy as np


def _pairwise_sq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = (x ** 2).sum(1)[:, None]
    y2 = (y ** 2).sum(1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)


def median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    """Median-heuristic RBF bandwidth over the pooled sample."""
    z = np.vstack([x, y])
    d2 = _pairwise_sq(z, z)
    iu = np.triu_indices(len(z), k=1)
    med = np.median(d2[iu])
    return float(med) if med > 0 else 1.0


def mmd2_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased MMD^2 with an RBF kernel.

    Unbiased: the diagonal (self-similarity) terms are excluded, so under the null
    the statistic is centred at 0 and can legitimately come out slightly negative.
    Do not clip it -- a small negative value is information, not an error.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError("MMD needs at least 2 samples per side")
    bw = median_bandwidth(x, y) if bandwidth is None else bandwidth
    g = 1.0 / (2.0 * bw)

    kxx = np.exp(-g * _pairwise_sq(x, x))
    kyy = np.exp(-g * _pairwise_sq(y, y))
    kxy = np.exp(-g * _pairwise_sq(x, y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1))
                 - 2.0 * kxy.mean())


def mmd_permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 1000,
                         seed: int = 0) -> dict:
    """Permutation test for MMD^2. Returns statistic, p-value and null quantiles.

    H0: x and y are drawn from the same distribution. A LARGE p-value is the
    good outcome here -- it means the generated samples are not distinguishable
    from real held-out ones. That inverts the usual reading of a p-value, so state
    it explicitly wherever it is reported.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bw = median_bandwidth(x, y)                 # fixed across permutations
    obs = mmd2_rbf(x, y, bandwidth=bw)

    pooled = np.vstack([x, y])
    n = len(x)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(pooled))
        null[i] = mmd2_rbf(pooled[p[:n]], pooled[p[n:]], bandwidth=bw)
    # +1 correction: the observed value is itself one draw from the null
    pval = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"mmd2": float(obs), "p_value": pval,
            "null_q95": float(np.quantile(null, 0.95)),
            "n_x": int(len(x)), "n_y": int(len(y))}


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Cramer/energy distance. Kernel-free, so it corroborates MMD without
    inheriting its bandwidth choice."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dxy = np.sqrt(_pairwise_sq(x, y)).mean()
    dxx = np.sqrt(_pairwise_sq(x, x)).mean()
    dyy = np.sqrt(_pairwise_sq(y, y)).mean()
    return float(2.0 * dxy - dxx - dyy)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int = 200,
                       p: int = 2, seed: int = 0) -> float:
    """Sliced Wasserstein-p distance via random 1-D projections.

    Exact OT in 512-d on ~100 samples is dominated by the curse of dimensionality;
    slicing keeps it stable and is the standard choice at this sample size.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    theta = rng.normal(size=(d, n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)

    px = np.sort(x @ theta, axis=0)
    py = np.sort(y @ theta, axis=0)
    if len(x) != len(y):     # resample quantiles onto a common grid
        q = np.linspace(0, 1, min(len(x), len(y)))
        px = np.stack([np.quantile(px[:, j], q) for j in range(n_proj)], axis=1)
        py = np.stack([np.quantile(py[:, j], q) for j in range(n_proj)], axis=1)
    return float((np.abs(px - py) ** p).mean() ** (1.0 / p))


def report(gen: np.ndarray, real: np.ndarray, n_perm: int = 1000,
           seed: int = 0) -> dict:
    """All three metrics for a generated vs real held-out set.

    Unbalanced by construction if ``gen`` holds every sample -- prefer
    :func:`report_balanced`, which is the statistically correct comparison.
    """
    out = mmd_permutation_test(gen, real, n_perm=n_perm, seed=seed)
    out["energy_distance"] = energy_distance(gen, real)
    out["sliced_w2"] = sliced_wasserstein(gen, real, seed=seed)
    return out


def report_balanced(gen: np.ndarray, real: np.ndarray, n_rep: int = 25,
                    n_perm: int = 500, seed: int = 0) -> dict:
    """Balanced two-sample comparison: subsample ``gen`` down to ``len(real)``.

    WHY THIS IS THE RIGHT TEST
        A permutation test's power grows with sample size. Comparing every
        generated draw (e.g. 100 samples x 25 patients = 2,500) against ~20 real
        held-out slides is a 100:1 imbalance: the test then rejects on an
        arbitrarily small systematic difference, so a small p-value says "the
        generator is not bit-identical to the truth" rather than "the generator is
        wrong". It is also O(n^2) per permutation, which is what made the pooled
        call intractable.

        Drawing ``len(real)`` generated samples per repetition puts both sides on
        equal footing, so the p-value answers the question actually being asked:
        would a pathologist-blind test distinguish a generated cohort from a real
        one of the same size? Repetitions average out the subsampling noise.

    Returns the median p-value across repetitions (robust to the odd unlucky
    draw), plus the mean statistic and the fraction of repetitions rejecting at
    0.05. A LARGE p-value remains the favourable outcome.
    """
    gen = np.asarray(gen, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    m = len(real)
    if m < 2:
        raise ValueError("need at least 2 real samples")
    rng = np.random.default_rng(seed)

    pvals, mmds, energies, sws = [], [], [], []
    for r in range(n_rep):
        idx = rng.choice(len(gen), size=min(m, len(gen)), replace=False)
        g = gen[idx]
        out = mmd_permutation_test(g, real, n_perm=n_perm, seed=int(rng.integers(1 << 30)))
        pvals.append(out["p_value"])
        mmds.append(out["mmd2"])
        energies.append(energy_distance(g, real))
        sws.append(sliced_wasserstein(g, real, seed=seed))

    pvals = np.asarray(pvals)
    return {
        "p_value_median": float(np.median(pvals)),
        "p_value_mean": float(pvals.mean()),
        "reject_rate_05": float((pvals < 0.05).mean()),
        "mmd2_mean": float(np.mean(mmds)),
        "energy_distance_mean": float(np.mean(energies)),
        "sliced_w2_mean": float(np.mean(sws)),
        "n_rep": int(n_rep), "n_per_side": int(min(m, len(gen))),
        "n_gen_pool": int(len(gen)), "n_real": int(m),
    }
