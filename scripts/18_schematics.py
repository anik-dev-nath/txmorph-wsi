"""18_schematics.py - Fig 1 (pipeline) and Fig 2 (diffusion architecture).

CLAUDE.md S15 main figures 1 and 2. Both depict the **delivered single-timepoint
system**, not the paired design in S1 of the specification: no post-treatment
slide exists in the primary cohort, so a schematic showing p(z_post | z_pre, drug)
would misrepresent what was built and evaluated. Blocks that the pivot left
undefined are drawn explicitly as unavailable rather than omitted.

Palette follows viz/style.py: teal = frozen/signal, purple = model, amber = outputs.
"""
import argparse
from pathlib import Path

TEAL, PURPLE, AMBER, GREY = "#2A9D8F", "#7B6CB0", "#E9A23B", "#9AA0A6"
TEAL_F, PURPLE_F, AMBER_F, GREY_F = "#EAF4F4", "#EFEBF7", "#FDF3E7", "#F1F3F4"


def _box(ax, x, y, w, h, text, ec, fc, fontsize=6.8, style="round,pad=0.08", lw=1.1,
         alpha=1.0, ls="-"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                                alpha=alpha, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, xy, xytext, color="#555", lw=1.0, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls))


def fig_pipeline(out_base: str):
    """Fig 1: WSI -> tiles -> frozen encoder -> MIL -> conditional diffusion -> P(pCR)."""
    import matplotlib.pyplot as plt
    from txmorph.viz.style import set_style, save_figure
    set_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    ax.set_xlim(0, 27); ax.set_ylim(0, 11.6); ax.axis("off")

    _box(ax, 0.3, 6.5, 3.5, 2.0, "Pre-treatment\nH&E WSI\nIMPRESS $n$=126", GREY, GREY_F, 6.2)
    _box(ax, 4.4, 6.5, 3.5, 2.0, "Tiling\n256$\\times$256 @ 20$\\times$\ntissue + blur", GREY, GREY_F, 6.2)
    _box(ax, 8.5, 6.5, 3.7, 2.0, "Patch encoder\nFROZEN\nSimCLR / Phikon-v2", TEAL, TEAL_F, 6.2)
    # The delivered system mean-pools the tile bag: paired.parquet latents are
    # byte-identical to a plain mean (max abs diff 0.0). attention_mil.py exists
    # and mil.pt was trained, but the kill-switch fix switched to mean pooling and
    # nothing downstream reads the MIL any more. Drawn as built, not as specified.
    _box(ax, 12.8, 6.5, 3.5, 2.0, "Mean pooling\nover tile bag\n(MIL bypassed)", TEAL, TEAL_F, 6.2)
    _box(ax, 16.9, 6.7, 2.5, 1.6, "$z_{pre}$\n512-d", TEAL, TEAL_F, 7.0)
    for a, b in ((3.8, 4.4), (7.9, 8.5), (12.2, 12.8), (16.3, 16.9)):
        _arrow(ax, (b, 7.5), (a, 7.5))

    _box(ax, 12.8, 3.9, 3.5, 1.5, "Drug encoder\none-hot", TEAL, TEAL_F, 6.2)
    _box(ax, 16.9, 4.0, 2.5, 1.3, "$d_{drug}$\n128-d", TEAL, TEAL_F, 7.0)
    _arrow(ax, (16.9, 4.65), (16.3, 4.65))
    _box(ax, 16.9, 1.9, 2.5, 1.2, "pCR bit", PURPLE, PURPLE_F, 6.6)

    _box(ax, 20.2, 3.4, 2.9, 5.0, "context $c$\n\nLN(W[$d\\,\\|\\,$pCR])\n\n512-d",
         PURPLE, PURPLE_F, 6.2)
    for y in (4.65, 2.5):
        _arrow(ax, (20.2, 5.6), (19.4, y), color=PURPLE)
    # z_pre is deliberately NOT routed into the context (see Fig 2)
    _arrow(ax, (20.2, 6.6), (19.4, 7.5), color=GREY, ls="--")

    _box(ax, 23.6, 3.4, 3.1, 5.0,
         "TxMorph\nconditional\nlatent DDPM\n\n3 blocks · FiLM\n$\\epsilon_\\theta$ · $v_\\theta$",
         PURPLE, PURPLE_F, 6.2)
    _arrow(ax, (23.6, 5.9), (23.1, 5.9), color=PURPLE)

    _box(ax, 9.6, 0.4, 6.4, 2.1,
         "Likelihood ratio\nNLL(pCR=1) vs NLL(pCR=0)\n$\\rightarrow$ calibrated $P(pCR)$",
         AMBER, AMBER_F, 6.2)
    _arrow(ax, (16.0, 1.8), (24.4, 3.4), color=AMBER)

    _box(ax, 1.4, 0.4, 7.0, 2.1,
         "Undefined without paired pre/post:\n$\\bar z_{post}$ · residual $r$\n"
         "$\\sigma^2_{tx}$ is $p(1{-}p)$, not independent",
         GREY, "white", 5.9, ls="--", lw=0.9)

    ax.text(13.5, 10.9, "Fig 1 — TxMorph-WSI delivered pipeline (single-timepoint)",
            ha="center", fontsize=8.4)
    ax.text(13.5, 10.0, "teal = frozen   ·   purple = trained   ·   amber = reported output"
                        "   ·   dashed grey = unavailable",
            ha="center", fontsize=5.9, color="#666")
    return save_figure(fig, out_base)


def fig_diffusion(out_base: str):
    """Fig 2: forward/reverse process and the conditioned denoiser."""
    import numpy as np
    import matplotlib.pyplot as plt
    from txmorph.viz.style import set_style, save_figure
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 24); ax.set_ylim(0, 12); ax.axis("off")

    xs = np.linspace(0.8, 13.4, 5)
    bw = 2.3
    labels = ["$z_0$", "$z_t$", "", "", "$z_T$"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        shade = i / (len(xs) - 1)
        _box(ax, x, 7.6, bw, 1.5, lab or "$\\cdots$", TEAL,
             plt.cm.Greys(0.05 + 0.5 * shade), fontsize=7.4)
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i + 1], 8.35), (xs[i] + bw, 8.35))
        ax.text((xs[i] + bw + xs[i + 1]) / 2, 8.9, "$q$", ha="center", fontsize=6.0,
                color="#555")
    ax.text(8.0, 10.5, "forward: $z_t=\\sqrt{\\bar\\alpha_t}z_0+"
                       "\\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$   (cosine schedule)",
            ha="center", fontsize=6.8)
    ax.text(15.9, 8.35, "$\\sim\\mathcal{N}(0,I)$", ha="left", va="center", fontsize=6.4,
            color="#555")

    for i in range(len(xs) - 1, 0, -1):
        _arrow(ax, (xs[i - 1] + bw, 6.9), (xs[i], 6.9), color=PURPLE, ls="--")
    ax.text(8.0, 5.9, "reverse: $z_{t-1}=\\mu_\\theta(z_t,t,c)+\\Sigma_\\theta^{1/2}\\xi$,   "
                      "$\\Sigma_\\theta=\\exp(v\\log\\beta_t+(1-v)\\log\\tilde\\beta_t)$",
            ha="center", fontsize=6.4, color=PURPLE)

    _box(ax, 18.0, 5.6, 5.4, 4.4, "", PURPLE, "white", lw=1.2)
    ax.text(20.7, 9.55, "denoiser $\\epsilon_\\theta$, $v_\\theta$", ha="center", fontsize=7.4)
    _box(ax, 18.5, 8.4, 2.1, 0.9, "$z_t$", GREY, GREY_F, fontsize=6.6)
    _box(ax, 20.9, 8.4, 2.1, 0.9, "sin($t$)", GREY, GREY_F, fontsize=6.6)
    for yb, lab in ((7.2, "MLP res-block  x3"), (6.2, "FiLM($c$): $\\gamma\\odot h+\\beta$")):
        _box(ax, 18.5, yb, 4.5, 0.85, lab, PURPLE, PURPLE_F, fontsize=6.4)
    _arrow(ax, (20.0, 8.1), (19.5, 8.4), color=PURPLE)
    _box(ax, 18.5, 3.9, 2.1, 0.9, "$\\epsilon_\\theta$", AMBER, AMBER_F, fontsize=6.8)
    _box(ax, 20.9, 3.9, 2.1, 0.9, "$v_\\theta$", AMBER, AMBER_F, fontsize=6.8)
    _arrow(ax, (19.5, 4.8), (19.8, 6.2), color=AMBER)
    _arrow(ax, (21.9, 4.8), (21.5, 6.2), color=AMBER)

    _box(ax, 0.8, 2.3, 6.6, 2.0,
         "context $c$ = LN(W[$d_{drug}\\,\\|\\,$pCR])\n\n"
         "$z_{pre}$ deliberately EXCLUDED", PURPLE, PURPLE_F, fontsize=6.2)
    _arrow(ax, (18.4, 6.4), (7.4, 3.3), color=PURPLE)
    _box(ax, 8.4, 2.3, 8.4, 2.0,
         "Conditioning on $z_{pre}$ while denoising $z_{pre}$ lets the\n"
         "network recover $\\epsilon$ from context alone, ignoring the pCR bit\n"
         "(toy: AUROC 0.436 leaked vs 0.864 clean)", GREY, "white",
         fontsize=5.8, ls="--", lw=0.9)

    ax.text(12, 11.4, "Fig 2 — conditional latent diffusion, single-timepoint formulation",
            ha="center", fontsize=8.6)
    ax.text(12, 0.9, "Loss  $L=\\|\\epsilon-\\epsilon_\\theta\\|^2+\\lambda L_{vlb}$,  "
                     "$\\lambda=10^{-3}$, mean stop-gradiented in $L_{vlb}$",
            ha="center", fontsize=6.6, color="#555")
    return save_figure(fig, out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/anik-server/runs/manuscript")
    args = ap.parse_args()
    out = Path(args.out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print("[Fig1]", fig_pipeline(str(out / "fig1_pipeline")))
    print("[Fig2]", fig_diffusion(str(out / "fig2_diffusion")))


if __name__ == "__main__":
    main()
