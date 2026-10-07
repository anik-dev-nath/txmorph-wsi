"""ONE publication plotting style - import everywhere (CLAUDE.md S15).
Vector output (PDF+SVG), 300-dpi PNG fallback, colorblind-safe, fonts >=7pt.
Reuse cheat-sheet palette: teal=signal, purple=model, amber=outputs.
"""
from pathlib import Path

PALETTE = {"signal": "#1D9E75", "model": "#6C63C4", "output": "#C8821A", "muted": "#9A9A93"}
# colorblind-safe categorical cycle for multi-group plots (clusters, arms, subtypes)
CATEGORICAL = ["#1D9E75", "#6C63C4", "#C8821A", "#3A7CA5", "#D1495B", "#8A8D3F",
               "#9A9A93", "#4B4B4B"]


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "savefig.bbox": "tight",
    })


def save_figure(fig, out_base: str):
    """Write a figure as PDF + SVG (vector) and PNG (300-dpi raster fallback).

    ``out_base`` is a path without extension. Parent dirs are created.
    """
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(base.with_suffix(f".{ext}"))
    return str(base)
