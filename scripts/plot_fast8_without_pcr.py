#!/usr/bin/env python3
"""Recompute and plot the audited eight-metric single-cell panel, excluding PCR.

All summaries are recalculated from the same seed-42 raw metric values as
Figure 2 fast-9.  No model training or metric estimation is performed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plot_figures import (
    DATA, OUTPUT, DATASETS, LABELS, METHODS, DISPLAY,
    METRICS, METRIC_LABELS, load_fast9
)

def score_circle(ax, x, y, value, *, bold=False):
    v=float(value)
    ax.scatter(x,y,s=610+300*v,c=[v],cmap=plt.colormaps["viridis"],
               vmin=0,vmax=1,linewidth=.85,edgecolor="white",zorder=3)
    ax.text(x,y,f"{v:.2f}",ha="center",va="center",fontsize=8.1,
            color="white" if v<.67 else "#193343",
            fontweight="bold" if bold else "medium",zorder=4)


def main():
    full, valid = load_fast9()
    included = [m for m in METRICS if m != "pcr"]
    included_labels = [label for metric, label in zip(METRICS, METRIC_LABELS) if metric != "pcr"]
    assert len(included) == len(included_labels) == 8
    assert "pcr" not in included

    valid = valid[valid.metric.isin(included)].copy()
    counts = valid.groupby(["method", "dataset"]).size()
    assert set(counts.index) == {(method, ds) for method in METHODS for ds in DATASETS}
    for method in METHODS:
        assert counts[method, "dkd"] == 7
        for dataset in DATASETS[1:]:
            assert counts[method, dataset] == 8
    assert len(valid) == 8 * (7 + 5 * 8)  # 376 eligible entries

    metric_mean = valid.groupby(["method", "metric"]).score.mean()
    dataset_mean = valid.groupby(["method", "dataset"]).score.mean()
    score_by_method = {
        m: float(dataset_mean.loc[m].mean()) for m in METHODS
    }  # Equal weight across all six datasets

    fig, ax = plt.subplots(figsize=(16.5, 7.25))
    fig.subplots_adjust(left=.165, right=.925, top=.785, bottom=.19)
    cmap = plt.colormaps["viridis"]
    for yi, method in enumerate(METHODS):
        for xi, metric in enumerate(included):
            value = float(metric_mean[(method, metric)])
            score_circle(ax,xi,yi,value)
            n = int(((valid.method == method) & (valid.metric == metric)).sum())
            if n != len(DATASETS):
                ax.text(xi + .31, yi - .27, f"{n}/6", fontsize=7.5,
                        ha="center", color="#aa5d42")
        for di, dataset in enumerate(DATASETS):
            value = float(dataset_mean[(method, dataset)])
            score_circle(ax,8+di,yi,value,bold=method=="bernn")
        score_circle(ax,14.4,yi,score_by_method[method],bold=True)

    ax.set_xlim(-.6, 15.1)
    ax.set_ylim(-.75, len(METHODS) - .4)
    ax.invert_yaxis()
    ax.set_yticks(range(len(METHODS)), [DISPLAY[m] for m in METHODS], fontsize=12)
    ax.set_xticks(list(range(8)) + list(range(8, 14)) + [14.4],
                  included_labels + [s.replace(" ", "\n") for s in LABELS] + ["Raw\nmean"],
                  fontsize=9, rotation=0)
    ax.tick_params(axis="x", length=0, pad=13)
    ax.tick_params(axis="y", length=0, pad=8)
    for x in [-.5, 1.5, 7.5, 13.55, 14.95]:
        ax.axvline(x, linewidth=1, color="#c3ccd3", zorder=0)
    for y in np.arange(.5, 8, .98):
        ax.axhline(y, linewidth=.5, color="#e4eaf0", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(.165, .878, "BATCH ALIGNMENT", fontsize=11, weight="bold", color="#174c6f")
    fig.text(.302, .878, "BIOLOGICAL CONSERVATION", fontsize=11, weight="bold", color="#174c6f")
    fig.text(.62, .878, "PER-DATASET RAW MEAN", fontsize=11, weight="bold", color="#174c6f")
    fig.text(.85, .878, "SUMMARY", fontsize=11, weight="bold", color="#174c6f")
    fig.text(.095, .965, "Single-cell integration | Matched fast-8 audit (PCR excluded)",
             fontsize=20, weight="bold", color="#133553")
    fig.text(.095, .925, "One fit (seed 42) per method/dataset | 8 metrics + 6 atlases + mean, all displayed as numeric bubbles",
             fontsize=11, color="#42647a")

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([.93, .235, .012, .45])
    fig.colorbar(sm, cax=cax, ticks=np.arange(0, 1.01, .2), label="Raw metric (0–1)")
    fig.text(.08, .075,
             "*Batch ASW is a fixed stratified-sample approximation, not the exhaustive OpenProblems value.\n"
             "Isolated F1 is N/A on DKD (no eligible isolated types); 5/6 denotes five eligible datasets.\n"
             "All dataset means and overall scores are recalculated without PCR. Raw summaries are descriptive, not official OpenProblems scores.",
             fontsize=9, color="#435a6d", va="top")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"Figure2_fast8_without_PCR_seed42_raw_audited.{ext}",
                    dpi=230, bbox_inches="tight")
    plt.close(fig)

    rows = [
        {"method": method,
         "dataset": dataset,
         "n_eligible_metrics": int(counts[method, dataset]),
         "dataset_mean_raw_no_pcr": float(dataset_mean[method, dataset]),
         "mean_of_six_datasets_raw_no_pcr": score_by_method[method]}
        for method in METHODS for dataset in DATASETS
    ]
    summary_path = OUTPUT / "Figure2_fast8_without_PCR_scores.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print("Eight-metric mean (equal dataset weight):")
    for method in sorted(METHODS, key=lambda m: -score_by_method[m]):
        print(f"  {DISPLAY[method]:<9} {score_by_method[method]:.4f}")
    print("Saved PNG, PDF and CSV into", OUTPUT)

if __name__ == "__main__":
    main()
