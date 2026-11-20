#!/usr/bin/env python3
"""Create DKD CV3 grouped bar figures using verified per-fold vm2 score JSON files.

Three outer test folds, with train+valid+test unlabeled expressions available to
the transductive integrations, but supervised fitting uses train labels only.
Bars: unweighted mean across folds; whiskers: sample SD (ddof=1), not CI.
No imputation for undefined isolated-label F1.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

METRICS = [
    ("asw_batch_sampled", "Batch ASW*"),
    ("ilisi", "iLISI"),
    ("cell_cycle_conservation", "Cell-cycle"),
    ("ari", "ARI"),
    ("nmi", "NMI"),
    ("graph_connectivity", "Graph conn."),
    ("isolated_label_f1", "Isolated F1"),
    ("clisi", "cLISI"),
    ("raw_mean", "Mean†"),
]
METHODS = ("bernn", "scanvi")
COLORS = {"bernn": "#19778C", "scanvi": "#D47F41"}
DISPLAY = {"bernn": "BERNN", "scanvi": "scANVI"}
MCC_GROUPS = [("train", "Train"), ("valid", "Validation"), ("test", "Test")]

def read_folds(root: Path | None):
    exp = root / "experiments" / "single_cell_paper_20261019" if root else None
    snapshots = Path(__file__).resolve().parents[1] / "data" / "cv3"
    records = []
    for fold in range(3):
        for method in METHODS:
            seed = 42 + fold
            if exp is None:
                classification = json.loads((snapshots / f"dkd_fold{fold}_{method}_classifier_scores.json").read_text())
                integration = json.loads((snapshots / f"dkd_fold{fold}_{method}_test_fast8_raw.json").read_text())
            else:
                fitted = (exp / "bernn" / str(seed) / "dkd") if method == "bernn" else (
                    exp / "comparators" / str(seed) / "scanvi" / "dkd")
                scoring = (exp / "cv3_fast8_runs" / "bernn" / str(seed) / "dkd" / "test"
                           if method == "bernn" else
                           exp / "cv3_fast8_runs" / "comparators" / str(seed) / "scanvi" / "dkd" / "test")
                classification = json.loads((fitted / "scores.json").read_text())
                integration = json.loads((scoring / "metrics_raw.json").read_text())
            assert integration["dataset"] == "dkd" and integration["scope"] == "test"
            assert integration["n_eligible_metrics"] == 7, "Dataset comparability changed"
            assert integration["metrics"]["isolated_label_f1"] is None
            assert integration["n_cells"] == classification["n_cells"]["test"]
            assert classification["native"].keys() >= {"train", "valid", "test"}
            row = {"method": method, "fold": fold + 1, "seed": seed,
                   "n_test": integration["n_cells"],
                   "n_test_batches": integration["n_batches"],
                   "native_mcc": classification["native"],
                   "common_probe_mcc": classification["common_probe"],
                   "fast8": {**integration["metrics"],
                             "raw_mean": integration["raw_equal_weight_mean_eligible_only"]}}
            records.append(row)
    assert len(records) == 6
    assert len({(r["method"], r["fold"]) for r in records}) == 6
    return records

def mean_sd(records, method, score):
    numeric = [float(score(r)) for r in records if r["method"] == method
               and score(r) is not None]
    if not numeric:
        return np.nan, np.nan, 0
    assert len(numeric) == 3, f"Missing CV fold values: {method}, {numeric}"
    return float(np.mean(numeric)), float(np.std(numeric, ddof=1)), len(numeric)

def style_ax(ax, ylabel, xlabels, *, head=None):
    ax.set_ylim(0, 1.075)
    ax.set_ylabel(ylabel, fontsize=10.7, color="#264159", labelpad=8)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=10.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color="#D9E2E7", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D9E2E7")
    ax.tick_params(axis="y", labelsize=9, length=0, colors="#5A6F7F")
    ax.tick_params(axis="x", length=0, pad=9, colors="#2E4557")
    if head:
        ax.set_title(head, loc="left", fontsize=13, fontweight="bold",
                     color="#17334C", pad=15)

def bar_panel(ax, records, labels, extractor, header, *, na_index=None,
              annotated=False):
    keys = [key for key, _ in labels]
    x = np.arange(len(keys), dtype=float)
    width = 0.36
    for method, offset in (("bernn", -width/2), ("scanvi", width/2)):
        yy = []; ee = []
        for key in keys:
            avg, sd, count = mean_sd(records, method, lambda r: extractor(r, key))
            yy.append(avg); ee.append(sd)
        yy = np.asarray(yy); ee = np.asarray(ee)
        mask = np.isfinite(yy)
        ax.bar(x[mask] + offset, yy[mask], width=width,
               color=COLORS[method], alpha=0.94,
               linewidth=0, zorder=3, label=DISPLAY[method])
        ax.errorbar(x[mask] + offset, yy[mask], yerr=ee[mask],
                    fmt="none", ecolor="#223445", elinewidth=1.25,
                    capsize=3.1, capthick=1.2, zorder=4)
        if annotated:
            for xi, yi, sd in zip(x[mask] + offset, yy[mask], ee[mask]):
                top = yi + sd
                ax.text(xi, min(1.049,top + 0.024), f"{yi:.3f}",
                        fontsize=8, ha="center", va="bottom", color="#2E4557")
    if na_index is not None:
        ax.text(na_index, 0.48, "N/A\n(no eligible\nisolated labels)",
                ha="center", va="center", fontsize=8.4, color="#8A98A3")
        ax.axvspan(na_index - 0.47, na_index + 0.47,
                   color="#F3F6F8", alpha=0.75, zorder=1)
    return [x, keys]

def create(root, out):
    records = read_folds(root)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for kind, label_set, accessor in (
        ("fast8", METRICS, lambda row, key: row["fast8"][key]),
        ("native_mcc", MCC_GROUPS, lambda row, key: row["native_mcc"][key]),
        ("common_probe_mcc", MCC_GROUPS, lambda row, key: row["common_probe_mcc"][key])):
        for key, _ in label_set:
            for method in METHODS:
                avg, sd, count = mean_sd(records, method, lambda r: accessor(r, key))
                summary.append({"dataset":"dkd", "score_family":kind, "score":key,
                                "method":method,"n_folds":count,
                                "mean":f"{avg:.9f}" if count else "",
                                "sd":f"{sd:.9f}" if count else ""})
    with (out/"Figure_CV3_DKD_3fold_bars_data.csv").open("w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    fig = plt.figure(figsize=(16.9, 9.3), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.28, 0.9],
                          hspace=0.38, top=0.85, bottom=0.105,
                          left=0.07, right=0.98)
    ax = fig.add_subplot(gs[0])
    ax_mcc = fig.add_subplot(gs[1])
    fig.text(0.07, 0.978, "DKD  |  Transductive CV3 evaluation", fontsize=22,
             color="#17334C", fontweight="bold", ha="left", va="top")
    fig.text(0.07, 0.922,
             "3 outer batch-held-out folds  •  TEST-ONLY integration metrics  •  Mean ± SD between folds",
             fontsize=11.2, color="#5F7785", ha="left")
    fig.legend(*ax.get_legend_handles_labels(),loc="upper right",
               bbox_to_anchor=(0.975,0.925),ncol=2,frameon=False,
               fontsize=10.8,handlelength=1.5,columnspacing=1.9)

    bar_panel(ax, records, METRICS, lambda r,k:r["fast8"][k], "Fast-8",na_index=6)
    style_ax(ax, "Raw score (0–1)", [v for _,v in METRICS],
             head="A   Fast-8: embedding quality on held-out batches only")
    ax.axvline(7.5,color="#CAD8DE",linewidth=1.4,linestyle=(0,(3,3)))
    ax.axvline(1.5,color="#CAD8DE",linewidth=1.0)
    ax.text(0.5,1.025,"BATCH MIXING",transform=ax.get_xaxis_transform(),
            ha="center",fontsize=8.5,fontweight="bold",color="#698291")
    ax.text(4.7,1.025,"BIOLOGICAL CONSERVATION",transform=ax.get_xaxis_transform(),
            ha="center",fontsize=8.5,fontweight="bold",color="#698291")

    bar_panel(ax_mcc, records, MCC_GROUPS,
              lambda r,k:r["native_mcc"][k], "Native MCC",annotated=True)
    style_ax(ax_mcc, "MCC (0–1)", [v for _,v in MCC_GROUPS],
             head="B   Native classifier MCC: train, validation and test")
    ax_mcc.axvspan(1.48,2.48,color="#F2F7FA",zorder=0)
    ax_mcc.text(2,0.08,"FINAL TEST",ha="center",fontsize=8.7,
                fontweight="bold",color="#698291")

    # Legend needs actual bars: create after the bars have been rendered.
    for leg in list(fig.legends):
        leg.remove()
    fig.legend(*ax.get_legend_handles_labels(),loc="upper right",
               bbox_to_anchor=(0.967,0.920),ncol=2,frameon=False,
               fontsize=10.8,handlelength=1.5,columnspacing=1.9)

    fig.text(0.07,0.052,
             "† Mean of 7 eligible raw metrics per fold; isolated-label F1 is undefined for DKD (not imputed)."
             "  * Batch ASW is a fixed, stratified-sample approximation.",
             fontsize=9.0,color="#5D7180")
    fig.text(0.07,0.027,
             "Error bars show sample SD across 3 folds (not confidence intervals)."
             " Scores are raw, not OpenProblems-normalized; the experiment is transductive, not inductive.",
             fontsize=9.0,color="#5D7180")
    for ext in ("png","pdf"):
        fig.savefig(out/f"Figure_CV3_DKD_Fast8_and_MCC_bars.{ext}",
                    dpi=230,facecolor="white",bbox_inches="tight")
    plt.close(fig)

    # Stand-alone fast-eight panel suitable for extraction into the paper.
    fig,ax=plt.subplots(figsize=(16.2,5.1),facecolor="white")
    fig.subplots_adjust(left=0.07,right=0.985,bottom=0.18,top=0.79)
    bar_panel(ax,records,METRICS,lambda r,k:r["fast8"][k],"Fast-8",na_index=6)
    style_ax(ax,"Raw score (0–1)",[v for _,v in METRICS])
    ax.axvline(1.5,color="#CAD8DE",linewidth=1.0)
    ax.axvline(7.5,color="#CAD8DE",linewidth=1.3,linestyle=(0,(3,3)))
    fig.text(0.07,0.965,"DKD  |  Fast-8 on held-out TEST batches",fontsize=17,
             fontweight="bold",color="#17334C",va="top")
    fig.text(0.07,0.89,"3 grouped outer folds  •  Bars = mean  •  Whiskers = SD (n = 3)",
             fontsize=10.5,color="#577081")
    fig.legend(*ax.get_legend_handles_labels(),loc="upper right",
               bbox_to_anchor=(0.98,0.89),ncol=2,frameon=False)
    fig.text(0.07,0.051,
             "N/A: no isolated labels eligible for F1.  * Sampled ASW.  † Equal-weight raw mean of 7 eligible metrics; not the official OpenProblems composite.",
             fontsize=8.65,color="#5D7180")
    for ext in ("png","pdf"):
        fig.savefig(out/f"Figure_CV3_DKD_Fast8_testonly_bars.{ext}",
                    dpi=230,facecolor="white",bbox_inches="tight")
    plt.close(fig)
    print("WROTE 4 figure files and CSV in",out)
    for kind in ("fast8","native_mcc"):
        print(kind, {m: [mean_sd(records,m,(lambda r:r["fast8"]["raw_mean"]) if kind=="fast8" else (lambda r:r["native_mcc"]["test"]))[:2]]
                     for m in METHODS})

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--workspace",type=Path,default=None,
                   help="Optional source vm2 CV3 experiment workspace. Defaults to frozen data/cv3 JSON snapshots.")
    p.add_argument("--out",type=Path,default=Path("figures"))
    a=p.parse_args()
    create(a.workspace.resolve() if a.workspace else None,a.out.resolve())
