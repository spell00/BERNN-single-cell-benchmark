#!/usr/bin/env python3
"""Restore historical 13 October 2026 fast-nine panel without PCR.

Unlike plot_fast8_without_pcr.py, which analyzes later single-seed, train-batch
experiments, THIS script uses the original archive's scaled_scores values,
preserving its historical score contract and the original method/control rows.

An audit CSV explicitly records missingness, fit scope, scaled vs raw values,
and both NA=0 (historical) and N/A-excluded averages.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/archived/results.json"
OUT = ROOT / "figures"
DSETS = ["dkd","gtex_v9","hypomap","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens"]
DSNAMES = ["DKD","GTEx v9","HypoMap","Immune Cell\nAtlas","Mouse\nPancreas","Tabula\nSapiens"]
METRICS = ["asw_batch","ilisi","cell_cycle_conservation","ari","nmi",
           "graph_connectivity","isolated_label_f1","clisi"]
MLABELS = ["Batch ASW","iLISI","Cell cycle","ARI","NMI",
           "Graph conn.","Isolated F1","cLISI"]
BASELINE = ["bernn","combat","scanvi","scvi","harmony",
            "batchelor_fastmnn","liger","scalex","uce",
            "no_integration","bbknn","scanorama","embed_cell_types"]
FULL = ["embed_cell_types","embed_cell_types_jittered","bernn",
        "combat","scanvi","scvi","harmony","harmonypy",
        "batchelor_fastmnn","batchelor_mnn_correct","liger","pyliger",
        "scalex","uce","no_integration","no_integration_batch",
        "bbknn","scanorama","geneformer","mnnpy","scimilarity",
        "scgpt_zeroshot","scgpt_finetuned","scprint",
        "shuffle_integration","shuffle_integration_by_batch",
        "shuffle_integration_by_cell_type"]
NAME = {"bernn":"BERNN","combat":"ComBat","scanvi":"scANVI","scvi":"scVI",
        "harmony":"Harmony","harmonypy":"HarmonyPy",
        "batchelor_fastmnn":"fastMNN","batchelor_mnn_correct":"MNN correct",
        "liger":"LIGER","pyliger":"pyLIGER","scalex":"SCALEX",
        "uce":"UCE","no_integration":"No integration",
        "no_integration_batch":"No integration (batch)",
        "bbknn":"BBKNN","scanorama":"Scanorama",
        "embed_cell_types":"True labels (oracle)",
        "embed_cell_types_jittered":"Oracle (jittered)",
        "geneformer":"Geneformer","mnnpy":"mnnpy","scimilarity":"scImilarity",
        "scgpt_zeroshot":"scGPT zero-shot","scgpt_finetuned":"scGPT fine-tuned",
        "scprint":"scPRINT",
        "shuffle_integration":"Shuffle integration",
        "shuffle_integration_by_batch":"Shuffle by batch",
        "shuffle_integration_by_cell_type":"Shuffle by cell type"}
CONTROLS = {"embed_cell_types","embed_cell_types_jittered",
             "no_integration","no_integration_batch",
             "shuffle_integration","shuffle_integration_by_batch",
             "shuffle_integration_by_cell_type"}
ORACLES={"embed_cell_types","embed_cell_types_jittered"}

def is_real(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def source():
    data=json.loads(SOURCE.read_text())
    assert len(data)==162, len(data)
    keyed={(a["method_id"],a["dataset_id"].split("/")[-1]):a for a in data}
    assert len(keyed)==162
    assert set(keyed)=={(m,d) for m in FULL for d in DSETS}
    records=[]
    for m in FULL:
        for ds in DSETS:
            r=keyed[m,ds]
            for metric in METRICS:
                raw=r["metric_values"].get(metric)
                scaled=r["scaled_scores"].get(metric)
                valid=is_real(raw) and is_real(scaled)
                # Explicit status rather than silently treating structurally missing as 0.
                status="ok" if valid else ("not_applicable" if metric=="isolated_label_f1" and ds=="dkd" else "not_reported")
                records.append({
                    "method":m,"dataset":ds,"metric":metric,
                    "raw":float(raw) if valid else np.nan,
                    "scaled":float(scaled) if valid else np.nan,
                    "score_historical_na0":float(scaled) if valid else 0.0,
                    "status":status,"comparison_source":r["comparison_source"],
                    "comparison_contract":r["comparison_contract"]})
    return pd.DataFrame.from_records(records),keyed

def audit(raw,keyed):
    a=[]
    for method in FULL:
        grp=raw[raw.method==method]
        source_name=keyed[method,"dkd"]["comparison_source"]
        dataset_scores={}
        dataset_valid={}
        alternate={}
        for ds in DSETS:
            r=grp[grp.dataset==ds]
            good=r[r.status=="ok"]
            dataset_valid[ds]=len(good)
            dataset_scores[ds]=float(r.score_historical_na0.mean()) if len(good)>0 else np.nan
            alternate[ds]=float(good.scaled.mean()) if len(good)>0 else np.nan
        available=[v for v in dataset_scores.values() if math.isfinite(v)]
        available_alt=[v for v in alternate.values() if math.isfinite(v)]
        a.append({
            "method":method,"display_name":NAME[method],
            "is_control":method in CONTROLS,"archive_fit_source":source_name,
            "n_available_metric_dataset_cells":int((grp.status=="ok").sum()),
            "n_possible_metric_dataset_cells":len(grp),
            "fast8_archive_na0":float(np.mean(available)) if available else np.nan,
            "fast8_exclude_missing":float(np.mean(available_alt)) if available_alt else np.nan,
            **{f"{ds}_na0":dataset_scores[ds] for ds in DSETS},
            **{f"{ds}_n_eligible":dataset_valid[ds] for ds in DSETS},
            "source_fast9_score":sum(
                keyed[method,ds]["mean_score"] for ds in DSETS
            )/len(DSETS)
        })
    result=pd.DataFrame(a)
    raw.to_csv(OUT/"Fast8_original_archive_metrics_without_PCR.csv",index=False)
    result.to_csv(OUT/"Fast8_original_archive_score_audit_without_PCR.csv",index=False)
    return result

def score_circle(ax, x, y, value, cmap, *, big=False, bold=False):
    """A uniformly readable numeric bubble for metrics AND dataset summaries."""
    v=float(value)
    size=(260 + 220*v) if big else (610 + 300*v)
    ax.scatter(x,y,s=size,c=[v],cmap=cmap,vmin=0,vmax=1,
               edgecolors="white",linewidths=.6 if big else .85,
               alpha=.97,zorder=4)
    # YlGnBu switches from a light yellow to dark blue; adapt text contrast.
    ink="white" if v>=.52 else "#173349"
    ax.text(x,y,f"{v:.2f}",ha="center",va="center",
            fontsize=5.6 if big else 8.2,
            fontweight="bold" if bold else "medium",
            color=ink,zorder=5)


def draw(metrics, summary, selected, filename, big=False):
    n=len(selected); cols_metric=len(METRICS)
    # 8 numeric bubble metrics + 6 numeric bubble datasets + numeric bubble summary
    width=18.4 if not big else 17.5
    height=9.0 if not big else 14.4
    fig,ax=plt.subplots(figsize=(width,height))
    fig.subplots_adjust(left=.15 if not big else .185,right=.945,top=.81 if not big else .88,bottom=.16 if not big else .095)
    ax.set_xlim(-.7,15.25)
    ax.set_ylim(-.9,n-.3)
    ax.invert_yaxis()
    ax.set_facecolor("white")
    cmap=plt.colormaps["YlGnBu"]
    x_split=8.0
    for i,method in enumerate(selected):
        if i%2==1:
            ax.axhspan(i-.49,i+.49,facecolor="#f0f4f7",zorder=0)
        if method in CONTROLS and not big:
            ax.axhspan(i-.49,i+.49,facecolor="#edf6f4",zorder=0)
        m=metrics[metrics.method==method]
        for j,metric in enumerate(METRICS):
            vals=m[m.metric==metric]
            assert len(vals)==6
            nvalid=int(vals.status.eq("ok").sum())
            if nvalid==0:
                ax.text(j,i,"N/A",ha="center",va="center",fontsize=7.8,color="#b46459")
                continue
            # Historical score contract: NA contributes zero.
            value=float(vals.score_historical_na0.mean())
            score_circle(ax,j,i,value,cmap,big=big)
            if nvalid<6:
                ax.text(j+.25,i-.27,f"{nvalid}/6",fontsize=7.0 if not big else 5.4,
                        color="#9d6250",ha="center")
        record=summary.set_index("method").loc[method]
        for k,ds in enumerate(DSETS):
            v=record[f"{ds}_na0"]
            if math.isfinite(v):
                score_circle(ax,8.1+k,i,v,cmap,big=big)
            else:
                ax.text(8.1+k,i,"N/A",fontsize=9 if not big else 7,
                        ha="center",va="center",color="#af6c5c")
        v=record["fast8_archive_na0"]
        if math.isfinite(v):
            score_circle(ax,14.55,i,v,cmap,big=big,bold=True)
        else:
            ax.text(14.55,i,"N/A",ha="center",va="center",
                    color="#b46459",fontsize=8.5)

    for sep in [-.48,1.5,7.5,13.5,15.00]:
        ax.axvline(sep,color="#b9cbd4",lw=1.0,zorder=1)
    # Divider just above oracle, consistent with original figure.
    if not big and selected[-1] in ORACLES:
        ax.axhline(n-1-.5,color="#6e98ac",ls="--",lw=1.2,zorder=3)
    ax.set_yticks(range(n),[NAME[m] for m in selected],
                  fontsize=10.6 if not big else 7.5)
    for j,label in enumerate(ax.get_yticklabels()):
        if selected[j] in ORACLES:label.set_fontweight("bold")
    ax.set_xticks(list(range(8))+list(8.1+np.arange(6))+[14.55],
                  MLABELS+DSNAMES+["Fast-8\nmean"],
                  fontsize=8.9 if not big else 7.3)
    ax.xaxis.tick_top();ax.tick_params(axis="x",length=0,pad=10,top=True)
    ax.tick_params(axis="y",length=0,pad=10)
    for sp in ax.spines.values():sp.set_visible(False)
    # Headers positioned relative to page; avoid clashing with column labels
    fig.text(.15 if not big else .185,.855 if not big else .925,
             "BATCH MIXING",weight="bold",fontsize=10,color="#3c6179")
    fig.text(.26 if not big else .31,.855 if not big else .925,
             "BIOLOGICAL CONSERVATION",weight="bold",fontsize=10,color="#3c6179")
    fig.text(.60,.855 if not big else .925,"SIX DATASETS",
             weight="bold",fontsize=10,color="#3c6179")
    fig.text(.87,.855 if not big else .925,"SCORE",weight="bold",fontsize=10,color="#3c6179")
    fig.text(.10 if not big else .115,.966 if not big else .98,
             "Batch integration | Original-score fast-8 (PCR excluded)",
             fontsize=20 if not big else 17,fontweight="bold",color="#17334e")
    fig.text(.10 if not big else .115,.934 if not big else .956,
             "13 Oct 2026 archived scaled scores | Numeric bubbles for all 15 columns | One fit per method/dataset",
             fontsize=10 if not big else 8.7,color="#526e7d")
    fig.text(.10 if not big else .115,.115 if not big else .054,
             "Historical score convention: scaled metric values, N/A scored as 0 in the composite. "
             "5/6 marks F1 inapplicable on DKD.\n"
             "BERNN historical archive uses selected FULL-LABEL fits (VAE configurations differ): "
             "not a matched five-split evaluation.\n"
             "Oracle is diagnostic only. Per-dataset and overall means were recomputed after excluding PCR.",
             fontsize=8.5 if not big else 7.5,color="#425c6b",va="top")
    sm=ScalarMappable(norm=Normalize(0,1),cmap=cmap)
    sm.set_array([])
    cax=fig.add_axes([.947,.21 if not big else .2,.012,.38 if not big else .50])
    fig.colorbar(sm,cax=cax,label="Archived scaled score (0–1)",ticks=np.arange(0,1.01,.25))
    for ext in ("png","pdf"):
        fig.savefig(OUT/(filename+"."+ext),dpi=190 if not big else 175,bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metrics,keyed=source()
    summary=audit(metrics,keyed)
    draw(metrics,summary,BASELINE,"Figure2_fast8_original_archive_controls_without_PCR")
    draw(metrics,summary,FULL,"Supplement_fast8_original_archive_all_27_without_PCR",big=True)
    print("Original 13-method figure and full 27-entry supplementary figure saved.")
    for method in BASELINE:
        q=summary[summary.method==method].iloc[0]
        print(f"{NAME[method]:22s}  Fast8 historical={q.fast8_archive_na0:.4f}"
              f"  NA-excluded={q.fast8_exclude_missing:.4f}"
              f"  source_old_fast9={q.source_fast9_score:.4f}")
    old=float(summary.loc[summary.method=="bernn","source_fast9_score"].iloc[0])
    print("WARNING: Archived BERNN source fast9 mean:",old,
          "while preserved historical PDF displays 0.761."
          " Do not claim to recover this exact PDF's BERNN source configuration.")

if __name__=="__main__": main()
