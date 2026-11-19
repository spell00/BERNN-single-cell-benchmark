#!/usr/bin/env python3
"""Make publication figures from frozen numerical evidence, without training.

Uncorrected raw values are never silently transformed to OpenProblems-scaled values.
NaN/structural N/A are excluded from descriptive averages (not converted to zero).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
OUTPUT=ROOT/"figures"
DATASETS=["dkd","gtex_v9","hypomap","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens"]
LABELS=["DKD","GTEx v9","HypoMap","Immune Cell Atlas","Mouse Pancreas","Tabula Sapiens"]
METHODS=["bernn","combat","scanvi","scvi","harmony","batchelor_fastmnn","liger","scalex"]
DISPLAY={"bernn":"BERNN","combat":"ComBat","scanvi":"scANVI","scvi":"scVI",
    "harmony":"Harmony","batchelor_fastmnn":"fastMNN","liger":"LIGER","scalex":"SCALEX"}
METRICS=["asw_batch_sampled","ilisi","pcr","cell_cycle_conservation",
         "ari","nmi","graph_connectivity","isolated_label_f1","clisi"]
METRIC_LABELS=["Batch ASW*","iLISI","PCR","Cell cycle","ARI","NMI","Graph conn.","Isolated F1","cLISI"]

def load_fast9():
    d=pd.read_csv(DATA/"matched/fast9_seed42_raw.csv",keep_default_na=False)
    keys=["dataset","method","seed","metric"]
    if d.duplicated(keys).any():raise ValueError("Duplicated method/dataset/seed/metric")
    assert len(d)==len(METHODS)*len(DATASETS)*len(METRICS),len(d)
    assert set(d.method)==set(METHODS)
    assert set(d.dataset)==set(DATASETS)
    assert set(d.metric)==set(METRICS)
    assert set(d.seed)=={42}
    valid=d[d.state=="ok"].copy()
    valid["score"]=valid.raw_score.astype(float)
    if not valid.score.between(0,1).all():raise ValueError("Metric out of the expected [0,1] range")
    invalid=d[d.state!="ok"]
    if not ((invalid.metric=="isolated_label_f1")&(invalid.dataset=="dkd")&(invalid.state=="not_applicable")).all():
       raise ValueError("Unexpected missing metrics")
    return d,valid

def main_fast9():
    full,valid=load_fast9()
    stat=valid.groupby(["method","metric"],as_index=False).score.agg(["mean","count"])
    agg=valid.groupby(["method","metric"]).score.mean()
    dataset_mean=valid.groupby(["method","dataset"]).score.mean()
    means=valid.groupby("method").score.mean().sort_values(ascending=False)
    # Recompute equal-weight mean over six datasets, then per-metric eligible scores.
    score_by_method={m:float(dataset_mean.loc[m].mean()) for m in METHODS}
    fig,ax=plt.subplots(figsize=(17.3,7.2))
    fig.subplots_adjust(left=.17,right=.93,top=.78,bottom=.18)
    cmap=plt.colormaps["viridis"]
    for yi,m in enumerate(METHODS):
        for xi,metric in enumerate(METRICS):
            if (m,metric) not in agg.index:
                ax.text(xi,yi,"N/A",fontsize=8,ha="center",va="center",color="#8c4b3e")
                continue
            v=float(agg[(m,metric)])
            ax.scatter(xi,yi,s=55+490*v,c=[v],cmap=cmap,vmin=0,vmax=1,lw=.8,
                       edgecolor="white",zorder=3)
            n=int(((valid.method==m)&(valid.metric==metric)).sum())
            if n!=6:
                ax.text(xi+.32,yi-.27,f"{n}/6",fontsize=7.5,ha="center",color="#aa5d42")
        for di,ds in enumerate(DATASETS):
            val=float(dataset_mean[(m,ds)])
            ax.text(9+di,yi,f"{val:.2f}",fontsize=10.3,ha="center",va="center",
                    fontweight="bold" if m=="bernn" else "normal",color="#263d50")
        ax.text(15.4,yi,f"{score_by_method[m]:.3f}",fontsize=11.3,ha="center",
                va="center",fontweight="bold",color="#183e5b")
    ax.set_xlim(-.6,16.15);ax.set_ylim(-.75,len(METHODS)-.4)
    ax.invert_yaxis()
    ax.set_yticks(range(len(METHODS)),[DISPLAY[m] for m in METHODS],fontsize=12)
    ax.set_xticks(list(range(9))+list(range(9,15))+[15.4],
                  METRIC_LABELS+[x.replace(" ","\n") for x in LABELS]+["Raw\nmean"],
                  fontsize=9,rotation=0)
    ax.tick_params(axis="x",length=0,pad=13)
    ax.tick_params(axis="y",length=0,pad=8)
    for xi in [-.5,2.5,8.5,14.55,15.95]:
        ax.axvline(xi,linewidth=1.0,color="#c3ccd3",zorder=0)
    for yi in np.arange(.5,8,.98):
        ax.axhline(yi,linewidth=.5,color="#e4eaf0",zorder=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.17,.875,"BATCH ALIGNMENT",fontsize=11,weight="bold",color="#174c6f")
    fig.text(.36,.875,"BIOLOGICAL CONSERVATION",fontsize=11,weight="bold",color="#174c6f")
    fig.text(.655,.875,"PER-DATASET RAW MEAN",fontsize=11,weight="bold",color="#174c6f")
    fig.text(.868,.875,"SUMMARY",fontsize=11,weight="bold",color="#174c6f")
    fig.text(.10,.963,"Single-cell integration | Matched fast-9 audit",fontsize=20,
             weight="bold",color="#133553")
    fig.text(.10,.925,"One common seed (42) per method and dataset, derived from persisted trained embeddings",
             fontsize=11,color="#42647a")
    sm=ScalarMappable(norm=Normalize(0,1),cmap=cmap);sm.set_array([])
    cax=fig.add_axes([.935,.235,.012,.45])
    fig.colorbar(sm,cax=cax,ticks=np.arange(0,1.01,.2),label="Raw metric (0–1)")
    fig.text(.08,.073,"*Batch ASW is a consistent stratified-sample approximation, not the exhaustive OpenProblems value.\n"
             "Isolated F1 is N/A on DKD (no eligible isolated cell types); 5/6 denotes five eligible datasets.\n"
             "Raw averages are descriptive, not official OpenProblems scores; methods have different training-information scopes.",
             fontsize=9,color="#435a6d",va="top")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf"]:
        fig.savefig(OUTPUT/f"Figure2_fast9_seed42_raw_audited.{ext}",dpi=230,bbox_inches="tight")
    plt.close(fig)
    audit_rows=[]
    for m in METHODS:
        for metric in METRICS:
            vals=valid[(valid.method==m)&(valid.metric==metric)].score
            audit_rows.append({"method":m,"metric":metric,"n_eligible_datasets":len(vals),
                              "mean_raw":vals.mean() if len(vals) else np.nan})
    pd.DataFrame(audit_rows).to_csv(OUTPUT/"Figure2_metric_means_audit.csv",index=False)
    print("FIGURE 2; raw mean by method (no official ranking):",score_by_method)

def isolated():
    dat=pd.read_csv(DATA/"matched/isolated_f1_5seeds_bernn_scanvi.csv")
    assert dat.shape[0]==60
    show=DATASETS[1:]
    fig,axes=plt.subplots(3,2,figsize=(11.4,11.4),constrained_layout=False)
    axs=axes.flatten()
    for ax,ds in zip(axs,show):
        sub=dat[(dat.dataset==ds)&(dat.state=="ok")]
        d={m:sub[sub.method==m].set_index("seed").isolated_label_f1.astype(float) for m in ["bernn","scanvi"]}
        assert set(d["bernn"].index)==set(d["scanvi"].index)==set(range(42,47))
        for seed in range(42,47):
            ax.plot([0,1],[d["bernn"][seed],d["scanvi"][seed]],
                     color="#a4bacb",alpha=.65,lw=.9,zorder=1)
            ax.scatter([0,1],[d["bernn"][seed],d["scanvi"][seed]],
                      c=["#287aa8","#d17d31"],s=29,zorder=3)
        ax.plot([0,1],[d["bernn"].mean(),d["scanvi"].mean()],color="#1c3444",
                linestyle="--",marker="D",lw=1.1,markersize=7,zorder=4)
        ax.set_title(LABELS[DATASETS.index(ds)],loc="left",fontsize=13,fontweight="bold")
        ax.set_xticks([0,1],["BERNN","scANVI"])
        ax.set_yticks(np.arange(0,1.01,.2))
        ax.set_ylim(-.06,1.06);ax.set_xlim(-.35,1.35)
        ax.grid(axis="y",color="#e0e6eb")
        ax.set_ylabel("Isolated Label F1")
        for edge in ["top","right"]:ax.spines[edge].set_visible(False)
    axs[-1].axis("off")
    axs[-1].text(.03,.88,"Definition",fontsize=14,fontweight="bold",transform=axs[-1].transAxes)
    txt=("Clustering-based recovery of types\noccurring in few batches.\n\n"
       "5 independently fitted models per dataset.\n"
       "Lines connect the same seed, not\nmatched individual isolated labels.\n\n"
       "DKD: metric not applicable.\n\n"
       "BERNN: train-labels only (inductive).\n"
       "scANVI: train-labels plus\nunlabeled target expressions.")
    axs[-1].text(.03,.80,txt,fontsize=11,va="top",transform=axs[-1].transAxes)
    fig.suptitle("Isolated Label F1 | BERNN vs scANVI across five fitted splits",
                 fontsize=17,fontweight="bold",y=.985)
    fig.subplots_adjust(left=.10,right=.98,top=.925,bottom=.055,hspace=.38,wspace=.23)
    for ext in ["png","pdf"]:fig.savefig(OUTPUT/f"Supplement_Isolated_F1_five_splits.{ext}",dpi=230,bbox_inches="tight")
    plt.close(fig)
    summary=dat[dat.state=="ok"].groupby(["dataset","method"]).isolated_label_f1.agg(["count","mean","std","min","max"]).reset_index()
    summary.to_csv(OUTPUT/"Supplement_Isolated_F1_five_splits_summary.csv",index=False)
    print(summary.to_string(index=False))

def mcc():
    df=pd.read_csv(DATA/"matched/split_aligned_mcc_summary.csv")
    order=["raw_no_integration","scalex","liger","batchelor_fastmnn","combat","harmony","scvi","scanvi","bernn"]
    seen=df[df.method.isin(order)]
    assert seen.groupby("dataset").method.nunique().to_dict()=={d:9 for d in DATASETS}
    assert seen.n_paired_seeds.astype(int).eq(5).all()
    fig,axs=plt.subplots(3,2,figsize=(12,10))
    for ax,ds in zip(axs.flatten(),DATASETS):
        g=seen[seen.dataset==ds].set_index("method")
        ys=np.arange(len(order))[::-1]
        for y,m in zip(ys,order):
            v=g.loc[m]
            col=("#247aaa" if m=="bernn" else "#c77c22" if m=="raw_no_integration" else "#566e80")
            marker="D" if m=="bernn" else "s" if m=="raw_no_integration" else "o"
            ax.errorbar(float(v.test_mcc_mean),y,xerr=float(v.test_mcc_sd),fmt=marker,ms=5.7,
                        color=col,ecolor=col,elinewidth=1.25,capsize=2.4)
        ax.set_yticks(ys,[{"raw_no_integration":"No integration","batchelor_fastmnn":"fastMNN"}.get(m,DISPLAY.get(m,m)) for m in order],fontsize=9)
        ax.set_xlim(.12,1.01);ax.set_xticks(np.arange(.2,1.01,.1));ax.tick_params(axis="x",labelbottom=True,labelsize=9)
        ax.set_xlabel("Held-out-batch test MCC");ax.set_title(LABELS[DATASETS.index(ds)],loc="left",fontweight="bold")
        ax.grid(axis="x",color="#e4e9ed")
        for edge in ["top","right"]:ax.spines[edge].set_visible(False)
    fig.suptitle("Held-out-batch prediction | Five splits per method",fontsize=17,fontweight="bold",y=.984)
    fig.text(.5,.945,"Mean ± split-level SD. MCC uses five group-disjoint evaluation splits; isolated F1 in Fig. 2 does not.",
             ha="center",fontsize=10)
    fig.subplots_adjust(top=.90,bottom=.065,left=.15,right=.97,hspace=.3,wspace=.25)
    for ext in ["png","pdf"]:fig.savefig(OUTPUT/f"Figure3_MCC_five_splits.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)

def composition():
    d=pd.read_csv(DATA/"matched/fig1_celltype_batch_counts.csv")
    summaries=[]
    for ds,g in d.groupby("dataset"):
        p=g.pivot_table(index="batch",columns="cell_type",values="count",aggfunc="sum",fill_value=0)
        n=p.sum(axis=1).to_numpy(float);pb=n/n.sum()
        q=p.to_numpy(float)/n[:,None];glob=(pb[:,None]*q).sum(axis=0)
        tv=.5*np.sum(pb[:,None]*abs(q-glob))
        summaries.append({"dataset":ds,"cells":int(n.sum()),"batches":len(n),
            "largest_batch_fraction":pb.max(),"cell_weighted_tv":tv,
            "effective_batches":1/np.sum(pb**2)})
    summary=pd.DataFrame(summaries).sort_values("dataset")
    summary.to_csv(OUTPUT/"Figure1_dataset_composition_stats.csv",index=False)
    fig,ax=plt.subplots(figsize=(8.5,5.7))
    for _,r in summary.iterrows():
        ax.scatter(r.largest_batch_fraction,r.cell_weighted_tv,s=80)
        ax.annotate(LABELS[DATASETS.index(r.dataset)],(r.largest_batch_fraction,r.cell_weighted_tv),
                    xytext=(6,5),textcoords="offset points",fontsize=10)
    ax.set_xlabel("Largest batch fraction");ax.set_ylabel("Cell-weighted compositional TV")
    ax.set_title("Dataset composition before integration",fontsize=15,fontweight="bold")
    ax.set_xlim(0,1.02);ax.set_ylim(0,.8)
    ax.grid(color="#e1e9ee",zorder=-2)
    for edge in ["top","right"]:ax.spines[edge].set_visible(False)
    fig.tight_layout()
    for ext in ["png","pdf"]:fig.savefig(OUTPUT/f"Figure1_composition_panelA.{ext}",dpi=230,bbox_inches="tight")
    plt.close(fig)

def historical():
    source=DATA/"archived/fast9_historical_long.csv"
    d=pd.read_csv(source)
    methods=["bernn","combat","scanvi","scvi","harmony","batchelor_fastmnn","liger","scalex","uce","no_integration","bbknn","scanorama"]
    names={**DISPLAY,"uce":"UCE","no_integration":"No integration","bbknn":"BBKNN","scanorama":"Scanorama"}
    names.update({"asw_batch_sampled":"Batch ASW"})
    ms=["asw_batch","ilisi","pcr","cell_cycle_conservation","ari","nmi","graph_connectivity","isolated_label_f1","clisi"]
    means=d[(d.method.isin(methods))&(d.status=="ok")].groupby(["method","metric"]).reference_scaled_score.mean()
    fig,ax=plt.subplots(figsize=(12,8))
    for i,m in enumerate(methods):
      for j,key in enumerate(ms):
        if (m,key) not in means.index:ax.text(j,i,"N/A",ha="center",va="center",fontsize=7,color="#ae5b4a")
        else:
          v=float(means[(m,key)])
          ax.scatter(j,i,s=40+360*v,c=[v],cmap="viridis",vmin=0,vmax=1,edgecolor="white",lw=.4)
          cov=len(d[(d.method==m)&(d.metric==key)&(d.status=="ok")])
          if cov<6:ax.text(j+.31,i-.25,f"{cov}/6",fontsize=7,color="#a85b43",ha="center")
    ax.set_xticks(range(9),METRIC_LABELS,fontsize=9)
    ax.set_yticks(range(len(methods)),[names[m] for m in methods],fontsize=10)
    ax.set_ylim(-.75,len(methods)-.3);ax.invert_yaxis();ax.set_xlim(-.5,8.6)
    fig.suptitle("ARCHIVAL ONLY | Historical fast-nine score coverage",fontsize=17,fontweight="bold")
    fig.text(.1,.035,"Historical runs have different label-access and fit-scope contracts. BERNN was fitted with full-data labels; VAE settings may differ by atlas.\n"
          "Figure is NOT matched to held-out MCC and is NOT evidence for superiority under new batches.",fontsize=9)
    fig.subplots_adjust(left=.18,right=.98,top=.91,bottom=.13)
    for ext in ["png","pdf"]:fig.savefig(OUTPUT/f"Supplement_legacy_fast9_not_matched.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--figure",default="all",choices=["all","fast9","isolated","mcc","composition","historical"])
    args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    funcs={"fast9":main_fast9,"isolated":isolated,"mcc":mcc,"composition":composition,"historical":historical}
    for name,func in funcs.items():
        if args.figure in ("all",name):
            print("RUN",name,flush=True);func()
