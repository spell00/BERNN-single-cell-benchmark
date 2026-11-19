#!/usr/bin/env python3
"""Dataset composition audit: recreate the four-panel main Figure 1 from count data."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COUNTS=pd.read_csv(ROOT/"data/matched/fig1_celltype_batch_counts.csv")
OUT=ROOT/"figures"
LABELS={
"dkd":"DKD","gtex_v9":"GTEx v9","hypomap":"HypoMap",
"immune_cell_atlas":"Immune Cell Atlas","mouse_pancreas_atlas":"Mouse Pancreas",
"tabula_sapiens":"Tabula Sapiens"}

def composition(ds,max_batches=12,max_types=6):
    a=COUNTS[COUNTS.dataset==ds]
    p=a.pivot_table(index="batch",columns="cell_type",values="count",aggfunc="sum",fill_value=0)
    # Keep largest batches and the most abundant cell types in the whole dataset.
    major=p.sum().sort_values(ascending=False).head(max_types).index.tolist()
    top=p.sum(axis=1).sort_values(ascending=False).head(max_batches).index
    v=p.loc[top,major].copy()
    v["All other types"]=p.loc[top].sum(axis=1)-v.sum(axis=1)
    fr=v.div(v.sum(axis=1),axis=0)
    return fr, p

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    totals=[]
    for ds in LABELS:
        _,p=composition(ds)
        cells=p.sum(axis=1).to_numpy(float)
        batch_fraction=cells/cells.sum()
        cell_global=p.sum(axis=0).to_numpy(float)/cells.sum()
        inside=p.to_numpy(float)/cells[:,None]
        tv=.5*np.sum(batch_fraction[:,None]*np.abs(inside-cell_global))
        totals.append({"dataset":ds,"largest":float(batch_fraction.max()),"tv":float(tv)})
    stats=pd.DataFrame(totals)
    fig=plt.figure(figsize=(15,10.8),layout="constrained")
    gs=fig.add_gridspec(2,2,height_ratios=[.9,1.2],width_ratios=[1.08,.92])
    ax1=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1])
    ax3=fig.add_subplot(gs[1,0]);ax4=fig.add_subplot(gs[1,1])
    fig.suptitle("Cell-type composition varies strongly between experimental batches",fontsize=19,fontweight="bold")
    for i,r in stats.iterrows():
        ax1.scatter(r.largest,r.tv,s=90)
        ax1.annotate(LABELS[r.dataset],(r.largest,r.tv),xytext=(6,5),
                    textcoords="offset points",fontsize=10)
    ax1.set_xlim(0,1.04);ax1.set_ylim(0,.8)
    ax1.set_xlabel("Largest batch fraction");ax1.set_ylabel("Cell-weighted compositional TV")
    ax1.set_title("A. Batch size versus biological composition",loc="left",fontweight="bold")
    ax1.grid(color="#e3e9ec")
    hyp=COUNTS[COUNTS.dataset=="hypomap"]
    batch_counts=hyp.groupby("batch")["count"].sum().sort_values(ascending=False)
    plotbatch=batch_counts.head(5)[::-1]
    ax2.barh(np.arange(len(plotbatch)),plotbatch.values/batch_counts.sum(),color="#3e83ad")
    ax2.set_yticks(range(len(plotbatch)),[str(x)[:28] for x in plotbatch.index],fontsize=9)
    ax2.set_xlim(0,1)
    ax2.set_xlabel("Fraction of HypoMap cells")
    ax2.set_title("B. HypoMap: dominance of largest batch",loc="left",fontweight="bold")
    for j,v in enumerate(plotbatch.values/batch_counts.sum()):
        ax2.text(v+.009,j,f"{v:.1%}",fontsize=9,va="center")
    for ax,ds,letter in ((ax3,"gtex_v9","C"),(ax4,"dkd","D")):
        share,_=composition(ds,max_batches=12,max_types=6)
        cmap=plt.colormaps["tab20"]
        bottom=np.zeros(len(share))
        for i,key in enumerate(share.columns):
            arr=share[key].to_numpy()
            ax.bar(np.arange(len(arr)),arr,bottom=bottom,width=.84,
                color=cmap(i),label=str(key).replace(" epithelial"," epit.").replace(" cell",""),linewidth=0)
            bottom+=arr
        ax.set_ylim(0,1.02);ax.set_ylabel("Cell-type fraction")
        ax.set_xticks(np.arange(len(share)),[str(x)[:14] for x in share.index],rotation=70,ha="right",fontsize=7.5)
        ax.set_title(f"{letter}. {LABELS[ds]}: largest batches",loc="left",fontweight="bold")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.30),ncol=3,
                  fontsize=8,frameon=False)
        ax.grid(axis="y",color="#e9eef1",zorder=-3)
    for ax in [ax1,ax2,ax3,ax4]:
        for edge in ["top","right"]:ax.spines[edge].set_visible(False)
    fig.suptitle("Dataset composition audit before integration",y=1.02,fontsize=18,fontweight="bold")
    for ext in ("pdf","png"):fig.savefig(OUT/f"Figure1_dataset_composition_full.{ext}",dpi=230,bbox_inches="tight")
    plt.close(fig)
    print("REPRODUCED Figure1_dataset_composition_full")

if __name__=="__main__":main()
