#!/usr/bin/env python3
"""Regenerate GTEx v9 10k-sample UMAP panels from frozen public coordinates.

Coordinate snapshots avoid retraining models or refitting UMAP and allow exact
figure recreation. Requires no original single-cell expression data.
"""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
HERE=Path(__file__).resolve().parents[1]
VIS=HERE/"data/visualization"
OUT=HERE/"figures"
METHODS=["raw","bernn","combat","harmony","harmonypy","batchelor_fastmnn","liger","pyliger","scvi","scanvi","scalex"]
NAMES={"raw":"Raw","bernn":"BERNN","combat":"ComBat","harmony":"Harmony",
"harmonypy":"HarmonyPy","batchelor_fastmnn":"fastMNN","liger":"LIGER","pyliger":"pyLIGER",
"scvi":"scVI","scanvi":"scANVI","scalex":"SCALEX"}

a=pd.read_csv(VIS/"gtex_sampled_cell_type_and_batch.csv")
assert len(a)==10000
cells=a.cell_type.to_numpy()
batch=a.batch.to_numpy()
target="basal epithelial cell of tracheobronchial tree"
target_sel=cells==target
assert target_sel.sum()>0
top9=a.cell_type.value_counts().index[:9].tolist()
bcats=list(pd.unique(batch))
subbatch=list(pd.unique(batch[target_sel]))
colors1={j:plt.colormaps["tab20"](i%20) for i,j in enumerate(bcats)}
colors2={j:plt.colormaps["tab10"](i%10) for i,j in enumerate(top9)}
colors3={j:plt.colormaps["tab10"](i%10) for i,j in enumerate(subbatch)}

def makefig(method):
    f=VIS/f"gtex_umap_{method}_coords.npy"
    x=np.load(f)
    assert x.shape==(len(a),2)
    fig,axs=plt.subplots(1,3,figsize=(17,5.6))
    for cat in bcats:
      q=batch==cat
      axs[0].scatter(x[q,0],x[q,1],s=.9,c=[colors1[cat]],alpha=.32,rasterized=True,linewidth=0)
    axs[0].set_title("All cell types | batch",fontweight="bold")
    other=~np.isin(cells,top9)
    axs[1].scatter(x[other,0],x[other,1],s=.8,c="#aeb6bd",alpha=.22,rasterized=True,linewidth=0)
    for t in top9:
      q=cells==t
      axs[1].scatter(x[q,0],x[q,1],s=.95,c=[colors2[t]],alpha=.4,rasterized=True,linewidth=0)
    axs[1].set_title("Nine abundant cell types",fontweight="bold")
    axs[2].scatter(x[~target_sel,0],x[~target_sel,1],s=.8,c="#adb5bd",alpha=.1,rasterized=True,linewidth=0)
    for b in subbatch:
      q=target_sel&(batch==b)
      axs[2].scatter(x[q,0],x[q,1],s=3,c=[colors3[b]],alpha=.72,rasterized=True,linewidth=0)
    axs[2].set_title("Tracheobronchial basal cells only",fontweight="bold")
    leg=[Line2D([0],[0],ls="",marker="o",color=colors3[b],label=f"{b} (n={int(sum(target_sel&(batch==b)))})")
          for b in subbatch]
    axs[2].legend(handles=leg,fontsize=7.2,loc="best")
    for ax in axs:
      ax.set_xlabel("UMAP 1");ax.set_ylabel("UMAP 2")
      ax.tick_params(labelsize=8)
      for edge in ["right","top"]:ax.spines[edge].set_visible(False)
    fig.suptitle(f"GTEx v9 | {NAMES[method]} | 10,000 shared cells",fontweight="bold",fontsize=17)
    fig.text(.04,.02,"Per-method UMAP coordinates are independently fitted; no spatial alignment or held-out inference is implied.",
          fontsize=9,color="#3a5365")
    fig.subplots_adjust(left=.055,right=.98,top=.86,bottom=.14,wspace=.20)
    return fig

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with PdfPages(OUT/"Supplement_GTEx_all_methods_common10k_UMAP.pdf") as pdf:
      for m in METHODS:
        fig=makefig(m)
        pdf.savefig(fig)
        if m in ("raw","bernn","combat"):
            fig.savefig(OUT/f"GTEx_{m}_UMAP_common10k.png",dpi=150)
        plt.close(fig)
    # Main Figure 4: raw vs BERNN; paired batch and cell-type views.
    fig,axs=plt.subplots(2,2,figsize=(11.8,9.5))
    for row,m in enumerate(("raw","bernn")):
        x=np.load(VIS/f"gtex_umap_{m}_coords.npy")
        for b in bcats:
            sel=batch==b
            axs[row,0].scatter(x[sel,0],x[sel,1],s=.8,color=colors1[b],alpha=.32,rasterized=True,linewidth=0)
        other=~np.isin(cells,top9)
        axs[row,1].scatter(x[other,0],x[other,1],s=.8,color="#b0b7bf",alpha=.18,rasterized=True,linewidth=0)
        for t in top9:
            sel=cells==t
            axs[row,1].scatter(x[sel,0],x[sel,1],s=.9,color=colors2[t],alpha=.40,rasterized=True,linewidth=0)
        for col in range(2):
            axs[row,col].set_ylabel(f"{NAMES[m]} | UMAP 2")
            axs[row,col].set_xlabel("UMAP 1")
            axs[row,col].tick_params(labelsize=8)
            for edge in ["top","right"]:axs[row,col].spines[edge].set_visible(False)
    axs[0,0].set_title("Colored by batch",fontweight="bold")
    axs[0,1].set_title("Colored by cell type",fontweight="bold")
    fig.suptitle("GTEx v9 | Raw versus BERNN representation | 10,000 shared cells",fontsize=17,weight="bold")
    fig.text(.045,.025,"Full-data descriptive representation (not a held-out experiment). Independent UMAP maps; absolute axes differ.",fontsize=9)
    fig.subplots_adjust(left=.10,right=.97,top=.91,bottom=.07,hspace=.24,wspace=.21)
    for ext in ("pdf","png"):fig.savefig(OUT/f"Figure4_GTEx_raw_vs_BERNN_UMAP.{ext}",dpi=210)
    plt.close(fig)
    print("REPRODUCED 11-method GTEx UMAP supplement + main Figure 4")

if __name__=="__main__":main()
