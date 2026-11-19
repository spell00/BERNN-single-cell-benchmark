#!/usr/bin/env python3
"""Read existing saved outputs only; no expensive computations or imputations."""
from pathlib import Path
import json,sys
import anndata as ad
import numpy as np
import pandas as pd
ROOT=Path("/home/simonp/task_batch_integration_bernn/experiments/single_cell_paper_20261019")
OUTPUT=ROOT/"reports";OUTPUT.mkdir(exist_ok=True)
D=["dkd","gtex_v9","hypomap","mouse_pancreas_atlas","immune_cell_atlas","tabula_sapiens"]
C=["combat","harmony","harmonypy","batchelor_fastmnn","liger","pyliger","scanvi","scvi","scalex","uce"]
F=["clustering_overlap","graph_connectivity","cell_cycle_conservation","pcr","lisi","isolated_label_f1","bras"]
imbalance=ROOT/"reference/class_batch_imbalance_all6.csv"
imbalance_by_ds=pd.read_csv(imbalance).set_index("dataset") if imbalance.exists() else None
rows=[]
def get_metrics(base):
    got={}
    for f in F:
        inp=base/"metrics"/(f+".h5ad")
        if not inp.exists():continue
        try:
            uns=ad.read_h5ad(inp,backed="r").uns
            for metric,value in zip(uns["metric_ids"],uns["metric_values"]):
                val=float(value)
                got[metric]=val if np.isfinite(val) else None
        except Exception as e:
            print("BAD_METRIC",inp,repr(e),file=sys.stderr)
    sampled=base/"metrics/asw_batch_sampled.json"
    if sampled.exists():
        record=json.loads(sampled.read_text())
        got["asw_batch_sampled"]=record.get("value")
        got["asw_batch_n_sampled"]=record.get("evaluated_cells")
    return got
def add(method,seed,ds,base,category):
    f=base/"scores.json"
    if not f.is_file():return
    payload=json.loads(f.read_text())
    row={"method":method,"dataset":ds,"seed":seed,"category":category}
    mcc=payload.get("common_probe")
    if not mcc:return
    row.update({"mcc_"+key:mcc.get(key) for key in ("train","valid","test")})
    if method=="bernn":row.update({"native_mcc_"+key:payload["native"].get(key) for key in ("train","valid","test")})
    if imbalance_by_ds is not None:
        im=imbalance_by_ds.loc[ds]
        row["nmi_celltype_batch_original"]=im["nmi_celltype_batch"]
        row["largest_batch_fraction"]=im["largest_batch_fraction"]
    row.update(get_metrics(base))
    rows.append(row)
for seed in range(42,47):
 for ds in D:add("bernn",seed,ds,ROOT/"bernn"/str(seed)/ds,"inductive_embedding")
for ds in D:
 f=ROOT/"bernn/42"/ds/"scores.json"
 if f.is_file():
  s=json.loads(f.read_text())
  rows.append({"method":"raw_unintegrated","dataset":ds,"seed":42,"category":"raw_trainonly_pca",
               **{"mcc_"+key:s["raw_no_integration_probe"].get(key) for key in ("train","valid","test")}})
 for method in C:add(method,42,ds,ROOT/"comparators/42"/method/ds,"transductive_embedding")
df=pd.DataFrame(rows)
first=["dataset","method","seed","category","mcc_train","mcc_valid","mcc_test","native_mcc_train","native_mcc_valid","native_mcc_test",
       "bras","asw_batch_sampled","ari","nmi","ilisi","clisi","pcr","graph_connectivity","cell_cycle_conservation",
       "isolated_label_f1","ari_batch","nmi_batch","nmi_celltype_batch_original","largest_batch_fraction","asw_batch_n_sampled"]
for col in first:
 if col not in df:df[col]=np.nan
df=df[first].sort_values(["dataset","method","seed"])
df.to_csv(OUTPUT/"mcc_and_integration_metrics_current.csv",index=False)
df[df.method.eq("bernn")].to_csv(OUTPUT/"bernn_mcc_and_integration_metrics_all_seeds.csv",index=False)
print("SAVED",len(df),"rows",OUTPUT/"mcc_and_integration_metrics_current.csv")
for _,x in df[(df.method=="bernn")&(df.seed==42)].iterrows():
 print(x["dataset"],"mcc_test",round(x["mcc_test"],4), "BRAS",x["bras"],"ARI",x["ari"],"iLISI",x["ilisi"],"PCR",x["pcr"],"ASW_sampled",x["asw_batch_sampled"])
