#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import anndata as ad,h5py,numpy as np,pandas as pd,joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import matthews_corrcoef
from anndata.experimental import read_elem
p=argparse.ArgumentParser();p.add_argument("--integrated",required=True);p.add_argument("--input",required=True);p.add_argument("--split",required=True);p.add_argument("--out",required=True);p.add_argument("--method",required=True);p.add_argument("--native-output",default=None);a=p.parse_args()
t=time.time();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
# The in-flight comparator worker does not pass the new optional argument.
# Native predictions live in the same scANVI output artifact, so autodetect.
if a.method=="scanvi" and a.native_output is None:
 a.native_output=str(out/"method_output.h5ad")

with h5py.File(a.input) as f:obs=read_elem(f["obs"])
x=ad.read_h5ad(a.integrated);assert x.obs_names.equals(obs.index)
if "X_emb" not in x.obsm:raise ValueError("No embedding; graph-only requires separate documented predictor")
z=x.obsm["X_emb"];assert np.isfinite(z).all()
s=json.loads(Path(a.split).read_text());idx={k:np.flatnonzero(np.isin(obs.batch.astype(str),v)) for k,v in s["batch_lists"].items()}
y=obs.cell_type.astype(str).to_numpy();tr=idx["train"];seed=s["seed"]
native_pred=None
if a.native_output:
 if a.method!="scanvi":raise ValueError("--native-output is exclusively for scANVI")
 with h5py.File(a.native_output) as f:
  native_obs=read_elem(f["obs"])
 if not native_obs.index.equals(obs.index):raise ValueError("Native prediction cell IDs do not align with input cells")
 if "scanvi_native_prediction" not in native_obs.columns:raise ValueError("Missing trained scANVI native predictions")
 native_pred=native_obs["scanvi_native_prediction"].astype(str).to_numpy()
 if len(native_pred)!=len(obs):raise ValueError("Native prediction length mismatch")
 if np.any(pd.isna(native_pred)) or np.any(native_pred=="UnknownUnknown"):raise ValueError("Invalid native predictions")

pc=None
if z.shape[1]>50:
 pc=PCA(n_components=min(50,len(tr)-1),svd_solver="randomized",random_state=seed).fit(z[tr]);z=pc.transform(z)
scale=StandardScaler().fit(z[tr]);z=scale.transform(z)
m=SGDClassifier(loss="log_loss",class_weight="balanced",alpha=1e-4,random_state=seed,max_iter=600,tol=1e-3).fit(z[tr],y[tr])
pred=m.predict(z).astype(str)
fit_scope="transductive_unlabeled_target_expression; training_labels_only" if a.method=="scanvi" else "transductive_unsupervised_full_expression"
known=set(y[tr]);coverage={k:float(np.isin(y[v],list(known)).mean()) for k,v in idx.items()}
r={"method":a.method,"dataset":s["dataset"],"seed":seed,"fit_scope":fit_scope,"probe_seconds":time.time()-t,
 "common_probe":{k:float(matthews_corrcoef(y[v],pred[v])) for k,v in idx.items()},"n_cells":s["n_cells"],"label_coverage":coverage}
if native_pred is not None:
 r["native"]={k:float(matthews_corrcoef(y[v],native_pred[v])) for k,v in idx.items()}
 r["native_predictor"]="scANVI.predict; trained labels from train batches only; unsupervised expressions from all batches"
(out/"scores.json").write_text(json.dumps(r,indent=2))
sp=np.empty(len(obs),dtype=object)
for k,v in idx.items():sp[v]=k
predictions={"cell_id":obs.index,"batch":obs.batch.astype(str),"split":sp,"true_label":y,"prediction":pred}
if native_pred is not None:predictions["native_prediction"]=native_pred
pd.DataFrame(predictions).to_csv(out/"predictions.csv.gz",index=False)
joblib.dump({"pca":pc,"scaler":scale,"probe":m},out/"probe.joblib")
print(json.dumps(r),flush=True)
