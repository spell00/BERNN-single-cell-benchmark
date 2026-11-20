#!/usr/bin/env python3
"""Reproduce the bounded batch ASW approximation on the fold-only cells."""
import argparse,json
from pathlib import Path
import numpy as np,anndata as ad
from sklearn.metrics import silhouette_samples
p=argparse.ArgumentParser();p.add_argument('--subset',type=Path,required=True)
a=p.parse_args();d=a.subset
obs=ad.read_h5ad(d/'solution_only.h5ad',backed='r').obs
z=np.asarray(ad.read_h5ad(d/'integrated_only.h5ad',backed='r').obsm['X_emb'])
assert len(z)==len(obs)
assert np.isfinite(z).all()
labels=obs.cell_type.astype(str).to_numpy();batch=obs.batch.astype(str).to_numpy()
rng=np.random.default_rng(42)
out=[]
for label in sorted(set(labels)):
    candidates=np.flatnonzero(labels==label)
    nb={x:int(np.count_nonzero(batch[candidates]==x)) for x in sorted(set(batch[candidates]))}
    if len(nb)<2:continue
    if len(candidates)<=400:chosen=candidates.tolist()
    else:
        chosen=[]
        for b in sorted(nb):
            possibles=candidates[batch[candidates]==b]
            chosen.extend(rng.choice(possibles,size=min(2,len(possibles)),replace=False).tolist())
        if len(chosen)>400:chosen=rng.choice(np.asarray(chosen),400,replace=False).tolist()
        remaining=400-len(chosen)
        if remaining>0:
            other=np.setdiff1d(candidates,np.asarray(chosen,dtype=int),assume_unique=True)
            chosen.extend(rng.choice(other,size=remaining,replace=False).tolist())
    ix=np.asarray(sorted(set(chosen)),dtype=int)
    if len(set(batch[ix]))<2 or len(set(batch[ix]))>=len(ix):continue
    asw=silhouette_samples(z[ix],batch[ix],metric='euclidean')
    out.append({'cell_type':label,'n_cells':len(ix),'n_batches':len(set(batch[ix])),
                'score':float(np.mean(1-np.abs(asw)))})
value=float(np.mean([v['score'] for v in out])) if out else None
(d/'asw_batch_sampled.json').write_text(json.dumps({
 'metric':'asw_batch_sampled','value':value,'per_type':out,
 'scope':'only selected fold cells; all method embeddings use identical deterministic expression-independent stratification',
 'note':'bounded approximation, not exact all-cell scIB batch ASW'},indent=2))
print('ASW_BATCH_SAMPLED',value,flush=True)
