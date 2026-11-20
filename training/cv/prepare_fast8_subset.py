#!/usr/bin/env python3
"""Prepare STRICT fold-only cell subsets, ensuring all neighbors/clusters can be recomputed."""
import argparse,json
from pathlib import Path
import numpy as np
import anndata as ad
p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,required=True)
p.add_argument('--data-dir',type=Path,required=True)
p.add_argument('--scope',choices=['test','valid'],default='test')
p.add_argument('--destination',type=Path,required=True)
a=p.parse_args()
base=a.base.resolve();s=json.loads((base.parents[2]/'bernn'/base.parents[1].name/base.name/'split.json').read_text()) if False else None
# The scANVI comparator stores its manifest under BERNN/<seed>/<dataset>.
if '/comparators/' in str(base):
    parts=base.parts; i=parts.index('comparators'); project=Path(*parts[:i]);seed=parts[i+1];ds=parts[i+3]
    split=project/'bernn'/seed/ds/'split.json'
else:
    split=base/'split.json';ds=base.name
s=json.loads(split.read_text())
assert s['dataset']==ds
src=a.data_dir/ds/'log_cp10k'
dst=a.destination.resolve()
dst.mkdir(parents=True,exist_ok=True)
ready=dst/'subset_ready.json'
if ready.exists():
    print('SUBSETS_ALREADY_READY',dst,flush=True);exit(0)
integ=ad.read_h5ad(base/'integrated.h5ad')
data=ad.read_h5ad(src/'dataset.h5ad')
sol=ad.read_h5ad(src/'solution.h5ad')
assert integ.obs_names.equals(data.obs_names) and data.obs_names.equals(sol.obs_names), 'Cell IDs mismatch'
idx=np.flatnonzero(np.isin(data.obs['batch'].astype(str).to_numpy(), s['batch_lists'][a.scope]))
assert idx.size==s['n_cells'][a.scope],(idx.size,s['n_cells'][a.scope])
assert len(np.unique(data.obs['batch'].astype(str).to_numpy()[idx]))==len(s['batch_lists'][a.scope])
for source,name in [(integ,'integrated'),(data,'dataset'),(sol,'solution')]:
    output=dst/(name+'_only.h5ad')
    if not output.exists():
        # .to_memory() materializes just the selected cells, never the full H5AD matrix.
        sub=source[idx,:].copy()
        # CRITICAL: a scANVI input has a full-atlas neighbors graph already.
        # Discard its sliced graph and force graph computation on TEST ONLY.
        if name=='integrated':
            sub.uns.pop('neighbors',None)
            for key in list(sub.obsp):del sub.obsp[key]
            for key in list(sub.obsm):
                if key!='X_emb':del sub.obsm[key]
            assert not sub.obsp and 'neighbors' not in sub.uns
        assert sub.n_obs==len(idx)
        assert sub.obs_names.equals(data.obs_names[idx])
        sub.write_h5ad(output,compression='gzip')
        print('WROTE',output,sub.shape,flush=True)
# Output boundaries are exact: metric components see ONLY these cells.
ready.write_text(json.dumps({'scope':a.scope,'dataset':ds,'n_cells':len(idx),
 'n_batches':len(s['batch_lists'][a.scope]), 'batch_lists':s['batch_lists'][a.scope],
 'neighbor_graph':'MUST be rebuilt on subset, never slice full-atlas graph',
 'selection':'validation labels used for checkpoint selection; test labels only for scoring'},indent=2))
print('SUBSET_READY',dst,flush=True)
