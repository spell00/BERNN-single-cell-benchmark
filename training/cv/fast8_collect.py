#!/usr/bin/env python3
"""Collect full-test-only Fast8 raw metrics without silently imputing N/A."""
import argparse,json,math
from pathlib import Path
import numpy as np,anndata as ad
p=argparse.ArgumentParser();p.add_argument('--subset',type=Path,required=True)
a=p.parse_args();d=a.subset
info=json.loads((d/'subset_ready.json').read_text())
values={};failures={}
name_map={'clustering_overlap':['ari','nmi'],
          'lisi':['ilisi','clisi'],
          'graph_connectivity':['graph_connectivity'],
          'isolated_label_f1':['isolated_label_f1'],
          'cell_cycle_conservation':['cell_cycle_conservation']}
for component,names in name_map.items():
    path=d/(component+'.h5ad')
    if not path.exists():
        failures[component]='missing component output'
        for m in names:values[m]=None
        continue
    try:
        result=ad.read_h5ad(path)
        arr=list(result.uns['metric_values'])
        ids=[str(x) for x in result.uns['metric_ids']]
        by_id=dict(zip(ids,arr))
        for metric in names:
            v=by_id.get(metric)
            if v is None:
                values[metric]=None;failures[metric]='metric id absent'
            else:
                num=float(v)
                values[metric]=num if math.isfinite(num) else None
    except Exception as e:
        failures[component]=repr(e)
        for m in names:values[m]=None
asw=d/'asw_batch_sampled.json'
values['asw_batch_sampled']=json.loads(asw.read_text())['value'] if asw.exists() else None
if info['n_batches']<2:
    values['ilisi']=None;values['asw_batch_sampled']=None
    failures['batch_mixing']='one batch in this evaluation split; iLISI and batch ASW are undefined'
# Isolated F1 is often undefined on DKD when no isolated types are eligible.
if info['dataset']=='dkd' and values['isolated_label_f1'] is None:
    failures['isolated_label_f1']='no eligible isolated label types, or component failed; inspect log'
metric_order=['asw_batch_sampled','ilisi','cell_cycle_conservation','ari','nmi',
              'graph_connectivity','isolated_label_f1','clisi']
assert set(values)==set(metric_order),(values.keys(),metric_order)
eligible=[float(values[k]) for k in metric_order if values[k] is not None and 0<=float(values[k])<=1]
value=float(np.mean(eligible)) if eligible else None
output={'dataset':info['dataset'],'scope':info['scope'],'n_cells':info['n_cells'],
        'n_batches':info['n_batches'],'metrics':values,
        'raw_equal_weight_mean_eligible_only':value,'n_eligible_metrics':len(eligible),
        'n_possible_metrics':8,'failures':failures,
        'score_semantics':'Fast8 raw scores, graph/clustering reconstructed only on selected fold; NOT official OpenProblems scaled composite'}
(d/'metrics_raw.json').write_text(json.dumps(output,indent=2,allow_nan=False))
print('FAST8_SUBSET_METRICS',json.dumps(output),flush=True)
