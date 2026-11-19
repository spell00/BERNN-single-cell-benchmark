#!/usr/bin/env python3
"""Validate frozen results and, optionally, compare against source vm2 H5AD/JSON outputs.

Reproduction levels:
- Snapshot audit: fast and self-contained, no data downloads or model training.
- --source-root: verify every saved fast-nine score and F1 against raw metric outputs.
"""
from pathlib import Path
import argparse,json,math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/matched"
METHODS={"bernn","combat","harmony","batchelor_fastmnn","liger","scvi","scanvi","scalex"}
DATASETS={"dkd","gtex_v9","hypomap","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens"}
METRICS={"asw_batch_sampled","ilisi","pcr","cell_cycle_conservation","ari","nmi","graph_connectivity","isolated_label_f1","clisi"}
SOURCE_METRICS={"ari":"clustering_overlap","nmi":"clustering_overlap","ilisi":"lisi","clisi":"lisi"}

def run_snapshot():
    d=pd.read_csv(DATA/"fast9_seed42_raw.csv")
    assert len(d)==8*6*9
    assert set(d.method)==METHODS
    assert set(d.dataset)==DATASETS
    assert set(d.metric)==METRICS
    assert set(d.seed)=={42}
    assert not d.duplicated(["method","dataset","seed","metric"]).any()
    bad=d[d.state!="ok"]
    assert len(bad)==8 and (bad.metric=="isolated_label_f1").all()
    assert (bad.dataset=="dkd").all() and (bad.state=="not_applicable").all()
    assert d[d.state=="ok"].raw_score.between(0,1).all()
    print("PASS: fast-nine snapshot has 8 x 6 x 9 rows; only DKD Isolated F1 is N/A")

    mcc=pd.read_csv(DATA/"split_aligned_mcc_summary.csv")
    methods=METHODS|{"raw_no_integration"}
    g=mcc[mcc.method.isin(methods)].copy()
    assert g.groupby("dataset").method.nunique().eq(9).all()
    assert (g.n_paired_seeds==5).all()
    long=pd.read_csv(DATA/"split_aligned_mcc_long.csv")
    tests=long[(long.method.isin(methods))&(long.split=="test")]
    for (ds,m),vals in tests.groupby(["dataset","method"]):
        assert len(vals)==5 and set(vals.seed)==set(range(42,47)),(ds,m)
        row=g[(g.dataset==ds)&(g.method==m)].iloc[0]
        assert math.isclose(vals.mcc.mean(),row.test_mcc_mean,abs_tol=1e-6), (ds,m,"mean")
        assert math.isclose(vals.mcc.std(ddof=1),row.test_mcc_sd,abs_tol=1e-6), (ds,m,"sd")
    print("PASS: 9 x 6 MCC comparisons have five splits; all means and SDs recomputed correctly")

    f1=pd.read_csv(DATA/"isolated_f1_5seeds_bernn_scanvi.csv")
    assert len(f1)==60 and set(f1.method)=={"bernn","scanvi"}
    assert f1[f1.dataset=="dkd"].state.eq("not_applicable").all()
    assert f1[f1.state=="ok"].isolated_label_f1.between(0,1).all()
    assert f1.groupby(["dataset","method"]).seed.nunique().eq(5).all()
    print("PASS: 5 seeds per dataset for BERNN and scANVI isolated-label F1")
    print("All local snapshot tests passed.")

def verify_raw(source_root):
    import h5py
    exp=source_root/"experiments/single_cell_paper_20261019"
    metrics=pd.read_csv(DATA/"fast9_seed42_raw.csv")
    for row in metrics.itertuples():
        base=exp/("bernn" if row.method=="bernn" else "comparators")/"42"
        if row.method!="bernn":base=base/row.method
        base=base/row.dataset/"metrics"
        if row.metric=="asw_batch_sampled":
            file=base/"asw_batch_sampled.json"
            value=float(json.loads(file.read_text())["value"])
        else:
            name=SOURCE_METRICS.get(row.metric,row.metric)
            file=base/(name+".h5ad")
            with h5py.File(file,"r") as h:
                labels=[z.decode() if isinstance(z,bytes) else str(z) for z in h["uns/metric_ids"][:]]
                scores=np.array(h["uns/metric_values"][:],float)
                value=float(scores[labels.index(row.metric)])
        if row.state=="not_applicable":
            assert np.isnan(value),(row,value)
        else:
            assert math.isclose(value,float(row.raw_score),abs_tol=1e-8,rel_tol=1e-8),(row,value)
    print("PASS: all 432 fast-nine records match source H5AD/JSON values")

    f1=pd.read_csv(DATA/"isolated_f1_5seeds_bernn_scanvi.csv")
    for row in f1.itertuples():
        base=exp/("bernn" if row.method=="bernn" else "comparators")/str(row.seed)
        if row.method!="bernn":base=base/row.method
        file=base/row.dataset/"metrics/isolated_label_f1.h5ad"
        with h5py.File(file,"r") as h:
            v=float(h["uns/metric_values"][0])
        if row.state=="not_applicable":assert np.isnan(v)
        else:assert math.isclose(v,float(row.isolated_label_f1),abs_tol=1e-8,rel_tol=1e-8),(row,v)
    print("PASS: all 60 five-seed isolated F1 records match original H5AD files")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=None,help="Directory containing source experiments, e.g. /home/simonp/task_batch_integration_bernn")
    args=p.parse_args()
    run_snapshot()
    if args.source_root:
        verify_raw(args.source_root.resolve())
    else:
        print("Source H5AD verification skipped. Use --source-root to verify original saved outputs.")
