#!/usr/bin/env python3
"""Deterministic five-fold batch-disjoint CV, with nested held-out validation batches."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/matched/fig1_celltype_batch_counts.csv"
SETS=("dkd","gtex_v9","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens")

def folds(dataset,counts,nfolds=3):
    subset=counts[counts.dataset==dataset]
    if subset.empty:raise ValueError(dataset)
    sizes={str(b):int(n) for b,n in subset.groupby("batch")["count"].sum().items()}
    if len(sizes)<nfolds+2:raise ValueError("Insufficient batches")
    ranked=sorted(sizes,key=lambda b:(-sizes[b],hashlib.sha256((dataset+"/"+b).encode()).hexdigest()))
    tests=[[] for _ in range(nfolds)]
    totals=[0]*nfolds
    for b in ranked:
        idx=min(range(nfolds),key=lambda i:(totals[i],len(tests[i]),i))
        tests[idx].append(b);totals[idx]+=sizes[b]
    result=[]
    for fold,test in enumerate(tests):
        pool=[b for b in sizes if b not in test]
        target=.2*sum(sizes[b] for b in pool)
        candidate=min(pool,key=lambda b:(abs(target-sizes[b]),b))
        valid=[candidate]
        left=[b for b in pool if b!=candidate]
        while len(left)>1:
            current=sum(sizes[b] for b in valid)
            candidate=min(left,key=lambda b:(abs(target-current-sizes[b]),b))
            if abs(target-current-sizes[candidate])>=abs(target-current):break
            valid.append(candidate);left.remove(candidate)
        train=[b for b in pool if b not in valid]
        assert train and valid and test
        assert not(set(train)&set(valid) or set(valid)&set(test) or set(train)&set(test))
        result.append({
          "dataset":dataset,"fold":fold,"seed":42+fold,
          "batch_lists":{"train":sorted(train),"valid":sorted(valid),"test":sorted(test)},
          "n_batches":{"train":len(train),"valid":len(valid),"test":len(test)},
          "n_cells":{key:sum(sizes[x] for x in group) for key,group in
                     [("train",train),("valid",valid),("test",test)]},
          "split_policy":"Grouped 3-fold outer CV; each batch is test once. Inner validation batches disjoint, chosen from remaining groups by cell count only.",
          "test_fraction":sum(sizes[b] for b in test)/sum(sizes.values())})
    assert sorted(b for entry in result for b in entry["batch_lists"]["test"])==sorted(sizes)
    assert sum(entry["n_cells"]["test"] for entry in result)==sum(sizes.values())
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=(*SETS,"all"),default="all")
    p.add_argument("--output",type=Path,default=Path(__file__).parent/"manifests")
    a=p.parse_args()
    counts=pd.read_csv(DATA)
    a.output.mkdir(parents=True,exist_ok=True)
    for ds in SETS if a.dataset=="all" else (a.dataset,):
        result=folds(ds,counts)
        for f in result:
            (a.output/f'{ds}_fold{f["fold"]}.json').write_text(json.dumps(f,indent=2))
        print(ds,"test fractions",[round(f["test_fraction"],3) for f in result],
              "test batches",[f["n_batches"]["test"] for f in result])
    print("Created",a.output)
if __name__=="__main__":main()
