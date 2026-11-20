#!/usr/bin/env python3
"""Check three-fold grouped outer CV manifests from frozen batch counts."""
from pathlib import Path
import json
import pandas as pd
BASE=Path(__file__).resolve().parents[2]
DATA=pd.read_csv(BASE/"data/matched/fig1_celltype_batch_counts.csv")
FOLDER=BASE/"training/cv/manifests"
DATASETS=["dkd","gtex_v9","immune_cell_atlas","mouse_pancreas_atlas","tabula_sapiens"]
assert len(list(FOLDER.glob("*_fold*.json")))==15
for ds in DATASETS:
    size={str(b):int(n) for b,n in DATA[DATA.dataset.eq(ds)].groupby("batch")["count"].sum().items()}
    manifest=[json.loads((FOLDER/f"{ds}_fold{i}.json").read_text()) for i in range(3)]
    test=[]
    for i,m in enumerate(manifest):
        assert m["dataset"]==ds and m["fold"]==i and m["seed"]==42+i
        b=m["batch_lists"]
        tr,va,te=(set(b[k]) for k in ("train","valid","test"))
        assert not(tr&va or tr&te or va&te)
        assert tr|va|te==set(size)
        assert m["n_cells"]=={k:sum(size[x] for x in b[k]) for k in b}
        assert len(te)>=3,(ds,i,len(te))
        test.extend(b["test"])
    assert sorted(test)==sorted(size)
    print(f"PASS: {ds}: {len(size)} batches, unique test coverage across three folds")
print("15 CV3 split manifest tests passed; HypoMap is not part of this experiment")
