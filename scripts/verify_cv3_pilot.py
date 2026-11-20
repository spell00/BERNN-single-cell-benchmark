#!/usr/bin/env python3
"""Audit CV3 DKD fold0 source score snapshot without touching model fitting."""
from pathlib import Path
import json,math
P=Path(__file__).resolve().parents[1]/"data/cv3"
scores={}
for m in ("bernn","scanvi"):
    a=json.loads((P/f"dkd_fold0_{m}_test_fast8_raw.json").read_text())
    b=json.loads((P/f"dkd_fold0_{m}_classifier_scores.json").read_text())
    assert a["dataset"]=="dkd" and a["scope"]=="test"
    assert a["n_cells"]==13761 and a["n_batches"]==3
    assert a["n_eligible_metrics"]==7 and a["n_possible_metrics"]==8
    assert a["metrics"]["isolated_label_f1"] is None
    v=[float(x) for x in a["metrics"].values() if x is not None]
    assert len(v)==7 and all(0<=x<=1 for x in v)
    assert math.isclose(sum(v)/7,a["raw_equal_weight_mean_eligible_only"],abs_tol=1e-12)
    assert a["raw_equal_weight_mean_eligible_only"]>0
    assert b["n_cells"]["test"]==a["n_cells"]
    assert 0<=b["native"]["test"]<=1
    scores[m]=(a,b)
assert scores["bernn"][0]["metrics"]["ari"] > scores["scanvi"][0]["metrics"]["ari"]
assert scores["scanvi"][0]["metrics"]["ilisi"] > scores["bernn"][0]["metrics"]["ilisi"]
diag=json.loads((P/"dkd_fold0_bernn_valid_fast8_raw_DIAGNOSTIC.json").read_text())
assert diag["scope"]=="valid" and diag["n_batches"]==1 and diag["n_cells"]==5278
assert diag["n_eligible_metrics"]==5
assert diag["metrics"]["ilisi"] is None and diag["metrics"]["asw_batch_sampled"] is None
for m,(a,b) in scores.items():
    print(f"{m}: test-only Fast8 raw (7/8)={a['raw_equal_weight_mean_eligible_only']:.6f}, native MCC={b['native']['test']:.6f}")
print("PASS: CV3 DKD test-only metric eligibility, score aggregation and diagnostic validation separation")
