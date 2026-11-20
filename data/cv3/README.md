# CV3 single-cell Fast-8 test-only evaluation

**Status (2026-11-20): pilot only, not the final 3-fold CV aggregate.**

The source experiments are a distinct semi-supervised **transductive** CV3 protocol:
- Five atlases (DKD, GTEx v9, Immune Cell Atlas, Mouse Pancreas Atlas, Tabula Sapiens).
- Three outer folds grouped by batch per dataset, with a distinct training-side validation group.
- BERNN and scANVI see **all raw expression** during unsupervised integration, including held-out test batches, but only train-batch biological labels in supervised classification. Validation labels select BERNN checkpoints. Test labels are used solely to score the resulting fixed representations.
- For each fold's Fast-8, take **test cells only**, remove all precomputed whole-atlas neighborhood graphs, rebuild the kNN graph, run seven Leiden resolutions, calculate scIB metrics, and compute a bounded stratified-sample batch ASW approximation. Never subset precomputed whole-atlas clusters or graphs.
- Isolated-label F1 eligibility is recomputed separately for each held-out test fold. If no eligible isolated cell types exist, show N/A, not 0.
- Report the mean and **SD across three fold scores** per metric, and the number of eligible folds. A per-fold composite (if shown) is the mean of applicable raw 0–1 metrics, **not** the official scaled OpenProblems composite, and is not necessarily comparable if eligible metric sets differ.
- **Fast-8 only on TEST for main figures.** Validation metrics, if computed, are explicitly exploratory diagnostics because checkpoint selection depends on validation labels and because validation batches may be too few for batch-mixing metrics.

## First DKD fold 0 results (raw values, seven applicable metrics)

| Metric | BERNN test | scANVI test |
|---|---:|---:|
| Batch ASW*, bounded batch-stratified sample | 0.8596 | 0.8723 |
| iLISI | 0.1572 | 0.3011 |
| Cell cycle conservation | 0.8787 | 0.8536 |
| ARI | 0.8457 | 0.7565 |
| NMI | 0.8418 | 0.8162 |
| Graph connectivity | 0.9798 | 0.9726 |
| Isolated-label F1 | N/A | N/A |
| cLISI | 0.9997 | 0.9998 |
| **Mean (7 eligible raw metrics)** | **0.7947** | **0.7960** |

Sample sizes: train 20,137, validation 5,278 (one batch), test 13,761 (three batches). MCC, native: BERNN train/valid/test 0.9855/0.9431/0.9276; scANVI train/valid/test 0.9998/0.9568/0.9425. Independent probe MCC: BERNN 0.9887/0.9222/0.9007; scANVI 0.9979/0.9475/0.9285.

**Diagnostic validation fast-eight BERNN** is saved separately as `dkd_fold0_bernn_valid_fast8_raw_DIAGNOSTIC.json`. It yields a mean of 0.8982 over **only five** available metrics. Batch ASW, iLISI and isolated-label F1 are unavailable. It must never be compared directly to the 0.7947 test mean, since both the populations and score denominators differ.

## Data files

- `dkd_fold0_bernn_test_fast8_raw.json`, `dkd_fold0_scanvi_test_fast8_raw.json`: test-cell-only **raw** metric values; no synthetic observations and no unavailable-score imputation.
- `dkd_fold0_bernn_valid_fast8_raw_DIAGNOSTIC.json`: nonindependent diagnostic, not a paper result.
- `dkd_fold0_bernn_classifier_scores.json` and `dkd_fold0_scanvi_classifier_scores.json`: saved fitted classifier scores from vm2. They are not integration Fast-8 scores.

The original raw H5AD inputs and intermediate test-only metric outputs (including logs, transformed embeddings and seven clustered objects) are saved in the **separate** vm2 CV3 workspace, not checked into GitHub. They can be generated again from the pinned original source and metric component images. Runtime Docker image tags include mutable `:test` dependencies and need immutable digest pinning before asserting bitwise reproduction.

## Reproduce metric calculations on vm2

After `training/cv/run_cv.py --prepare` and training a BERNN/scANVI fold, run:

```bash
python training/cv/evaluate_fast8_test.py \
  --repo ~/bernn_single_cell_cv_workspace \
  --data-dir ~/task_batch_integration_bernn/resources_full/task_batch_integration/datasets/cellxgene_census \
  --scope test --once dkd-0-bernn

python training/cv/evaluate_fast8_test.py \
  --repo ~/bernn_single_cell_cv_workspace \
  --data-dir ~/task_batch_integration_bernn/resources_full/task_batch_integration/datasets/cellxgene_census \
  --scope test --poll
```

These commands must be run on a host with Docker, original preprocessed source H5AD files and the requisite CPU memory (currently capped at 135 GB per metric container). Neither test scoring nor the CPU metric worker changes fitted models or uses test labels for checkpoint tuning.

The evaluation code resides in `training/cv/prepare_fast8_subset.py`, `evaluate_fast8_test.py`, `fast8_sampled_asw.py` and `fast8_collect.py`. Important safeguard: all test/valid subsets are physically materialized as separate H5AD files, and **the original scANVI full-atlas neighbor graph is always dropped before test-only neighbors are rebuilt**.
