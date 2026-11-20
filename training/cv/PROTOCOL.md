# CV3 transductive semi-supervised evaluation (prepared, not trained)

This is a NEW experimental protocol. It does not modify frozen October 13 fast-nine scores or the October 19 five-seed results.

## Experimental design

- **Datasets (5):** DKD, GTEx v9, Immune Cell Atlas, Mouse Pancreas Atlas, Tabula Sapiens.
- **Excluded dataset:** HypoMap. One pooled batch contains roughly 92% of cells, which makes a balanced, interpretable three-fold grouped evaluation impractical. The dataset remains in Figure 1 for the batch-composition analysis.
- **CV:** three outer folds, grouped by batch. Each batch belongs to exactly one test fold across the three fits. Inner validation groups are disjoint from outer test and other training groups. This is not random per-cell CV.
- **Training:** both BERNN and scANVI use the expression matrices of train, validation and test cells for **unsupervised** objectives. Only **train** cell-type labels enter supervised losses. BERNN uses reconstruction and batch/domain objectives with unlabeled validation/test expression; scANVI masks cell-type labels of the nontrain batches.
- **Model selection:** inner validation labels select the BERNN checkpoint, not the test labels. Never tune on outer test scores. A separate common-label classifier is fitted with train labels only for MCC.
- **Test-only evaluation:** compute the integration metrics from the **test cells and their batches alone**, after fitting. Do not compute on the full integrated atlas and then call it a held-out metric. Each metric is evaluated independently on each fold; report the arithmetic mean over the three folds and a sample SD (ddof=1) as error bars. Report N/A rather than 0 when a subset is not eligible for a metric.
- **Caveat:** this is **transductive batch adaptation**, not inductive prediction on unseen expression. Test batch membership and expression may be used without its biological labels. ScIB isolated-label eligibility and iLISI may change if test groups lack the relevant populations: define and disclose these cases.

### Existing archive versus the proposed CV3

The October 13 fast-nine experiment used BERNN trained on all cell labels and evaluated all cells (with selected VAE settings). The October 19–21 experiment used five nonexhaustive batch splits for MCC, training-label-only BERNN and transductive scANVI, with integration metrics calculated on all cells. These datasets and scores remain archived unchanged.

This new CV3 does **not yet produce valid new figures** until the models are trained and their test-only metrics are calculated. The provided runner prepares a fresh study workspace, then trains both methods, produces full embeddings, scores MCC and saves fold manifests. The independent test-only scIB metric pipeline is being specified separately and must be run before final publication claims.

## Recreate fold assignments

```bash
pip install pandas
python training/cv/create_folds.py
```

Creates 15 JSON manifests (5 datasets × 3 folds) in `training/cv/manifests`. Each test batch appears exactly once. Fold assignment is deterministic from `data/matched/fig1_celltype_batch_counts.csv`, with no test-label-based optimization.

## Train the models (requires Docker, NVIDIA GPU, original dataset H5AD files)

The training code derives from the saved October 2026 experimental workflow and the BERNN transductive version. It is provided as an **experimental first-run candidate**, not a validated one-command reproduction of final paper scores.

```bash
python training/cv/run_cv.py \
  --prepare \
  --workspace ~/bernn_single_cell_cv_workspace \
  --data-dir ~/task_batch_integration_bernn/resources_full/task_batch_integration/datasets/cellxgene_census

python training/cv/run_cv.py \
  --run --dataset dkd --fold 0 --method bernn \
  --workspace ~/bernn_single_cell_cv_workspace \
  --data-dir ~/task_batch_integration_bernn/resources_full/task_batch_integration/datasets/cellxgene_census \
  --bernn-image bernn-openproblems-vm2:cu121-fast-644b5a9

python training/cv/run_cv.py \
  --run --dataset dkd --fold 0 --method scanvi \
  --workspace ~/bernn_single_cell_cv_workspace \
  --data-dir ~/task_batch_integration_bernn/resources_full/task_batch_integration/datasets/cellxgene_census \
  --scanvi-image sc-paper-scvi-cu121:20261020
```

After inspecting the single-fold logs, use `--dataset all --fold all --method both` with both image flags for all 30 fits. The runner is resumable (skips completed outputs) and never touches the earlier paper results. The BERNN transductive implementation records `transductive_audit.json`, embeddings and held-out MCC, including an audit of train-only labels.

The fixed parameter file `training/cv/fixed_hpo_config.json` uses BERNN's documented baseline parameters rather than selecting separate hyperparameters using outer-test performance. BERNN training still uses validation checkpoint selection. This configuration is a starting point, not a claim of optimized performance.

### Data and container requirements

The five preprocessed datasets must exist as `dataset.h5ad` and `solution.h5ad` under `<data-dir>/<dataset>/log_cp10k/`. They originate from the OpenProblems dataset-processing workflow; raw CELLxGENE download/normalization has not yet been packaged into this repository.

On vm2 the launcher uses explicit NVIDIA character devices and CUDA library mounts rather than Docker `--gpus`, because that daemon has no NVIDIA container runtime registered for `--gpus`. The training allocation is GPU 0; an unrelated existing container uses GPU 1.

The original local BERNN Docker image on vm2 is tagged `bernn-openproblems-vm2:cu121-fast-644b5a9` (image ID `sha256:28f55482de9d0297ffa76bffcf2104510492108132018bc7f039e5c4ea598ff3`). This local image is not available automatically on a fresh machine. Similarly the scANVI CUDA 12.1 compatibility image `sc-paper-scvi-cu121:20261020` must be built or loaded separately. The source scripts preserve the exact requested tags and source commit; the existing upstream `:test` Docker tags are mutable, so these are **not yet bitwise portable**.

The source OpenProblems task is pinned at commit `9cc7960d89d409d76715449e8f8a4f4860a5dfe7`. The original experiment scripts are in `training/experiment_code`, including the frozen transductive BERNN implementation. Generated artifacts under the workspace are intentionally not committed.

## Live vm2 launch

The original workspace clone at `/home/simonp/bernn_single_cell_cv_workspace` is using the frozen source revision and this CV3 protocol. Its detached queue is controlled by `cv3_master.pid`, and the log is `cv3_master.log`. Per-fit logs are in `experiments/single_cell_paper_20261019/cv_logs/`. The queue runs BERNN then scANVI per fold sequentially on GPU 0; never confuse its first live losses with completed fold scores. The repository includes the historical untracked `bernn_gkf1` adapter source in `training/source_patch/` so a fresh source checkout can reproduce the training code.
