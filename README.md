# BERNN single-cell integration: reproducible figures and score audit

**Status:** exploratory research evidence, frozen 19 November 2026. This is the **figure/benchmark companion repository**, not BERNN's model implementation.

- **BERNN model + scikit-learn-inspired estimator API:** https://github.com/spell00/BERNN_MSMS
- **BERNN tutorial:** https://github.com/spell00/BERNN_MSMS/blob/main/TUTORIAL.md
- **OpenProblems source/benchmark fork:** https://github.com/spell00/task_batch_integration_clean
- **Reference scIB/OpenProblems benchmark:** https://openproblems.bio/benchmarks/batch_integration/

## Reproduce all existing plots (no training)

From a fresh checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_results.py
python scripts/plot_fig1_full.py
python scripts/plot_figures.py --figure all
python scripts/plot_gtex_umap.py
```

Results appear in `figures/`, including vector PDFs. `make all` runs the same commands. The repository includes the required **frozen numerical output data**, allowing plots to be reproduced without access to the six large H5AD datasets or an NVIDIA GPU. In a tested source environment, `python scripts/verify_results.py --source-root /home/simonp/task_batch_integration_bernn` additionally checks **every saved fast-nine score and every five-seed BERNN/scANVI isolated F1** against the original H5AD/JSON metric artifacts.

### What each plot is

| File | Source experiment | Meaning |
| --- | --- | --- |
| `Figure1_dataset_composition_full.pdf` | Original cell-type-by-batch counts from six atlases | Batch imbalance and cell-weighted total variation; no integration performed |
| `Figure2_fast9_seed42_raw_audited.pdf` | **Seed 42** matching method/dataset entries | 8 methods × 6 datasets × 9 metrics, primarily OpenProblems metrics (with an explicit sampled batch ASW variant). **One fitted representation** per method/dataset, **not five splits**. |
| `Figure3_MCC_five_splits.pdf` | 21 Oct 2026 matched predictive report | Mean ± SD test MCC across five batch-disjoint splits, 9 entries including no integration |
| `Figure4_GTEx_raw_vs_BERNN_UMAP.pdf` | Saved **full-data** GTEx embeddings, 10k common display sample | Descriptive visualization only; no held-out inference |
| `Supplement_Isolated_F1_five_splits.pdf` | Saved five models, seeds 42–46, for BERNN and scANVI | Variation in isolated-label F1; **not** a zero-shot classification experiment |
| `Supplement_GTEx_all_methods_common10k_UMAP.pdf` | Existing GTEx UMAP coordinates, 11 methods/baseline | All types, batch coloring and highlighted tracheobronchial basal cells |
| `Supplement_legacy_fast9_not_matched.pdf` | Local **13 Oct historical** comparison | **Archival only**; do not substitute for revised Fig. 2 |

### Critical comparability and missingness caveats

1. The **historical BERNN** integration run used **all cell-type labels from the entire dataset**. Its near-perfect isolated-label F1 is *in-sample* and **cannot** be used to claim new-batch generalization. Some historical BERNN dataset runs also differed in VAE settings. The archival figure has been retained as an explicitly labeled supplement, **not** the main result.
2. The **revised fast-nine Figure 2** uses existing **seed-42** outputs from the later matched experiment. BERNN's representation was fitted on training batches and their labels. The scANVI fit masked labels outside training batches, **but included unlabeled expression from the other batches** (transductive). Several other integration methods fit all batches in an unsupervised/transductive setting. They have **different information scopes**: report this when comparing them.
3. The revised Figure 2 shows **raw metric scores already in [0,1]**, and a descriptive equal-weight mean over available metrics/datasets. This **is not** the OpenProblems cross-method normalized composite. In these raw averages ComBat 0.677, BERNN 0.663, scANVI 0.654; these are *not* official rankings. Changing normalization or evaluation protocol can change the order.
4. **Isolated Label F1 is undefined for DKD** under the applicable isolated-label definition. The metric was run; its log reads `isolated labels: []`, and returned NaN. Show **N/A**, not 0. Other 8 methods × 6 datasets × 9 metrics values were recovered after refreshing a stale report and validated against source metric artifacts. The old score-zero imputation, especially for methods with unsupported metric formats, is not used in Figure 2.
5. **Batch ASW** is evaluated on a fixed stratified subsample of cells shared across methods; it is labeled with an asterisk. It is **not** the exhaustive OpenProblems batch-ASW metric.
6. The five F1 scores for BERNN and scANVI vary strongly with split, for example GTEx mean 0.241/0.138, with extreme individual split scores. Do not generalize an advantage based on a single seed. This metric concerns **clustering preservation of isolated populations**, *not* classification of a class never labeled in training.
7. **Figure 4** comes from saved full-data model representations and an independently fitted 10k-cell UMAP per method. The raw UMAP axes are not aligned with BERNN, and full-data BERNN may see the target labels when trained. UMAP alone does not establish biological retention or test MCC.

## Data inventory and verified scopes

```
data/
  provenance.json                     # Source commit and metric definition/scope summary
  matched/
    fast9_seed42_raw.csv              # 432 method/dataset/metric rows + NA status
    isolated_f1_5seeds_bernn_scanvi.csv # 60 method/dataset/seed rows
    split_aligned_mcc_summary.csv     # 5-seed MCC summary
    split_aligned_mcc_long.csv        # 5-seed MCC individual observations
    mcc_and_integration_metrics_current.csv
    fig1_celltype_batch_counts.csv    # Derived count-only table (no expression)
  visualization/
    gtex_sampled_cell_type_and_batch.csv # 10,000 category labels; no cell identifiers
    gtex_umap_*_coords.npy            # frozen two-dimensional common-sample coordinates
  archived/
    results.json, metric_info.json, selected_bernn_runs.csv
    fast9_historical_long.csv
```

This is a reproducible **results and plotting snapshot**: it recreates the paper's displayed numerical analyses from saved values and verifies those values against source files when available. It is **not yet a fully self-contained workflow that downloads every raw atlas and retrains all integration models**. The latter requires the external multi-TB datasets, Docker images, model environments and compute used by the source study. Original OpenProblems metric-component Docker images are referenced by mutable `:test` tags in the source scripts, so **bitwise end-to-end retraining cannot yet be guaranteed** until their immutable digests and preprocessing versions are frozen.

## For source-metric verification

On a machine with the existing OpenProblems outputs:

```bash
git clone https://github.com/spell00/task_batch_integration_clean.git
cd task_batch_integration_clean
git checkout 9cc7960d89d409d76715449e8f8a4f4860a5dfe7
# Populate experiments/single_cell_paper_20261019 and resources_full from the research workspace.
# Then, from this repository checkout:
python scripts/verify_results.py --source-root /path/to/task_batch_integration_clean
```

The `scripts/source_*.py` files are **archived reference scripts** from the original working experiment, not directly portable entry points. They document the exact metric component orchestration, common SGD probe and how score files were collected.

## Figure updates and governance

This repository separates **archive** from **matched experiment** to prevent silent mixing. If additional metrics or methods are computed, save a new versioned raw output CSV with the same `dataset,method,seed,metric,raw_score,state,metric_variant,fit_scope` schema and rerun `scripts/verify_results.py` before merging into the main comparison. Figure 2's descriptive composite intentionally does not use cross-method min-max normalization.

For paper submission, audit any differences in model fit scope, label visibility, dataset eligibility and test-time expression access. Match all comparisons on frozen batch splits. Report both MCC and biological-conservation metrics, without claiming either alone proves full preservation of within-type biology.
