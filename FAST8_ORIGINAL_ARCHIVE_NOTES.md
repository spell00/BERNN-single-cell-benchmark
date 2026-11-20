## Original archived fast-8, with PCR removed

The restored original-score fast-8 is located at:
- `figures/Figure2_fast8_original_archive_controls_without_PCR.pdf` (the original 13 display rows including True labels oracle and No integration).
- `figures/Supplement_fast8_original_archive_all_27_without_PCR.pdf` (all 27 methods and diagnostic controls retained in the 13 October 2026 reference snapshot).
- `figures/Fast8_original_archive_metrics_without_PCR.csv` (original raw and normalized metric values, missingness and provenance).
- `figures/Fast8_original_archive_score_audit_without_PCR.csv` (per-dataset and overall PCR-free aggregates, plus the effect of excluding unavailable metrics).

Reproduce both original-score figures from the archived `results.json`:

```bash
python scripts/plot_fast8_original_archive_without_pcr.py
```

**Do not confuse this with the newer seed-42 experiment.** The original-score fast-8 uses the historical `scaled_scores` and preserves the archive's convention of counting unavailable scores as zero in the composite. Isolated-label F1 on DKD is genuinely N/A, not an unsuccessful correction. The CSV also provides alternative N/A-excluded aggregates.

**Provenance warning:** `selected_bernn_runs.csv` confirms that the historical BERNN values came from dataset-specific, full-label, selectively chosen fits, including some VAE-on configurations. The older PDF displayed BERNN fast-9 = 0.761, whereas the preserved source JSON reports 0.7834 when its 9-metric score is aggregated. The new eight-metric BERNN score 0.8323 is reproduced exactly from that source JSON; the code cannot reproduce the exact old PDF's BERNN run from this source. Other methods come from the same historical snapshot.

Example: GTEx v9 BERNN's original iLISI = 0.0887 raw and 0.2884 normalized, while the newer matched seed-42 run reports 0.0834 raw. Original ARI = 0.9887; newer seed-42 ARI = 0.8192. These differ because the fitted models and evaluation setup differ, not because PCR was removed. The original archive results must not be represented as matched held-out performance.
