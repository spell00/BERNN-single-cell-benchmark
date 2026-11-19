# GTEx reduced UMAP snapshots

The 10,000 cells are a common deterministic random subset (seed 42) of GTEx v9's 209,126 cells. A single sample-index ordering is shared by all 11 method-coordinate arrays. Only category annotations are saved, no original expression or cell IDs. The source data were public research data.

UMAP was fit independently per method using n_neighbors=30, min_dist=0.3, random_state=42; coordinates are not registered across methods. Embeddings are from saved **full-dataset** integration fits, and this figure is visual/descriptive only, not a held-out batch prediction experiment. Labels may have been available to BERNN during original full-data embedding training. UMAP is nonlinear; apparent islands are not definitive evidence of biological fragmentation.
