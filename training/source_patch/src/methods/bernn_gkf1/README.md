# BERNN GKF1 (batch-disjoint validation)

Separate BERNN method using a deterministic ~20% validation split by **whole batches**. No batch appears in both training and validation. The validation fraction is measured in cells, so it is approximate because batches are indivisible. BERNN receives the validation set as both validation and bookkeeping test; these are **not independent** scores. Embeddings are produced for all cells, including held-out validation batches. `train_after_warmup=True` is explicitly enabled. The original `bernn` method remains unchanged.
