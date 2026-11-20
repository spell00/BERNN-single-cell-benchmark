.PHONY: all check figures clean

all: check figures

check:
	python scripts/verify_results.py

figures:
	python scripts/plot_fig1_full.py
	python scripts/plot_figures.py --figure all
	python scripts/plot_fast8_without_pcr.py
	python scripts/plot_gtex_umap.py

fast8-original:
	python scripts/plot_fast8_original_archive_without_pcr.py

.PHONY: cv3-check
cv3-check:
	python training/cv/create_folds.py
	python training/cv/verify_manifests.py
