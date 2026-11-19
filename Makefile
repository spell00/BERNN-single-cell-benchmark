.PHONY: all check figures clean

all: check figures

check:
	python scripts/verify_results.py

figures:
	python scripts/plot_fig1_full.py
	python scripts/plot_figures.py --figure all
	python scripts/plot_gtex_umap.py
