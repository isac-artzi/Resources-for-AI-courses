.PHONY: help install serve ui test lint migrate train ablation report-budget equivalence
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:      ## install the full (training) environment
	pip install -r requirements-train.txt

train:        ## train the deployable agent and export policies/vpg_cartpole.npz
	python -m train.train --episodes 400 --seeds 3

ablation:     ## the 2x2 at the sandbox budget — for finding bugs cheaply
	python -m train.ablation --episodes 120 --seeds 3

report-budget: ## the 2x2 at the budget you actually report: 12 runs x 1000 episodes
	python -m train.ablation --episodes 1000 --seeds 3 --out runs/ablation.json

equivalence:  ## just the required NumPy/PyTorch check, with the measured difference
	python -m pytest tests/test_equivalence.py -q -s

serve:        ## run the service tier locally — this is what you screen-capture
	uvicorn api.main:app --reload --port 8000

ui:           ## run the Streamlit app locally
	streamlit run ui/app.py

test:         ## run the standing test suite, including the no-torch guard
	pytest -q

lint:
	ruff check .

migrate:      ## print the SQL to paste into the Supabase SQL editor
	@cat db/migrations/*.sql
