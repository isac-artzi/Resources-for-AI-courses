.PHONY: help install serve ui test lint migrate train quick sweep compare equivalence report-budget
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:      ## install the full (training) environment
	pip install -r requirements-train.txt

quick:        ## sandbox budget: all three agents in a couple of minutes. Not a result.
	python -m train.train --quick

train:        ## the three deployable agents at the report budget, exported to policies/
	python -m train.train

compare:      ## A2C vs PPO at matched environment-step budgets, 3 seeds, both envs
	python -m train.compare --steps 60000 --seeds 3 --out runs/compare.json

sweep:        ## the SAC entropy sweep: 3 temperature regimes x 3 seeds, 20k steps each
	python -m train.entropy_sweep --seeds 3 --out runs/entropy_sweep.json

report-budget: compare sweep  ## everything the Quantitative Analysis section needs

equivalence:  ## just the required NumPy/PyTorch checks, with the three measured differences
	python -m pytest tests/test_equivalence.py -q -s

serve:        ## run the service tier locally — this is what you screen-capture
	uvicorn api.main:app --reload --port 8000

ui:           ## run the Streamlit app locally
	streamlit run ui/app.py

test:         ## run the standing test suite, including the no-torch guard
	pytest -q

lint:
	ruff check .

migrate:      ## print the SQL to paste into the Supabase SQL editor, in order
	@cat db/migrations/*.sql
