.PHONY: help install serve ui test lint migrate train plan learn compare
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:      ## install the full (training) environment
	pip install -r requirements-train.txt

train:        ## the one command that reproduces every number in the README (~25s)
	python -m train.train

plan:         ## value iteration only — the exact solution
	python -m train.value_iteration

learn:        ## monte carlo control + evaluation only
	python -m train.monte_carlo

compare:      ## the 10-seed convergence study and its statistics
	python -m train.compare

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
