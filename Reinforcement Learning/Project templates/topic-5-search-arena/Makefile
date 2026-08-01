.PHONY: help install serve ui test lint migrate train benchmark verify
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:      ## install the full (training) environment
	pip install -r requirements-train.txt

serve:        ## run the service tier locally — this is what you screen-capture
	uvicorn api.main:app --reload --port 8000

ui:           ## run the Streamlit app locally
	streamlit run ui/app.py

test:         ## run the standing test suite, including the no-torch guard
	pytest -q

lint:
	ruff check .

train:        ## self-play the learned agent and export policies/alphazero_c4.npz
	python -m train.train

benchmark:    ## round-robin tournament + scalability sweep -> games, matches, search_probes
	python -m train.benchmark --games 30

evidence:     ## the SUBMISSION budget: 200 games per pairing. Takes ~25 minutes.
	python -m train.benchmark --agents all --games 200

verify:       ## the four correctness checks at a budget worth quoting (~3 min)
	python -m train.verify

migrate:      ## print the SQL to paste into the Supabase SQL editor
	@cat db/migrations/*.sql
