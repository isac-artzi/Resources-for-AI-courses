.PHONY: help install serve ui test lint migrate pipeline quick equivalence clean
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:      ## install the full (training) environment
	pip install -r requirements-train.txt

quick:        ## the whole pipeline on the offline fallbacks, ~2 minutes. NOT a result.
	python -m train.train --offline --quick

pipeline:     ## the offline pipeline at report budget: 7 betas, ~3 minutes
	python -m train.train --offline

serve:        ## run the service tier locally — this is what you screen-capture
	uvicorn api.main:app --reload --port 8000

ui:           ## run the Streamlit app locally
	streamlit run ui/app.py

test:         ## run the test suite, including the no-torch guard
	pytest -q

equivalence:  ## print the measured NumPy/PyTorch agreement for the deployed head
	pytest -q -s tests/test_equivalence.py -k reproduces

lint:
	ruff check .

migrate:      ## print the SQL to paste into the Supabase SQL editor, in order
	@cat db/migrations/*.sql

clean:        ## remove the embedding cache and the generated charts, not the artifacts
	rm -rf cache reports
