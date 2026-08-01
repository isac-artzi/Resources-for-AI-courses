.PHONY: help install env train smoke serve ui test lint migrate
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:      ## install the full (training) environment
	pip install -r requirements-train.txt

env:          ## print the observation and action spaces — build step 3 wants these
	python -m envs

train:        ## the product configuration: 20,000 episodes x 3 seeds, then export
	python -m train.train

smoke:        ## a ten-second version that proves the pipeline runs end to end
	python -m train.train --episodes 4000 --seeds 2 --random-episodes 200

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
