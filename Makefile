.PHONY: install sample train train-sample test lint api docker

install:            ## install everything (pipeline, API, tests)
	pip install -r requirements-dev.txt && pip install -e .

sample:             ## synthetic data with the IEEE-CIS schema (for a quick dry run)
	python scripts/make_sample_data.py --out data/sample --rows 30000

train:              ## full run on the Kaggle data in data/, writes results into README
	python -m fraud.pipeline --update-readme

train-sample: sample ## dry run on synthetic data (results meaningless)
	python -m fraud.pipeline --data-dir data/sample --skip-logistic --output-dir sample_run
	@echo "Dry run outputs are in sample_run/ (not committed). Serve it with: MODEL_PATH=sample_run/artefacts/model.joblib make api"

test:
	pytest -q

lint:
	ruff check .

api:                ## serve the trained model on localhost:8000 (docs at /docs)
	uvicorn api.app:app --reload

docker:             ## build and run the scoring API in Docker
	docker build -t fraud-api . && docker run --rm -p 8000:8000 fraud-api
