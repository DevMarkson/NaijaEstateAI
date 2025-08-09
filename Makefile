VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip
UVICORN?=$(PY) -m uvicorn

.PHONY: venv install train api test lint docker-build docker-run

venv:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install pytest

train: install
	$(PY) train.py --data lagos-rent.csv --skip-xgb --cv-folds 1 --test-size 0.25

api: install
	$(UVICORN) api_app:app --host 0.0.0.0 --port 8000 --reload

streamlit: install
	$(PY) -m streamlit run streamlit_app.py

test: install
	$(PY) -m pytest -q

docker-build:
	docker build -t naijaestateai:latest .

docker-run:
	docker run -p 8000:8000 naijaestateai:latest
