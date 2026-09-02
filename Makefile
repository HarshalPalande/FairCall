.PHONY: install train test demo backtest app clean

VENV := .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

train:
	$(PY) -m src.model

test:
	$(PY) -m pytest tests/ -v

demo: train
	$(PY) -m scripts.demo

backtest: train
	$(PY) -m scripts.backtest

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf data/*.csv models/*.pkl models/*.json artifacts/*.png artifacts/*.json audit_log/*.jsonl
