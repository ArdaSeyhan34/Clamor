.PHONY: install demo app test lint evaluate

install:        ## install Clamor with the dashboard, Claude and dev extras
	pip install -e ".[all]"

demo:           ## run the full pipeline on the synthetic dataset and write reports/demo
	clamor demo

app:            ## start the interactive dashboard
	streamlit run app/streamlit_app.py

evaluate:       ## benchmark embedding backends against the ground truth
	clamor evaluate

test:
	pytest

lint:
	ruff check . && ruff format --check .
