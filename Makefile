PYTHON ?= .venv/bin/python
CONFIG ?= configs/case_study.yaml

.PHONY: reproduce
reproduce:
	$(PYTHON) -m pipeline --config $(CONFIG) --noninteractive
