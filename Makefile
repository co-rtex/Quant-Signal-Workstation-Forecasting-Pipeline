PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
PYTEST ?= .venv/bin/pytest
UVICORN ?= .venv/bin/uvicorn
ALEMBIC ?= .venv/bin/alembic

.PHONY: install lint typecheck test validate migrate run-api db-up db-down

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint:
	$(RUFF) check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

validate: lint typecheck test

migrate:
	$(ALEMBIC) upgrade head

run-api:
	$(UVICORN) quant_signal.api.app:app --reload --host 0.0.0.0 --port 8000

db-up:
	docker compose up -d postgres

db-down:
	docker compose down
