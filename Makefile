VENV=.venv
PY=python3

.PHONY: venv install up down build logs test seed

venv:
	$(PY) -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate && pip install -r requirements.txt

up:
	docker compose up --build

down:
	docker compose down

build:
	docker compose build --no-cache

logs:
	docker compose logs -f api

test:
	pytest -q

seed:
	docker compose exec api python seed_db.py
