.PHONY: dev test lint fmt docker-up docker-down

PYTHON ?= python3

dev:
	docker compose up -d ollama searxng
	$(PYTHON) -m deepdive.api.main

test:
	pytest tests/ -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff format .
	ruff check --fix .

docker-up:
	docker compose up -d

docker-down:
	docker compose down
