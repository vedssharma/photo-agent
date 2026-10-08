.PHONY: install api lint format typecheck test check

## Install dependencies for both halves
install:
	cd server && uv sync
	cd web && npm ci

## Regenerate api/openapi.json from the backend and the TypeScript types from it
api:
	cd server && uv run python -m photo_agent.openapi
	cd web && npm run gen:api

## Lint and check formatting without changing files
lint:
	cd server && uv run ruff check . && uv run ruff format --check .
	cd web && npm run lint && npm run format:check

## Auto-format everything
format:
	cd server && uv run ruff check --fix . && uv run ruff format .
	cd web && npm run format

## Static type checks
typecheck:
	cd server && uv run mypy
	cd web && npm run typecheck

## Run both test suites
test:
	cd server && uv run pytest
	cd web && npm test

## Everything CI runs
check: lint typecheck test
