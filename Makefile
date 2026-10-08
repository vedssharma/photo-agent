SERVER_PORT ?= 8000
WEB_PORT ?= 5173

.PHONY: dev deps install api lint format typecheck test check

## Start the backend and the web app together (Ctrl-C stops both)
dev: deps .env
	@echo "web:    http://localhost:$(WEB_PORT)"
	@echo "server: http://localhost:$(SERVER_PORT)/docs"
	@trap 'kill 0' INT TERM; \
	(cd server && uv run uvicorn photo_agent.main:app --reload --port $(SERVER_PORT)) & \
	(cd web && API_URL=http://127.0.0.1:$(SERVER_PORT) npm run dev -- --port $(WEB_PORT) --strictPort) & \
	wait

## Install or update dependencies only if needed (fast when already installed)
deps:
	cd server && uv sync --quiet
	cd web && npm install --no-audit --no-fund --silent

.env:
	cp .env.example .env
	@echo "Created .env from .env.example; add your ANTHROPIC_API_KEY there."

## Clean install of dependencies for both halves, exactly as locked
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
