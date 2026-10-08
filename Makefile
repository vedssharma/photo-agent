.PHONY: api

## Regenerate api/openapi.json from the backend and the TypeScript types from it
api:
	cd server && uv run python -m photo_agent.openapi
	cd web && npm run gen:api
