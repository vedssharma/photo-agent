# api

`openapi.json` is the API contract between `web/` and `server/`. It is generated from the FastAPI app, and the TypeScript types in `web/src/api/schema.d.ts` are generated from it. Both files are committed; CI fails if either is out of date.

Regenerate after changing any backend route or model:

```sh
make api
```
