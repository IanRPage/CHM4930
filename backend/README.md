# CHM4930: Backend

Express 5 + TypeScript BFF between the frontend and the hosted MuFu model. All
routes are under `/api/*` (same-origin with the frontend on Vercel).

## Layout

```
backend/
├── src/
│   ├── app.ts          # createApp(): app factory, no listen
│   ├── server.ts       # listen entry point (excluded from coverage)
│   ├── errors.ts       # httpError() for client-facing 4xx
│   ├── config/         # env access
│   ├── routes/         # routers, mounted under /api in routes/index.ts
│   ├── controllers/
│   ├── middleware/     # notFound, errorHandler
│   ├── models/         # request/response types + validation
│   └── services/       # MuFuClient, outbound calls
├── docs/               # API contracts
└── tests/              # Vitest + supertest, mirrors src/
```

## Conventions

- ESM with `NodeNext` resolution. Relative imports use `.ts` extensions;
  `rewriteRelativeImportExtensions` rewrites them to `.js` in `dist/`.
- Test against `createApp()` with supertest; don't bind ports in tests.
- Don't try/catch just to forward errors: Express 5 propagates async rejections
  to `errorHandler`. Unknown errors become a generic 500 (logged server-side).
  Errors with a numeric `status` and `expose: true` (http-errors convention) are
  returned as-is. Create client-facing 4xx with `httpError()` (`src/errors.ts`).
- JSON body limit is 100 kB (`app.ts`).
- Type-aware ESLint (`recommendedTypeChecked`), so floating promises fail lint.
- Prettier is pinned to 3.1.0 to match the root pre-commit hook. Don't bump one
  without the other.
- RDKit.js and OpenChemLib (`src/models/molInfo.ts`) load once per instance, on
  the first `/mol-info` request. They read data files by runtime paths that
  Vercel's file tracer can't follow, so those files are listed in the root
  `vercel.json` `includeFiles`.

## Setup

Node 24.22.1 (root [`.nvmrc`](../.nvmrc), `engines` enforces `>=24.22.1 <25`).
Older lines are affected by
[CVE-2026-111497](https://www.aikido.dev/blog/node2shell-nodejs-brotli-rce-cve-2026-111497).

```bash
npm ci
cp .env.example .env   # optional, loaded by dev and start
npm run dev            # tsx watch, GET /api/health -> {"status":"ok","modelVersion":...}
```

Env vars (`MUFU_API_URL`, `MUFU_API_KEY`, `PORT`): see `.env.example`.

| Script                   |                                        |
| ------------------------ | -------------------------------------- |
| `dev`                    | `tsx watch src/server.ts`              |
| `build` / `start`        | `tsc -p tsconfig.build.json` → `dist/` |
| `lint`                   | ESLint                                 |
| `format:check`           | Prettier                               |
| `typecheck`              | `tsc --noEmit` (src + tests)           |
| `test` / `test:coverage` | Vitest; coverage writes cobertura      |

## CI

[`backend.yml`](../.github/workflows/backend.yml) runs `Backend lint` (ESLint,
Prettier, `tsc`) and `Backend tests` (Vitest + `diff-cover --fail-under=60` on
changed lines) on PRs to `main` that touch `backend/`, `.nvmrc`, or the
workflow. Otherwise both jobs are skipped. PRs must not also touch `ai/` or
`frontend/` (`PR scope` check).

## Planned API

| Endpoint                                                                | Screen                | Notes                                                                                                                                              |
| ----------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /api/predict` ([docs](docs/api.md#post-apipredict))               | Single prediction     | Implemented. Per target (EGFR, BACE1): pIC50 + 95% CI, IC50, P(active); 12 Tox21 probs; ClinTox clinical-trial toxicity prob; applicability domain |
| `GET /api/mol-info?smiles=` ([docs](docs/api.md#get-apimol-infosmiles)) | Molecular information | Implemented. Identifiers, descriptors, Lipinski violations, 2D SVG, 3D MolBlock. Computed in the backend with RDKit.js + OpenChemLib               |
| `POST /api/batch`, `GET /api/batch/:id`, results (paged/filter/CSV)     | Batch                 | 10 MB CSV/SDF/TXT, progress, cancel, ~2 min. Exceeds a single serverless invocation: needs a job store or chunking. Stretch goal                   |
| `GET /api/dataset/summary`, `GET /api/dataset/molecules?q=&page=`       | Dataset explorer      | precompute from `ai/data/baseline/combined.csv`                                                                                                    |
