# CHM4930

Our UF Senior Project where we use deep learning to accurately predict molecular
properties of existing chemicals, and hopefully novel ones too. Members:

- Varvara Folimonova
- Ian Rodriguez-Page
- Eman Daraiseh
- Artem Rospotniuk
- Qiyu Ma

## Repo Structure

```text
CHM4930
├── ai/          # model training code, datasets, checkpoints
├── api/         # Vercel Function entry, wraps backend/dist
├── backend/     # Express + TypeScript API
├── frontend/    # React + TypeScript web UI
├── vercel.json  # Vercel build, routing, function config
⋮
└── README.md
```

Setup for each lives in its own README. Node version is pinned in
[`.nvmrc`](.nvmrc).

## Dev Setup

PRs to `main` may touch only one of `ai/`, `backend/`, `frontend/` (enforced by
`PR scope`). Root files are exempt.

Every PR to `main` runs through
[super-linter](https://github.com/super-linter/super-linter), which checks
formatting, style, and security across every file. You can wait for the CI job
to complete, or set up `pre-commit` locally to catch issues before you push:

```bash
pip install pre-commit
pre-commit install
```

Now every `git commit` auto-formats certain staged files with Prettier for
YAML/JSON/etc., and [Ruff](https://docs.astral.sh/ruff/) for Python. If a file
gets reformatted, `git add` it and commit again. Note that this doesn't replace
CI (super-linter still runs on every PR), it just means less hassle for you in
the pipeline.

## Deployment

Frontend and backend deploy as one Vercel project from the repo root, so the
frontend's relative `/api/*` calls are same-origin. Every PR gets a Preview
deployment; `main` deploys to Production.

[`vercel.json`](vercel.json) installs and builds both packages, serves
`frontend/dist`, and routes:

- `/api/*` to [`api/index.js`](api/index.js), a Vercel Function exporting
  `createApp()` from the compiled `backend/dist` (`backend/src/server.ts` is
  local-only). `maxDuration` is 40 s, above the 30 s `MuFuClient` timeout.
- every other path that isn't a static file to `index.html`, so `BrowserRouter`
  deep links like `/predict` load.

Environment variables (Project Settings -> Environment Variables, Production and
Preview):

| Variable       | Value                                                                  |
| -------------- | ---------------------------------------------------------------------- |
| `MUFU_API_URL` | inference server's Cloudflare quick tunnel URL (`*.trycloudflare.com`) |
| `MUFU_API_KEY` | a key named `vercel` from `mufu_server.keys add` (see `ai/README.md`)  |

> The quick tunnel URL changes whenever `cloudflared` restarts. If it does,
> update `MUFU_API_URL` and redeploy (Deployments -> ... -> Redeploy): env var
> changes only apply to new deployments.
