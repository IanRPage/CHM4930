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
├── backend/     # Express + TypeScript API
├── frontend/    # React + TypeScript web UI
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
