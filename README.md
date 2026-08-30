# CHM4930

Our UF Senior Project where we use deep learning to accurately predict molecular
properties of existing chemicals, and hopefully novel ones too. Members:

- Varvara Folimonova
- Ian Rodriguez-Page
- Eman Daraiseh
- Artem Rospotniuk
- Qiyu Ma

## Repo Structure

```
.
├── ai/          # model training code, datasets, checkpoints (see ai/README.md)
├── backend/     # API / server (TODO)
├── frontend/    # UI           (TODO)
├── LICENSE
└── README.md
```

Right now, work is focused in `ai/`, training a model to predict molecular
properties. See [`ai/README.md`](ai/README.md) for environment setup and the
directory layout there. `backend/` and `frontend/` are placeholders for once
we're ready to build something around the trained model.

## Dev Setup

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
