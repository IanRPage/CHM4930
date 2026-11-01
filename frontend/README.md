# CHM4930: Frontend

React 19 + TypeScript SPA on Vite, with React Router 7 (declarative mode). Talks
only to the backend via relative `/api/*` paths: proxied to `localhost:3000` in
dev, same-origin on Vercel in prod.

## Layout

```
frontend/
├── vite.config.ts        # /api proxy, Vitest + coverage config
├── public/
└── src/
    ├── main.tsx          # createRoot + BrowserRouter (excluded from coverage)
    ├── App.tsx           # <Routes>, all pages nested under <Layout>
    ├── styles/theme.css  # color palette, shadows, fonts as CSS variables
    ├── index.css         # global layout (body, sidebar, main)
    ├── components/
    │   ├── common/       # Components shared by each page
    │   └── layout/       # Layout (sidebar + <Outlet/>), Sidebar
    ├── pages/            # one folder per route: page, its CSS, sub-components, tests
    ├── services/         # predict() adapter + PredictError (typed errors)
    ├── types/            # request/response types, mirror backend models
    └── test/             # setup.ts (jest-dom, RTL cleanup), shared fixtures
```

## Setup

Node 24.22.1 (root [`.nvmrc`](../.nvmrc), `engines` enforces `>=24.22.1 <25`).

```bash
npm ci
npm run dev   # http://localhost:5173; run the backend for /api
```

| Script                   |                                            |
| ------------------------ | ------------------------------------------ |
| `dev`                    | Vite dev server, `/api` → `localhost:3000` |
| `build` / `preview`      | `tsc -b && vite build` → `dist/`           |
| `lint`                   | ESLint                                     |
| `format:check`           | Prettier                                   |
| `typecheck`              | `tsc -b` (app + vite config)               |
| `test` / `test:coverage` | Vitest; coverage writes cobertura          |

## CI

[`frontend.yml`](../.github/workflows/frontend.yml) runs `Frontend lint`
(ESLint, Prettier, `tsc`) and `Frontend tests` (Vitest +
`diff-cover --fail-under=60` on changed lines) on PRs to `main` that touch
`frontend/`, `.nvmrc`, or the workflow.

## Code Conventions

Follow these when changing anything under `frontend/`.

### Structure

- Each route gets a folder in `pages/` holding the page component, its CSS,
  sub-components, and tests. Split a page with tabs or sections into
  subfolders the same way (see `pages/single-prediction/`).
- To add a page, create `pages/<route>/`, add a `<Route>` in `App.tsx`, and add
  a `NAV_ITEMS` entry in `components/layout/Sidebar.tsx`.
- Put anything used by more than one page in `components/common/`. Move a
  page-local piece there once a second page needs it, not before.
- A component file exports only components (`react-refresh`). Put helpers in a
  `.ts` file next to it.

### Data and API

- Components never call `fetch`. Go through `services/`.
- Call the backend with relative `/api/...` URLs. Don't add a base URL env var.
- Types in `types/` mirror `backend/src/models/`. When the backend contract
  changes, update them in a follow-up frontend PR (PRs touch one area only).
- `predict()` throws `PredictError` with a `kind` (`invalid-smiles`, `busy`,
  `unavailable`, `timeout`, `server`, `network`, `invalid-response`) and a
  user-facing `message`. If the caller aborts, it rethrows the `AbortError`;
  ignore it.
- A missing or malformed output arrives as `{ value: null, reason }`. Render it
  as unavailable (e.g. "—" with the reason). Never substitute a default.

### Styling

- Plain CSS, one file per component, imported by that component. No CSS
  framework.
- Prefix class names with the page (`predict-`, `home-`, `about-`). Common
  components use unprefixed names (`card`, `panel`, `pill`, `meter`).
- Use the variables in `styles/theme.css` (`var(--slate-500)`), never raw hex.
  SVG `fill`/`stroke` attributes can't read variables, so set them via `style`.
- Reach for `components/common/` before writing new CSS:
  - `PageHeader`: page title and subtitle.
  - `Card`: white surface. `as` sets the element (`form`, `article`).
    Override its default 24px padding with a class.
  - `Panel`: titled section inside a card, with `icon`, `subtitle`,
    `controls` (next to the title), and `aside` (right side).
    `tone="good" | "neutral"` tints it and sets `--tint`, `--line`, `--track`,
    `--fill`, `--value`, and `--strong` for its contents.
  - `Pill`: status badge, `good` or `bad`.
  - `Meter`: bar for a 0–1 value, styled with `--meter-height`,
    `--meter-track`, and `--meter-color`.

### TypeScript and tooling

- Relative imports keep the `.ts`/`.tsx` extension. `verbatimModuleSyntax` is
  on, so type-only imports use `import type`.
- Run `lint`, `format:check`, `typecheck`, and `test` before pushing. CI runs
  the same checks.
- Prettier is pinned to 3.1.0 to match the root pre-commit hook. Bump both or
  neither.

### Testing

- Colocate tests (`Foo.test.tsx`). Vitest + Testing Library on jsdom.
- Vitest globals are off: import `describe`/`it`/`expect`/`vi` from `vitest`.
- Render anything using router hooks or `NavLink` inside a `MemoryRouter` (see
  `App.test.tsx`).
- Mock `fetch` with `vi.stubGlobal` in service tests. Mock the service module
  with `vi.mock` in page tests. Shared response fixtures live in
  `test/fixtures.ts`.
- Query by role, label, or text, not class names, unless the class is the
  behavior under test (e.g. a panel's tone).
