# CHM4930: Frontend

React 19 + TypeScript SPA on Vite, with React Router 7 (declarative mode). Talks
only to the backend via relative `/api/*` paths: proxied to `localhost:3000` in
dev, same-origin on Vercel in prod.

## Layout

```
frontend/
├── vite.config.ts      # /api proxy, Vitest + coverage config
├── public/
└── src/
    ├── main.tsx        # createRoot + BrowserRouter (excluded from coverage)
    ├── App.tsx         # <Routes>, all pages nested under <Layout>
    ├── components/
    │   └── layout/     # Layout (sidebar + <Outlet/>), Sidebar
    ├── pages/          # route-level components, one per sidebar entry
    ├── services/       # API client (SCRUM-71)
    ├── types/          # request/response types, mirror backend models (SCRUM-81)
    └── test/setup.ts   # jest-dom matchers, RTL cleanup
```

## Conventions

- Relative imports keep `.tsx`/`.ts` extension (`allowImportingTsExtensions`).
  `verbatimModuleSyntax` is on, so type-only imports need `import type`.
- Components never call `fetch` directly; go through `services/`. Use relative
  URLs only, no base URL env var.
- Tests are colocated (`Foo.test.tsx`), Vitest + Testing Library on jsdom.
  Vitest globals are off: import `describe`/`it`/`expect` from `vitest`.
  Anything that uses router hooks or `NavLink` must render inside a
  `MemoryRouter` (see `src/App.test.tsx`).
- ESLint: `typescript-eslint` recommended, `react-hooks`, `react-refresh`
  (component files should only export components), `eslint-config-prettier`.
- Prettier is pinned to 3.1.0 to match the root pre-commit hook. Don't bump one
  without the other.
- No styling library or state manager yet; `index.css` is placeholder CSS.

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
`frontend/`, `.nvmrc`, or the workflow. Otherwise both jobs are skipped. PRs
must not also touch `ai/` or `backend/` (`PR scope` check).

## Routes

From the [app screens](https://www.figma.com/design/hQeLEZcHn3QAtqUVicSccX/MuFu-%E2%80%93-App-Screens).
All pages are stubs.

| Path        | Page                   | Figma sub-screens                                        |
| ----------- | ---------------------- | -------------------------------------------------------- |
| `/`         | `HomePage`             | 01 Home                                                  |
| `/predict`  | `SinglePredictionPage` | 02 input, 03 results, 03b molecule info, 03c 3D view     |
| `/batch`    | `BatchPredictionPage`  | 04 upload, 04b processing, 04c results                   |
| `/explorer` | `DatasetExplorerPage`  | 05 overview, 05b distribution, 05c elements, 05d samples |
| `/about`    | `AboutPage`            | 06 About                                                 |

New page: add it to `pages/`, a `<Route>` in `App.tsx`, and a `NAV_ITEMS` entry
in `Sidebar.tsx`.
