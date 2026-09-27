# ClaimLens web demo (Angular)

A small Angular 22 front end for ClaimLens with two modes:

- **Replay real runs**: 12 sample documents with the *actual* outputs of the baseline, SFT v2 and GRPO models,
  taken from each run's hash-chained audit log. Every citation box is drawn on the page and scored against the
  ground truth. It is static, so it can be hosted free on GitHub Pages.
- **Live API**: a client for `POST /v1/extract`. It connects to a running ClaimLens service and shows the fields,
  evidence boxes, review flags and audit receipt it returns.

## Run it

Requires **Node.js 22.22.3+ or 24 LTS** (Angular 22's minimum).

```bash
cd web
npm install
npm start                       # http://localhost:4200
```

For the Live API tab, start the service in another terminal from the repo root:

```bash
uvicorn claimlens.service:app --port 8000
```

`http://localhost:4200`, `https://claimlens.netlify.app` and `https://troyclarke69.github.io` are allowed by CORS
by default. Use `CLAIMLENS_CORS_ORIGINS` to change that. On a laptop the service runs the *simulated* predictor
(a test double, not a model), which needs the `doc_id` of a document in your `data/` folder. The sample documents send theirs
automatically.

## Refresh the replay data

After new Kaggle runs are in `runs/`:

```bash
npm run replay                  # = python scripts/build_replay.py
```

This writes `public/replay/replay.json` and copies the sample images. Choose which documents to show in the
`DOCS` list in the script.

## Deploy

**Netlify (recommended).** `netlify.toml` in the repo root holds all the settings: base `web`, build `npm run build`,
publish `dist/claimlens-web/browser`, Node 24.

1. In Netlify, choose **Add new project → Import an existing project → GitHub** and pick `claimLens`.
2. Leave the build settings as they are (Netlify reads them from `netlify.toml`) and choose **Deploy**.
3. Under **Project configuration → Change project name**, rename it to `claimlens` if that name is free, so the
   URL is `https://claimlens.netlify.app`. If you pick another name, update the demo link in the root `README.md`,
   and add the URL to `CLAIMLENS_CORS_ORIGINS` when you run the service.

Every push to `main` redeploys. Pull requests get their own preview URLs.

**GitHub Pages (alternative).** `.github/workflows/web.yml` builds with `--base-href /claimLens/` and publishes to
Pages. Turn it on under **Settings → Pages → Source: GitHub Actions**.

## Where things are (Angular refresher map)

| File | Angular features |
|---|---|
| `src/app/app.config.ts` | Standalone bootstrap, `provideHttpClient(withFetch())`, router with `withComponentInputBinding` and `withHashLocation` |
| `src/app/app.routes.ts` | Lazy-loaded routes (`loadComponent`), route titles |
| `src/app/core/replay.service.ts` | `httpResource`: an HTTP request exposed as signals (`value`, `isLoading`, `error`) |
| `src/app/core/extract-api.service.ts` | `HttpClient` + RxJS `timeout`/`firstValueFrom`, `FormData` upload, a persisted `signal` with `effect` |
| `src/app/pages/demo/demo.ts` | Query params bound straight to `input()`s, `Router.navigate` with `queryParamsHandling` |
| `src/app/pages/demo/replay-panel.ts` | `computed` state, `output()` events |
| `src/app/pages/demo/live-panel.ts` | Async actions with signals, a union-typed state, `DestroyRef` cleanup |
| `src/app/shared/document-viewer.ts` | `input.required`, two-way `model()` binding, SVG overlay in page coordinates |
| `src/app/shared/extraction-result.ts` | Control flow (`@if`, `@for`, `@let`), pipes, `[(selected)]` shared with the viewer |

All components use `OnPush` and signals. There are no NgModules and no zone.js.
