# CST-MIL WebApp

[Installation and quick start](../README.md#quick-start)

This local application displays risk categories, dynamic discussion topics and source evidence from the five-source semantic model. It uses React, Express, a serial inference queue, a resident Python worker and SQLite. Startup verifies the prepared model assets; it does not train a model or call a paid model API.

Follow the root README to create `cst_mil/.venv` and run `python -m cst_mil.deployment setup` from the repository root. Then run these commands in `WebApp`:

```powershell
npm ci
npm start
```

Open **http://127.0.0.1:3210**. On subsequent runs, use `npm start` from this directory. The health endpoint, analysis output, history and exports record the model identity. Scores, Top-3 candidates and topics support contextual review.

Development checks:

```powershell
npm run build
npx tsx --test tests/ui-source.test.ts tests/portable-deployment.test.ts
```

Real-model integration and browser tests require the model assets. Built-in campus examples are fictional inputs for demonstrating the interface. Experimental predictions, metrics, timing records and historical benchmark results are not bundled with the application. Personal imports, analysis history and review notes stay in local runtime storage.
