# Singularity Canvas interface

React + TypeScript + Vite. See the [project README](../README.md) for local setup, model scope, database behavior, and the optional Qwen guide.

From the repository root:

```bash
npm ci --prefix client
npm --prefix client run dev
```

The development server uses `127.0.0.1:5173` and proxies `/api` to `127.0.0.1:8000`. Start the Python API separately. `npm --prefix client run build` checks TypeScript and creates the frontend build.
