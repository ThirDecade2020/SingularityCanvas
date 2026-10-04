# Singularity Canvas

**Lock a few pixels. Explore the possibilities.**

A local probability playground by **Singularity Machines** — Maximum Control, Minimum Abstraction.

Fix green and purple cells on a 16 × 16 grid, generate conditional samples, and inspect how your clues change the model's probabilities. Torx and JAX perform the sampling; SQLite stores experiment history; a local Qwen model explains saved experiments on demand.

> Current scope: a software demonstration using a model trained on 16 symbols. No physical thermodynamic chip is connected, and no speed or energy advantage has been established.

## Try the live demo

**[Launch Singularity Canvas](https://berkeley-assets-years-nutten.trycloudflare.com)**

Temporary demo hosted on the founder's Mac, with a separate demo database. The link works while the Mac, app, and tunnel are running and changes when the tunnel restarts. A stable address is planned.

Usernames are shared labels, not private accounts: anyone using the same name can view, resume, change, and export that workspace. Use demo information only.

## Features

- Freely lock cells: unknown → green (1) → purple (0) → unknown.
- Automatically save and generate a new batch after editing clues.
- Inspect eight raw sampled completions with every lock preserved.
- View exact conditional pixel probabilities and changes between completed batches.
- Inspect the 16 training references, including clue matches and conflicts.
- Ask the local Qwen guide about a saved experiment.
- Resume a shared username, browse its sessions, and download its stored records.
- Inspect the model fingerprint, seed, execution backend, and computation timing.

Playback cycles through saved samples; it is not a continuous physical p-bit trajectory. Reference filtering does not select, clean up, or replace sampled pixels.

## Stack

| Layer | Technology |
| --- | --- |
| Interface | React, TypeScript, Vite, CSS |
| API | Python, FastAPI, Uvicorn |
| Probability engine | Torx (`extro-torx`), JAX, NumPy, Equinox |
| Model | Binary restricted Boltzmann machine (RBM) |
| Persistence | SQLite |
| Optional local guide | `qwen3.8:27b`, served locally by Ollama |

The original development machine is an Apple M3 Max MacBook Pro with 48 GB RAM. JAX sampling ran on its CPU; Ollama used its GPU. Versions are recorded in `requirements.txt` and the npm lockfiles. Other platforms have not been verified with these exact pins.

## Run locally

Use Python 3.11 and Node.js 24, matching the development environment. Download or clone the repository and open a terminal in its root folder. The included model and catalogue are sufficient to run the canvas; training is not required.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
npm ci --prefix client --cache "$PWD/.cache/npm"
```

Installation requires network access. Normal canvas operation uses local services.

### 2. Start the API

From the repository root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. The application creates missing database tables automatically at `data/singularity_canvas.sqlite3`, preserving existing records. Startup loads the model and prepares its sampling engine.

### 3. Start the interface

In another terminal, from the same repository root:

```bash
npm --prefix client run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to the local API on port 8000. Keep both processes running. This is the local development setup, not a public hosting configuration.

### 4. Optional local AI guide

The canvas works without Ollama. For the guide, run a local Ollama server and install the model configured in `backend/guide_api.py`:

```bash
ollama pull qwen3.8:27b
ollama list
```

If Ollama is not already running, start `ollama serve` in a separate terminal. The guide calls `http://127.0.0.1:11434/api/generate`, not a cloud model. Its model name is currently configured in source, not through an environment variable. Check model availability and local memory capacity on your installation; the original machine used an approximately 18 GB loaded model.

## First experiment

Enter a username such as `firstspark` and select **Open / resume**. Click cells to add green or purple clues; the application saves and samples automatically. Compare the completion grid with **Chance of green**, inspect the current batch, and use the reference gallery to see which training examples satisfy your clues. A compatible reference is not a prediction of your intended symbol.

## How the engine works

The served model has 256 visible bits and 16 active hidden bits, stored in arrays padded to 128 hidden units. `engine/exact_sampler.py` enumerates **65,536 hidden configurations**, analytically sums over unlocked visible bits, and computes conditional hidden weights from the clues.

Torx's `JaxPRNGSampler` draws a hidden configuration. Our custom Torx `ConditionalUpdate` factor uses JAX Bernoulli draws for unlocked pixels and preserves locked values. JAX compiles the draw function. A weighted sum over hidden configurations yields pixel marginals, up to numerical precision; the heatmap is not estimated from the eight displayed samples.

This reference method avoids Gibbs burn-in by enumerating the small hidden space. It does not enumerate every 256-bit image, and its exponential hidden-state cost does not scale to large hidden layers. Earlier Gibbs and training experiments remain in `tools/` for inspection.

The Qwen guide receives a structured experiment summary. It does not generate canvas samples, change clues, or recognize the displayed image; its explanations may be wrong.

## Shared usernames and local records

**A username is a shared workspace label, not authentication.** Anyone who can reach this backend and knows the username can resume its sessions and view or export its history. Keep the development services bound to loopback unless you deliberately design access controls for a shared deployment.

SQLite stores usernames, sessions and current clues, sampling requests and results, seeds, model fingerprints, and AI questions, snapshots, and answers. Account export includes these records; it excludes model weights and source code. Individual clue edits between saved runs are not a complete event timeline.

The repository excludes account databases, account exports, credentials, environment folders, dependency downloads, and backups. It includes small Canvas model checkpoints and symbol data, not Ollama model weights.

## Checks

From the repository root, after installing dependencies:

```bash
python -m unittest discover -s tests -v
python -m engine.exact_sampler
npm --prefix client run build
```

Storage tests use temporary databases. The sampler self-check verifies sample shape and fixed clues. The frontend command checks TypeScript and produces a build; serving that build with API routing is separate from the development setup above.

## Model and dataset scope

The catalogue contains 2,068 unique binary rasterizations derived from 2,078 Bootstrap Icons 1.13.1 source icons. The reproducible split has 1,654 training patterns and 414 validation patterns. **The served diagnostic model learned only 16 training patterns**, identified in `models/rbm_diagnostic_exact16_refined.json`.

Recognizable output is not guaranteed. Five clues can be ambiguous or conflict with every training reference. Alarm and stopwatch completions remain weak. Training-example results do not establish generalization, recognition accuracy, or hardware advantage.

Experimental scripts are development history, not one unified training CLI: some require artifacts generated by earlier experiments under `data/evaluation/`, which is not included. Many trainers refuse to overwrite checkpoints. Use a separate working copy for retraining.

## Next iterations

- Improve and evaluate model quality on a broader symbol catalogue.
- Define a compatible hardware execution adapter while preserving the interface and experiment records.
- Compare sampling quality, latency, throughput, and energy at matched workloads.
- Explore physical thermodynamic hardware access; compatibility and any benefit remain to be demonstrated.

Singularity Canvas is an independent Singularity Machines project using Extropic's tools, not an Extropic product or a claim of endorsement.

## Attribution and licensing

Bootstrap Icons source and derived symbol rasterizations retain the included [MIT notice](data/symbols/LICENSE). Torx, JAX, and other dependencies retain their respective licenses. The Singularity Machines promotional card is included as project branding. Original application code is available under the [MIT License](LICENSE), copyright 2026 Jason Ricciardi.
