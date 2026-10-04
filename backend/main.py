from backend.symbol_api import build_reference, install as install_symbols
import hashlib
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from threading import Lock
import json
import secrets
import time

import jax
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from engine.exact_sampler import ExactRBMSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/rbm_diagnostic_exact16_refined.npz"


class Clue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(strict=True, ge=0, le=255)
    value: int = Field(strict=True, ge=0, le=1)


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clues: list[Clue] = Field(default_factory=list, max_length=256)
    count: int = Field(default=8, strict=True, ge=1, le=64)
    seed: int | None = Field(default=None, strict=True, ge=0, le=4294967295)
    revision: int = Field(default=0, strict=True, ge=0)

    @model_validator(mode="after")
    def unique_cells(self):
        indices = [clue.index for clue in self.clues]
        if len(indices) != len(set(indices)):
            raise ValueError("Each cell may appear only once.")
        return self


@asynccontextmanager
async def lifespan(app):
    sampler = ExactRBMSampler(MODEL)
    sampler.sample(
        np.zeros(256, dtype=bool),
        np.zeros(256, dtype=np.uint8),
        seed=0,
        count=1,
    )
    metadata = json.loads(MODEL.with_suffix(".json").read_text())
    app.state.sampler = sampler
    app.state.model_sha256 = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    app.state.sampling_lock = Lock()
    app.state.engine_info = {
        "backend": "torx-exact-hidden-enumeration",
        "model": MODEL.name,
        "model_scope": "16-symbol training diagnostic",
        "training_symbols": len(metadata["training_ids"]),
        "active_hidden_units": sampler.active,
        "enumerated_hidden_states": 2 ** sampler.active,
        "temperature": 1.0,
        "devices": [str(device) for device in jax.devices()],
        "torx_version": version("extro-torx"),
        "jax_version": jax.__version__,
        "physical_thermodynamic_hardware": False,
        "limitations": [
            "Full-catalogue model quality remains unresolved.",
            "Alarm and stopwatch completions remain weak.",
            "Exact hidden enumeration does not scale to large hidden layers.",
            "No computational or energy advantage has been established.",
        ],
    }
    app.state.symbol_reference = build_reference(MODEL, app.state.model_sha256)
    yield


app = FastAPI(title="Singularity Canvas Local API", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {
        "status": "ready",
        "grid": {"width": 16, "height": 16, "order": "row-major"},
        **app.state.engine_info,
    }


@app.post("/api/sample")
def sample(request: SampleRequest):
    locked = np.zeros(256, dtype=bool)
    values = np.zeros(256, dtype=np.uint8)
    for clue in request.clues:
        locked[clue.index] = True
        values[clue.index] = clue.value
    seed = request.seed if request.seed is not None else secrets.randbits(32)

    with app.state.sampling_lock:
        start = time.perf_counter()
        result = app.state.sampler.sample(
            locked, values, seed=seed, count=request.count
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        **result,
        "samples": result["samples"].tolist(),
        "revision": request.revision,
        "clues": [clue.model_dump() for clue in request.clues],
        "compute_ms": elapsed_ms,
        "timing_scope": (
            "Conditioning, sample generation, and exact pixel marginals; excludes startup, "
            "lock waiting, JSON serialization, network, and rendering."
        ),
        "grid": {"width": 16, "height": 16, "order": "row-major"},
        "engine": app.state.engine_info,
    }


# Session routes share the existing sampling engine.
from backend.session_api import install
install(app, sample, SampleRequest)


from backend.guide_api import install as install_guide
install_guide(app)

install_symbols(app)

# Browse stored account records without modifying experiments.
from backend.history_api import router as history_router
app.include_router(history_router)
