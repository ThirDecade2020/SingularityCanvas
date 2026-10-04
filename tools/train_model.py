import hashlib
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from engine.rbm import SWEEP, VISIBLE

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models/rbm_v1.npz"
if DESTINATION.exists():
    raise SystemExit("rbm_v1.npz already exists; left unchanged.")

def load_bits(name):
    with np.load(ROOT / f"data/symbols/{name}.npz", allow_pickle=False) as data:
        return data["bits"].astype(np.float32)

training = load_bits("train")
validation = jnp.asarray(load_bits("validation"))
with np.load(ROOT / "models/rbm_initial.npz", allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}
velocity = jax.tree.map(jnp.zeros_like, params)

EPOCHS = 100
BATCH_SIZE = 32
CD_STEPS = 5
RATE = 0.02
SEED = 45
rng = np.random.default_rng(SEED)
key = jax.random.key(SEED)


@eqx.filter_jit
def train_batch(params, velocity, batch, key):
    def advance(pixels, step_key):
        keys = jax.random.split(step_key, pixels.shape[0])

        def draw(k, image):
            return SWEEP.sample(k, {
                "pixels": image,
                "temperature": jnp.array(1., dtype=jnp.float32),
                "locked": jnp.zeros(VISIBLE, dtype=jnp.bool_),
                "clues": jnp.zeros(VISIBLE, dtype=jnp.float32),
            }, {"model": params})

        return jax.vmap(draw)(keys, pixels), None

    negative, _ = jax.lax.scan(
        advance, batch, jax.random.split(key, CD_STEPS)
    )
    positive_hidden = jax.nn.sigmoid(
        batch @ params["weights"] + params["hidden_bias"]
    )
    negative_hidden = jax.nn.sigmoid(
        negative @ params["weights"] + params["hidden_bias"]
    )
    gradients = {
        "weights": (
            batch.T @ positive_hidden - negative.T @ negative_hidden
        ) / batch.shape[0] - 0.0001 * params["weights"],
        "visible_bias": jnp.mean(batch - negative, axis=0),
        "hidden_bias": jnp.mean(positive_hidden - negative_hidden, axis=0),
    }
    velocity = jax.tree.map(
        lambda old, gradient: 0.5 * old + RATE * gradient,
        velocity, gradients,
    )
    params = jax.tree.map(lambda value, change: value + change, params, velocity)
    return params, velocity


@eqx.filter_jit
def reconstruction_error(params, images):
    hidden = jax.nn.sigmoid(
        images @ params["weights"] + params["hidden_bias"]
    )
    logits = hidden @ params["weights"].T + params["visible_bias"]
    return jnp.mean(jnp.logaddexp(0., logits) - images * logits)


started = time.perf_counter()
reports = []

def report(epoch):
    error = float(reconstruction_error(params, validation))
    if not np.isfinite(error):
        raise RuntimeError("Non-finite validation error; stopping.")
    reports.append({"epoch": epoch, "validation_reconstruction_bce": error})
    print(f"Epoch {epoch:3d}: validation reconstruction BCE = {error:.5f}",
          flush=True)

report(0)
for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(len(training))
    for start in range(0, len(order), BATCH_SIZE):
        batch = jnp.asarray(training[order[start:start + BATCH_SIZE]])
        key, batch_key = jax.random.split(key)
        params, velocity = train_batch(params, velocity, batch, batch_key)
    if epoch % 10 == 0:
        report(epoch)

arrays = {name: np.asarray(value) for name, value in params.items()}
assert all(np.isfinite(value).all() for value in arrays.values())
np.savez_compressed(DESTINATION, **arrays)

manifest = (ROOT / "data/symbols/split_manifest.json").read_bytes()
metadata = {
    "architecture": "binary RBM, 256 visible / 128 hidden",
    "sampling_engine": "Torx DFG",
    "training_method": "CD-5",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": RATE,
    "momentum": 0.5,
    "weight_decay": 0.0001,
    "seed": SEED,
    "augmentation": "none in this first experiment",
    "split_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    "device": str(jax.devices()[0]),
    "elapsed_seconds": time.perf_counter() - started,
    "reports": reports,
}
DESTINATION.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
print("Saved:", DESTINATION)
print("Next: evaluate actual sampled completions.")
