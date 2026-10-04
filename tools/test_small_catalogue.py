import json
from pathlib import Path
from html import escape
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from engine.rbm import SWEEP, sample_chain, VISIBLE, HIDDEN

root = Path(__file__).resolve().parents[1]
output = root / "data/evaluation/small_catalogue"
output.mkdir(parents=True, exist_ok=True)
destination = root / "models/rbm_diagnostic_16.npz"
if destination.exists():
    raise SystemExit("Diagnostic model already exists; left unchanged.")

rng = np.random.default_rng(74)
with np.load(root / "data/symbols/train.npz", allow_pickle=False) as data:
    indices = rng.choice(len(data["bits"]), 16, replace=False)
    examples = data["bits"][indices].astype(np.float32)
    ids = data["ids"][indices]

catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
names = {p["id"]: p["names"][0] for p in catalogue["patterns"]}
batch = jnp.asarray(examples)
frequency = np.clip(examples.mean(axis=0), 0.01, 0.99)
params = {
    "weights": jnp.asarray(
        rng.normal(0, 0.01, (VISIBLE, HIDDEN)).astype(np.float32)
    ),
    "visible_bias": jnp.asarray(np.log(frequency / (1 - frequency))),
    "hidden_bias": jnp.zeros(HIDDEN),
}
velocity = jax.tree.map(jnp.zeros_like, params)
particles = jnp.asarray(rng.integers(0, 2, (64, VISIBLE)), dtype=jnp.float32)
mask = jnp.zeros(VISIBLE, dtype=jnp.bool_)
clues = jnp.zeros(VISIBLE)
key = jax.random.key(75)

@eqx.filter_jit
def train_step(params, velocity, particles, key):
    def advance(states, draw_key):
        keys = jax.random.split(draw_key, states.shape[0])
        updated = jax.vmap(lambda state, k: SWEEP.sample(
            k, {
                "pixels": state,
                "temperature": jnp.array(1.0),
                "locked": mask,
                "clues": clues,
            }, {"model": params}
        ))(states, keys)
        return updated, None

    negative, _ = jax.lax.scan(
        advance, particles, jax.random.split(key, 20)
    )
    positive_hidden = jax.nn.sigmoid(
        batch @ params["weights"] + params["hidden_bias"]
    )
    negative_hidden = jax.nn.sigmoid(
        negative @ params["weights"] + params["hidden_bias"]
    )
    gradients = {
        "weights": (
            batch.T @ positive_hidden / batch.shape[0]
            - negative.T @ negative_hidden / negative.shape[0]
            - 0.0001 * params["weights"]
        ),
        "visible_bias": batch.mean(0) - negative.mean(0),
        "hidden_bias": positive_hidden.mean(0) - negative_hidden.mean(0),
    }
    velocity = jax.tree.map(
        lambda old, gradient: 0.5 * old + 0.02 * gradient,
        velocity, gradients,
    )
    params = jax.tree.map(lambda p, change: p + change, params, velocity)
    return params, velocity, negative

print("Training a diagnostic model on 16 training-set symbols.", flush=True)
for step in range(1, 3001):
    key, draw_key = jax.random.split(key)
    params, velocity, particles = train_step(
        params, velocity, particles, draw_key
    )
    if step % 500 == 0:
        assert all(
            np.isfinite(np.asarray(value)).all() for value in params.values()
        ), "Nonfinite model parameters."
        print(
            f"Update {step}: persistent-chain green fraction "
            f"{float(particles.mean()):.1%}",
            flush=True,
        )

np.savez_compressed(destination, **{
    name: np.asarray(value) for name, value in params.items()
})

# Evaluation starts are independent random grids, not training examples.
initials = jax.random.bernoulli(
    jax.random.key(76), 0.5, (16, VISIBLE)
).astype(jnp.float32)

@eqx.filter_jit
def generate(params, initials):
    return jax.vmap(lambda initial, k: sample_chain(
        params, initial, mask, clues, k, steps=2000
    )[0])(initials, jax.random.split(jax.random.key(77), 16))

samples = np.asarray(generate(params, initials)).astype(np.uint8)
distances = np.mean(samples[:, None, :] != examples[None, :, :], axis=-1)
nearest = distances.argmin(axis=1)
agreements = 1 - distances.min(axis=1)

def picture(bits, label):
    colors = ("#7028ba", "#62ef62")
    cells = "".join(
        f'<rect x="{i % 16}" y="{i // 16}" width="1" height="1" '
        f'fill="{colors[int(bit)]}"/>'
        for i, bit in enumerate(bits)
    )
    return (
        f'<figure><svg viewBox="0 0 16 16">{cells}</svg>'
        f'<figcaption>{escape(label)}</figcaption></figure>'
    )

references = "".join(
    picture(bits, names[str(identifier)])
    for bits, identifier in zip(examples, ids)
)
generated = "".join(
    picture(
        sample,
        f"Sample {i + 1} · nearest: {names[str(ids[nearest[i]])]} "
        f"({agreements[i]:.1%} pixel agreement)",
    )
    for i, sample in enumerate(samples)
)
page = """<!doctype html><meta charset="utf-8">
<title>Singularity Canvas — 16-symbol diagnostic</title>
<style>
body{background:#101014;color:#eee;font:16px system-ui;margin:32px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:20px}
figure{margin:0}svg{width:100%;max-width:220px;display:block}
figcaption{margin-top:8px;color:#ccc}
p{max-width:900px;line-height:1.5}
</style>
<h1>16-symbol learning diagnostic</h1>
<p>Separate model trained only on these 16 examples.
This is an overfitting test, not evidence of generalization.</p>
<h2>Training examples</h2><div class="grid">"""
page += references + """</div><h2>Actual Torx samples</h2>
<p>16 independent random starts, 2,000 sweeps each, temperature 1,
no locked clues. Every final sample is shown without filtering.
Nearest-symbol labels are assigned afterward for inspection;
they do not alter the sampled pixels. Pixel agreement is not a
measure of recognizability or proof of convergence.</p><div class="grid">"""
page += generated + "</div>"

(output / "preview.html").write_text(page)
(output / "results.json").write_text(json.dumps({
    "purpose": "16-symbol overfitting diagnostic",
    "model": destination.name,
    "training_ids": ids.tolist(),
    "selection_seed": 74,
    "training_seed": 75,
    "initial_state_seed": 76,
    "sampling_seed": 77,
    "updates": 3000,
    "persistent_chains": 64,
    "sweeps_per_update": 20,
    "learning_rate": 0.02,
    "momentum": 0.5,
    "weight_decay": 0.0001,
    "sampling_sweeps": 2000,
    "temperature": 1.0,
    "samples": samples.tolist(),
    "nearest_pixel_agreements": agreements.tolist(),
}, indent=2))
print(f"Mean nearest-example pixel agreement: {agreements.mean():.1%}")
print("Saved model:", destination)
print("Saved preview:", output / "preview.html")
