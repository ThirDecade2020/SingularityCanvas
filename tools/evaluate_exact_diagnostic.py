import json
from itertools import product
from pathlib import Path
from html import escape
import numpy as np
from scipy.special import expit, logsumexp
import equinox as eqx
import jax
import jax.numpy as jnp
from engine.rbm import sample_chain, VISIBLE

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
model_path = root / "models/rbm_diagnostic_exact8.npz"
metadata = json.loads(model_path.with_suffix(".json").read_text())
ids = metadata["training_ids"]
catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
lookup = {p["id"]: p for p in catalogue["patterns"]}
examples = np.array([lookup[i]["bits"] for i in ids], dtype=np.uint8)
names = [lookup[i]["names"][0] for i in ids]

with np.load(model_path, allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}
    W = data["weights"][:, :8].astype(np.float64)
    vb = data["visible_bias"].astype(np.float64)
    hb = data["hidden_bias"][:8].astype(np.float64)
    assert np.all(data["weights"][:, 8:] == 0)
    assert np.all(data["hidden_bias"][8:] == 0)

# Exact hidden marginal: visible states are summed out analytically.
hidden = np.array(list(product((0, 1), repeat=8)), dtype=np.float64)
visible_logits = hidden @ W.T + vb
hidden_log_weights = (
    hidden @ hb + np.logaddexp(0, visible_logits).sum(axis=1)
)
log_z = logsumexp(hidden_log_weights)
hidden_probabilities = np.exp(hidden_log_weights - log_z)

# Independent model samples: draw h from its exact marginal, then v|h.
# This is a reference sampler, explicitly separate from Torx Gibbs.
N = 256
rng = np.random.default_rng(96)
chosen = rng.choice(len(hidden), size=N, p=hidden_probabilities)
reference = (
    rng.random((N, VISIBLE)) < expit(visible_logits[chosen])
).astype(np.uint8)

mask = jnp.zeros(VISIBLE, dtype=jnp.bool_)
clues = jnp.zeros(VISIBLE)
initials = jax.random.bernoulli(
    jax.random.key(97), 0.5, (N, VISIBLE)
).astype(jnp.float32)

@eqx.filter_jit
def generate(initials):
    return jax.vmap(lambda initial, key: sample_chain(
        params, initial, mask, clues, key, temperature=1.0, steps=2000
    )[0])(initials, jax.random.split(jax.random.key(98), N))

print("Generating Torx samples from 256 random starts...", flush=True)
torx_samples = np.asarray(generate(initials)).astype(np.uint8)

center_log_probabilities = (
    examples @ vb
    + np.logaddexp(0, examples @ W + hb).sum(axis=1)
    - log_z
)
print(f"Exact mean NLL: {-center_log_probabilities.mean():.4f}")
print(
    "Exact total probability of the 16 unmodified training images: "
    f"{np.exp(logsumexp(center_log_probabilities)):.6e}"
)

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

sections = [
    "<h2>Training examples</h2><div class='grid'>"
    + "".join(picture(bits, name) for bits, name in zip(examples, names))
    + "</div>"
]
reports = {}
for label, samples in (
    ("Independent reference samples", reference),
    ("Torx Gibbs samples — 2,000 sweeps", torx_samples),
):
    distances = np.mean(
        samples[:, None, :] != examples[None, :, :], axis=-1
    )
    nearest = distances.argmin(axis=1)
    agreement = 1 - distances.min(axis=1)
    matched = agreement >= 0.90
    coverage = len(set(nearest[matched].tolist()))
    print(f"\n{label}")
    print(f"Mean nearest-example pixel agreement: {agreement.mean():.1%}")
    print(f"Samples with >=90% agreement: {matched.mean():.1%}")
    print(f"Distinct matched symbols: {coverage}/16")

    reports[label] = {
        "mean_nearest_agreement": float(agreement.mean()),
        "fraction_at_least_90_percent": float(matched.mean()),
        "distinct_matches": coverage,
        "nearest_indices": nearest.tolist(),
        "agreements": agreement.tolist(),
    }
    panels = "".join(
        picture(
            samples[i],
            f"{i + 1}: nearest {names[nearest[i]]} ({agreement[i]:.1%})",
        )
        for i in range(16)
    )
    sections.append(
        f"<h2>{escape(label)}</h2><p>First 16 of 256 samples, "
        f"without selection or cleanup.</p><div class='grid'>{panels}</div>"
    )

page = """<!doctype html><meta charset="utf-8">
<title>Singularity Canvas — Exact model comparison</title>
<style>
body{background:#101014;color:#eee;font:16px system-ui;margin:32px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:20px}
figure{margin:0}svg{width:100%;max-width:220px;display:block}
figcaption{margin-top:8px;color:#ccc}p{max-width:900px;line-height:1.5}
</style>
<h1>Same model, two sampling methods</h1>
<p>The reference method draws independent samples using exact hidden-state
enumeration. Torx runs Gibbs chains from random starts.
Neither method uses locked clues. Nearest-symbol labels are assigned
after sampling and do not change the pixels.</p>
"""
(folder / "exact_comparison.html").write_text(page + "".join(sections))
np.savez_compressed(
    folder / "exact_comparison_samples.npz",
    reference=reference,
    torx=torx_samples,
    center_log_probabilities=center_log_probabilities,
)
(folder / "exact_comparison.json").write_text(json.dumps({
    "model": model_path.name,
    "sample_count_per_method": N,
    "reference_seed": 96,
    "initial_state_seed": 97,
    "torx_sampling_seed": 98,
    "torx_sweeps": 2000,
    "temperature": 1.0,
    "reports": reports,
}, indent=2))
print("\nSaved comparison and all 512 samples; model unchanged.")
