import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp
import equinox as eqx
import jax
import jax.numpy as jnp
from torx import JaxPRNGSampler
from engine.rbm import ConditionalUpdate, VISIBLE, HIDDEN

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
model_path = root / "models/rbm_diagnostic_exact16_refined.npz"

with np.load(model_path, allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}
    W = data["weights"][:, :16].astype(np.float64)
    vb = data["visible_bias"].astype(np.float64)
    hb = data["hidden_bias"][:16].astype(np.float64)
    assert np.all(data["weights"][:, 16:] == 0)
    assert np.all(data["hidden_bias"][16:] == 0)

hidden = np.array(list(product((0, 1), repeat=16)), dtype=np.float64)
pixel_logits = hidden @ W.T + vb
hidden_padded = jnp.zeros((65536, HIDDEN)).at[:, :16].set(
    jnp.asarray(hidden, dtype=jnp.float32)
)
categorical = JaxPRNGSampler()
visible_update = ConditionalUpdate(True)

@eqx.filter_jit
def draw_many(log_weights, mask, clues, key):
    def draw(draw_key):
        hidden_key, visible_key = jax.random.split(draw_key)
        index = categorical.categorical(hidden_key, log_weights)
        return visible_update.sample(
            visible_key,
            {
                "state": hidden_padded[index],
                "temperature": jnp.array(1.0),
                "locked": mask,
                "clues": clues,
            },
            params,
        )
    keys = jax.random.split(key, 4096).reshape(64, 64)
    def batch_step(carry, batch_keys):
        return carry, jax.vmap(draw)(batch_keys)
    _, batches = jax.lax.scan(batch_step, None, keys)
    return batches.reshape(4096, VISIBLE)

results = {}
saved_samples = {}

diagnostic = json.loads((folder / "results.json").read_text())
catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
lookup = {p["id"]: p for p in catalogue["patterns"]}
targets = [lookup[i] for i in diagnostic["training_ids"]]
clue_rng = np.random.default_rng(110)

for case, target in enumerate(targets):
    label = target["id"]
    mask = np.zeros(VISIBLE, dtype=bool)
    clues = np.zeros(VISIBLE, dtype=np.float32)
    target_bits = np.array(target["bits"], dtype=np.uint8)
    green = clue_rng.choice(np.flatnonzero(target_bits == 1), 3, replace=False)
    purple = clue_rng.choice(np.flatnonzero(target_bits == 0), 2, replace=False)
    positions = np.concatenate((green, purple))
    mask[positions] = True
    clues[positions] = target_bits[positions]
    assert np.all(target_bits[mask] == clues[mask])
    print("\nTarget:", target["names"][0], flush=True)

    # Exact hidden posterior given fixed visible clues:
    # sum over free pixels, substitute fixed values for locked pixels.
    log_weights = (
        hidden @ hb
        + np.logaddexp(0, pixel_logits[:, ~mask]).sum(axis=1)
        + pixel_logits[:, mask] @ clues[mask]
    )
    probabilities = np.exp(log_weights - logsumexp(log_weights))
    expected_green = probabilities @ expit(pixel_logits)
    expected_green[mask] = clues[mask]

    samples = np.asarray(draw_many(
        jnp.asarray(log_weights - log_weights.max(), dtype=jnp.float32),
        jnp.asarray(mask),
        jnp.asarray(clues),
        jax.random.key(100 + case),
    )).astype(np.uint8)

    assert np.all(samples[:, mask] == clues[mask])
    errors = np.abs(samples.mean(axis=0) - expected_green)
    print(f"\n{label}: 4,096 direct Torx samples")
    print(f"Mean pixel-probability error: {errors.mean():.3%}")
    print(f"Largest pixel-probability error: {errors.max():.3%}")
    assert errors.max() < 0.04, "Outside diagnostic tolerance."

    results[label] = {
        "seed": 100 + case,
        "locked_indices": np.flatnonzero(mask).tolist(),
        "clue_values": clues[mask].tolist(),
        "mean_pixel_probability_error": float(errors.mean()),
        "maximum_pixel_probability_error": float(errors.max()),
        "expected_green": expected_green.tolist(),
    }
    saved_samples[label] = samples

np.savez_compressed(folder / "supported_clue16_refined_samples.npz", **saved_samples)
(folder / "supported_clue16_refined_results.json").write_text(json.dumps({
    "model": model_path.name,
    "method": "exact hidden posterior; Torx categorical and visible sampling",
    "samples_per_case": 4096,
    "temperature": 1.0,
    "results": results,
}, indent=2))
print("\nPASS: pixel marginals agree with exact model probabilities.")
print("PASS: all five clues remain fixed.")
print("No burn-in or Gibbs mixing required by this direct method.")
print("This checks pixel marginals, not every joint-state probability.")


from html import escape

def picture(bits, label):
    colors = {-1: "#34343f", 0: "#7028ba", 1: "#62ef62"}
    cells = "".join(
        f'<rect x="{i % 16}" y="{i // 16}" width="1" height="1" '
        f'fill="{colors[int(bit)]}"/>'
        for i, bit in enumerate(bits)
    )
    return (
        f'<figure><svg viewBox="0 0 16 16">{cells}</svg>'
        f'<figcaption>{escape(label)}</figcaption></figure>'
    )

sections = []
for target in targets:
    identifier = target["id"]
    record = results[identifier]
    preview_clues = np.full(256, -1, dtype=np.int16)
    preview_clues[record["locked_indices"]] = record["clue_values"]
    panels = picture(target["bits"], "Source of clues; not passed to sampler")
    panels += picture(preview_clues, "Five locked clues")
    panels += "".join(
        picture(sample, f"Sample {i + 1}")
        for i, sample in enumerate(saved_samples[identifier][:4])
    )
    sections.append(
        f'<h2>{escape(target["names"][0])}</h2><div class="grid">{panels}</div>'
    )

page = """<!doctype html><meta charset="utf-8">
<title>Singularity Canvas — Supported Clues</title>
<style>
body{background:#101014;color:#eee;font:16px system-ui;margin:32px}
.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:16px}
figure{margin:0}svg{width:100%;display:block}
figcaption{color:#ccc;margin-top:8px}h2{margin-top:36px}
@media(max-width:900px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
</style>
<h1>Refined 16-hidden-unit model: all 16 diagnostic symbols</h1>
<p>Each case uses three green and two purple clues from a training example.
Only those five values and positions reach the sampler.
The first four draws are shown without selection or cleanup.</p>
<p>Five clues may support multiple symbols. This tests completion quality
on familiar examples, not generalization to unseen symbols.</p>
"""
preview = folder / "supported_clues16_refined_preview.html"
preview.write_text(page + "".join(sections))
print("Saved:", preview)
