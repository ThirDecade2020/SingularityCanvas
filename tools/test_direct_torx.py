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
model_path = root / "models/rbm_diagnostic_exact8.npz"

with np.load(model_path, allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}
    W = data["weights"][:, :8].astype(np.float64)
    vb = data["visible_bias"].astype(np.float64)
    hb = data["hidden_bias"][:8].astype(np.float64)
    assert np.all(data["weights"][:, 8:] == 0)
    assert np.all(data["hidden_bias"][8:] == 0)

hidden = np.array(list(product((0, 1), repeat=8)), dtype=np.float64)
pixel_logits = hidden @ W.T + vb
hidden_padded = jnp.zeros((256, HIDDEN)).at[:, :8].set(
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
    return jax.vmap(draw)(jax.random.split(key, 4096))

results = {}
saved_samples = {}

for case, label in enumerate(("unconstrained", "five_clues")):
    mask = np.zeros(VISIBLE, dtype=bool)
    clues = np.zeros(VISIBLE, dtype=np.float32)
    if case == 1:
        positions = np.array([17, 54, 119, 180, 238])
        mask[positions] = True
        clues[positions] = [1, 0, 1, 0, 1]

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

np.savez_compressed(folder / "direct_torx_samples.npz", **saved_samples)
(folder / "direct_torx_results.json").write_text(json.dumps({
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
