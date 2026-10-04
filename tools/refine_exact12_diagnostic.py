import json
from itertools import product
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from engine.rbm import VISIBLE, HIDDEN

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
metadata = json.loads((folder / "results.json").read_text())
ids = metadata["training_ids"]
destination = root / "models/rbm_diagnostic_exact12_refined.npz"
if destination.exists():
    raise SystemExit("Exact diagnostic model already exists; left unchanged.")

catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
lookup = {p["id"]: p for p in catalogue["patterns"]}
data = jnp.asarray([lookup[i]["bits"] for i in ids], dtype=jnp.float32)
hidden_states = jnp.array(
    list(product((0, 1), repeat=12)), dtype=jnp.float32
)

frequency = jnp.clip(data.mean(axis=0), 0.01, 0.99)
params = {
    "weights": 0.01 * jax.random.normal(jax.random.key(95), (VISIBLE, 12)),
    "visible_bias": jnp.log(frequency / (1 - frequency)),
    "hidden_bias": jnp.zeros(12),
}

def log_partition(params):
    # Sum visible states analytically, enumerate all hidden states.
    visible_logits = (
        hidden_states @ params["weights"].T + params["visible_bias"]
    )
    hidden_log_weights = (
        hidden_states @ params["hidden_bias"]
        + jax.nn.softplus(visible_logits).sum(axis=1)
    )
    return jax.scipy.special.logsumexp(hidden_log_weights)

def loss(params):
    data_log_weights = (
        data @ params["visible_bias"]
        + jax.nn.softplus(
            data @ params["weights"] + params["hidden_bias"]
        ).sum(axis=1)
    )
    return log_partition(params) - data_log_weights.mean()

# Continue from saved weights; restart Adam moments.
parent_path = root / "models/rbm_diagnostic_exact12.npz"
with np.load(parent_path, allow_pickle=False) as saved:
    params = {
        "weights": jnp.asarray(saved["weights"][:, :12]),
        "visible_bias": jnp.asarray(saved["visible_bias"]),
        "hidden_bias": jnp.asarray(saved["hidden_bias"][:12]),
    }

first = jax.tree.map(jnp.zeros_like, params)
second = jax.tree.map(jnp.zeros_like, params)

@eqx.filter_jit
def train_step(params, first, second, step):
    value, gradients = jax.value_and_grad(loss)(params)
    first = jax.tree.map(
        lambda moment, g: 0.9 * moment + 0.1 * g, first, gradients
    )
    second = jax.tree.map(
        lambda moment, g: 0.999 * moment + 0.001 * g * g,
        second, gradients,
    )
    params = jax.tree.map(
        lambda p, m, v: p - 0.005 * (m / (1 - 0.9 ** step))
        / (jnp.sqrt(v / (1 - 0.999 ** step)) + 1e-8),
        params, first, second,
    )
    return params, first, second, value

print("16 symbols; 12 active hidden units; exact likelihood training.")
print(f"Initial mean negative log likelihood: {float(loss(params)):.4f}",
      flush=True)

reports = []
for step in range(1, 3001):
    params, first, second, value = train_step(
        params, first, second, jnp.asarray(step, dtype=jnp.float32)
    )
    if step % 500 == 0:
        current = float(loss(params))
        assert np.isfinite(current), "Nonfinite training loss."
        reports.append({"step": step, "mean_nll": current})
        print(f"Update {step}: mean NLL = {current:.4f}", flush=True)

# Pad inactive hidden units so our existing Torx engine can load it.
# Their zero connections and biases add only a constant to log Z.
padded_weights = np.zeros((VISIBLE, HIDDEN), dtype=np.float32)
padded_weights[:, :12] = np.asarray(params["weights"])
padded_hidden_bias = np.zeros(HIDDEN, dtype=np.float32)
padded_hidden_bias[:12] = np.asarray(params["hidden_bias"])
np.savez_compressed(
    destination,
    weights=padded_weights,
    visible_bias=np.asarray(params["visible_bias"]),
    hidden_bias=padded_hidden_bias,
)
destination.with_suffix(".json").write_text(json.dumps({
    "purpose": "exact-likelihood diagnostic, not full-catalogue model",
    "training_ids": ids,
    "active_hidden_units": 12,
    "stored_hidden_units": HIDDEN,
    "seed": 95,
    "updates": 3000,
    "optimizer": "Adam",
    "parent_model": parent_path.name,
    "parent_updates": 3000,
    "total_updates": 6000,
    "optimizer_state_restarted": True,
    "learning_rate": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "negative_phase": "exact enumeration of 4096 hidden configurations",
    "reports": reports,
}, indent=2))
print("Saved:", destination)
print("Uniform probability on exactly 16 symbols would give NLL:",
      round(float(np.log(16)), 4))
print("Next: compare exact model probabilities with Torx samples.")
