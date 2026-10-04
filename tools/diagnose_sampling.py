from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from engine.rbm import sample_chain

root = Path(__file__).resolve().parents[1]
with np.load(root / "models/rbm_v1.npz", allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}
with np.load(root / "data/symbols/train.npz", allow_pickle=False) as data:
    training = data["bits"].astype(np.float32)

rng = np.random.default_rng(47)
symbol_starts = training[rng.choice(len(training), 16, replace=False)]
random_starts = rng.integers(0, 2, (16, 256)).astype(np.float32)
starts = jnp.asarray(np.concatenate([symbol_starts, random_starts]))


def free_energy(pixels):
    hidden_logits = pixels @ params["weights"] + params["hidden_bias"]
    return (
        -(pixels @ params["visible_bias"])
        - jax.nn.softplus(hidden_logits).sum(axis=-1)
    )


@eqx.filter_jit
def run(initials, keys):
    return jax.vmap(lambda initial, key: sample_chain(
        params,
        initial,
        jnp.zeros(256, dtype=jnp.bool_),
        jnp.zeros(256, dtype=jnp.float32),
        key,
        steps=2000,
    )[1])(initials, keys)


print(f"Training-set mean green fraction: {training.mean():.1%}", flush=True)
print(f"Training-set mean free energy: "
      f"{float(free_energy(jnp.asarray(training)).mean()):.2f}", flush=True)

history = run(starts, jax.random.split(jax.random.key(48), 32))
history.block_until_ready()

for label, selection in (
    ("Symbol starts", slice(0, 16)),
    ("Random starts", slice(16, 32)),
):
    print(f"\n{label}:")
    for step in (0, 1, 5, 50, 500, 2000):
        pixels = starts[selection] if step == 0 else history[selection, step - 1]
        print(
            f"Sweep {step:4d}: green {float(pixels.mean()):.1%}, "
            f"mean free energy {float(free_energy(pixels).mean()):.2f}"
        )

print("\nLower free energy means the model favors those states.")
print("These summaries diagnose behavior; they do not prove convergence.")
