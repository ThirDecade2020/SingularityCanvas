import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from engine.rbm import SWEEP, VISIBLE

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
metadata = json.loads((folder / "results.json").read_text())
ids = metadata["training_ids"]
catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
lookup = {p["id"]: p for p in catalogue["patterns"]}
examples = jnp.asarray([lookup[i]["bits"] for i in ids])
names = [lookup[i]["names"][0] for i in ids]

with np.load(root / "models/rbm_diagnostic_16.npz",
             allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}

LADDERS, REPLICAS, ROUNDS = 16, 8, 2000
temperatures = jnp.geomspace(1.0, 4.0, REPLICAS)
mask = jnp.zeros(VISIBLE, dtype=jnp.bool_)
clues = jnp.zeros(VISIBLE)
initial = jax.random.bernoulli(
    jax.random.key(90), 0.5, (LADDERS, REPLICAS, VISIBLE)
).astype(jnp.float32)

def update(states, temps, key):
    keys = jax.random.split(key, states.shape[0])
    return jax.vmap(lambda state, temp, k: SWEEP.sample(
        k, {
            "pixels": state,
            "temperature": temp,
            "locked": mask,
            "clues": clues,
        }, {"model": params}
    ))(states, temps, keys)

def log_weight(states, temperature):
    # Exact unnormalized visible marginal at this temperature.
    # Hidden units are summed out analytically.
    logits = states @ params["weights"] + params["hidden_bias"]
    return (
        states @ params["visible_bias"] / temperature
        + jax.nn.softplus(logits / temperature).sum(axis=-1)
    )

def classify(states):
    distances = jnp.mean(
        states[:, None, :] != examples[None, :, :], axis=-1
    )
    agreement = 1 - distances.min(axis=1)
    labels = jnp.where(agreement >= 0.90, distances.argmin(axis=1), -1)
    return labels, agreement

@eqx.filter_jit
def run_tempering(initial, key):
    def step(states, round_key):
        keys = jax.random.split(round_key, REPLICAS)
        flat = update(
            states.reshape(-1, VISIBLE),
            jnp.tile(temperatures, LADDERS),
            keys[0],
        )
        states = flat.reshape(LADDERS, REPLICAS, VISIBLE)
        accepted = []
        # Sequential adjacent exchanges, each with a Metropolis test.
        for edge in range(REPLICAS - 1):
            left, right = states[:, edge], states[:, edge + 1]
            t_left, t_right = temperatures[edge], temperatures[edge + 1]
            log_ratio = (
                log_weight(right, t_left) + log_weight(left, t_right)
                - log_weight(left, t_left) - log_weight(right, t_right)
            )
            accept = jnp.log(jax.random.uniform(
                keys[edge + 1], (LADDERS,),
                minval=1e-7, maxval=1.0,
            )) < jnp.minimum(0.0, log_ratio)
            states = states.at[:, edge].set(
                jnp.where(accept[:, None], right, left)
            )
            states = states.at[:, edge + 1].set(
                jnp.where(accept[:, None], left, right)
            )
            accepted.append(accept.mean())
        labels, agreement = classify(states[:, 0])
        return states, (labels, agreement, jnp.stack(accepted))

    return jax.lax.scan(
        step, initial, jax.random.split(key, ROUNDS)
    )

@eqx.filter_jit
def run_ordinary(initial, key):
    def step(states, round_key):
        def sweep(states, draw_key):
            return update(states, jnp.ones(LADDERS), draw_key), None
        states, _ = jax.lax.scan(
            sweep, states, jax.random.split(round_key, REPLICAS)
        )
        return states, classify(states)

    return jax.lax.scan(
        step, initial, jax.random.split(key, ROUNDS)
    )

reports = {}
arrays = {"temperatures": np.asarray(temperatures)}

for method in ("ordinary", "tempering"):
    print(f"\nRunning {method}...", flush=True)
    start = time.perf_counter()
    if method == "ordinary":
        final, history = run_ordinary(initial[:, 0], jax.random.key(91))
        labels, agreement = history
    else:
        final, history = run_tempering(initial, jax.random.key(92))
        labels, agreement, swaps = history
    final.block_until_ready()
    elapsed = time.perf_counter() - start
    labels, agreement = np.asarray(labels), np.asarray(agreement)

    # Equal sweep-budget warm-up: first 25% of each run.
    retained = labels[ROUNDS // 4:]
    visited = set(retained.ravel().tolist()) - {-1}
    per_chain = [
        len(set(retained[:, i].tolist()) - {-1})
        for i in range(LADDERS)
    ]
    counts = Counter(retained.ravel().tolist())
    occupancy = {
        "unmatched" if label == -1 else names[label]: count / retained.size
        for label, count in counts.most_common()
    }

    print(f"Runtime including compilation: {elapsed:.2f} seconds")
    print(f"Distinct matched symbols after warm-up: {len(visited)}/16")
    print(f"Median distinct matches per chain: {np.median(per_chain):.1f}")
    print(f"Checkpoints with >=90% pixel agreement: {(retained >= 0).mean():.1%}")
    print("Checkpoint occupancy:")
    for name, fraction in occupancy.items():
        print(f"  {name}: {fraction:.1%}")

    reports[method] = {
        "seconds_including_compilation": elapsed,
        "distinct_matches": len(visited),
        "distinct_matches_per_chain": per_chain,
        "checkpoint_occupancy": occupancy,
    }
    arrays[f"{method}_labels"] = labels
    arrays[f"{method}_agreements"] = agreement
    arrays[f"{method}_final"] = np.asarray(final)

    if method == "tempering":
        rates = np.asarray(swaps).mean(axis=0)
        reports[method]["exchange_acceptance"] = rates.tolist()
        print("Adjacent-temperature exchange acceptance:")
        for i, rate in enumerate(rates):
            print(
                f"  {float(temperatures[i]):.2f} <-> "
                f"{float(temperatures[i + 1]):.2f}: {rate:.1%}"
            )

np.savez_compressed(folder / "tempering_comparison.npz", **arrays)
(folder / "tempering_comparison.json").write_text(json.dumps({
    "model": "rbm_diagnostic_16.npz",
    "training_ids": ids,
    "initial_seed": 90,
    "ordinary_seed": 91,
    "tempering_seed": 92,
    "temperatures": np.asarray(temperatures).tolist(),
    "rounds": ROUNDS,
    "warmup_rounds": ROUNDS // 4,
    "individual_sweeps_per_method": LADDERS * REPLICAS * ROUNDS,
    "reports": reports,
}, indent=2))
print("\nSaved comparison; trained model unchanged.")
print("Nearest matches are inspection labels, not verified recognition.")
print("Correlated checkpoint counts are not established equilibrium probabilities.")
print("Equal sweep counts do not mean equal runtime or hardware advantage.")
