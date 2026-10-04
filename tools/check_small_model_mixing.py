import json
from pathlib import Path
from collections import Counter
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from engine.rbm import sample_chain, VISIBLE

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
metadata = json.loads((folder / "results.json").read_text())
ids = metadata["training_ids"]

with np.load(root / "data/symbols/train.npz", allow_pickle=False) as data:
    lookup = {
        str(identifier): bits
        for identifier, bits in zip(data["ids"], data["bits"])
    }
examples = np.stack([lookup[identifier] for identifier in ids])

catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
names_by_id = {p["id"]: p["names"][0] for p in catalogue["patterns"]}
names = [names_by_id[identifier] for identifier in ids]

with np.load(root / "models/rbm_diagnostic_16.npz",
             allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}

random_starts = jax.random.bernoulli(
    jax.random.key(80), 0.5, (16, VISIBLE)
).astype(jnp.float32)
states = jnp.concatenate((jnp.asarray(examples, dtype=jnp.float32),
                          random_starts))
mask = jnp.zeros(VISIBLE, dtype=jnp.bool_)
clues = jnp.zeros(VISIBLE)

@eqx.filter_jit
def advance(states, key):
    return jax.vmap(lambda state, k: sample_chain(
        params, state, mask, clues, k,
        temperature=1.0, steps=100,
    )[0])(states, jax.random.split(key, states.shape[0]))

def classify(states):
    distances = np.mean(
        states[:, None, :] != examples[None, :, :], axis=-1
    )
    nearest = distances.argmin(axis=1)
    agreement = 1 - distances.min(axis=1)
    # This threshold is an inspection aid, not a recognition guarantee.
    labels = np.where(agreement >= 0.90, nearest, -1)
    return nearest, agreement, labels

snapshots = [np.asarray(states).astype(np.uint8)]
records = []
key = jax.random.key(81)

def record(sweep, array):
    nearest, agreement, labels = classify(array)
    records.append({
        "sweep": sweep,
        "nearest_indices": nearest.tolist(),
        "agreements": agreement.tolist(),
        "labels_at_90_percent": labels.tolist(),
    })
    if sweep in (0, 100, 500, 2000, 10000):
        print(f"\nSweep {sweep}", flush=True)
        for group, region in (
            ("Symbol starts", slice(0, 16)),
            ("Random starts", slice(16, 32)),
        ):
            counts = Counter(labels[region].tolist())
            summary = ", ".join(
                f"{'unmatched' if label == -1 else names[label]}: {count}"
                for label, count in counts.most_common()
            )
            print(f"  {group}: {summary}", flush=True)

record(0, snapshots[0])
for block in range(1, 101):
    key, draw_key = jax.random.split(key)
    states = advance(states, draw_key)
    array = np.asarray(states).astype(np.uint8)
    snapshots.append(array)
    record(block * 100, array)

final = records[-1]
observed_labels = np.array([
    record["labels_at_90_percent"] for record in records[1:]
])
print("\nEach symbol-start chain at sweep 10,000:")
for index, name in enumerate(names):
    nearest = final["nearest_indices"][index]
    agreement = final["agreements"][index]
    visited = set(observed_labels[:, index].tolist()) - {-1}
    print(
        f"  {name} -> {names[nearest]} ({agreement:.1%}); "
        f"{len(visited)} distinct matches at recorded checkpoints"
    )

np.savez_compressed(
    folder / "mixing_checkpoints.npz",
    sweeps=np.arange(0, 10001, 100),
    states=np.stack(snapshots),
)
(folder / "mixing_results.json").write_text(json.dumps({
    "model": "rbm_diagnostic_16.npz",
    "training_ids": ids,
    "names": names,
    "random_start_seed": 80,
    "sampling_seed": 81,
    "temperature": 1.0,
    "match_threshold": 0.90,
    "checkpoint_interval": 100,
    "records": records,
}, indent=2))
print("\nSaved mixing results and checkpoint states.")
print("Checkpoint matches can miss transitions between checkpoints.")
print("This diagnostic does not prove equilibrium or symbol probabilities.")
