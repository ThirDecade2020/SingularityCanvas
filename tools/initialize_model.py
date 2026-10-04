import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
with np.load(root / "data/symbols/train.npz", allow_pickle=False) as data:
    training = data["bits"].astype(np.float32)

seed = 43
hidden_units = 128
rng = np.random.default_rng(seed)

# Initial pixel biases use training data only.
frequency = np.clip(training.mean(axis=0), 0.01, 0.99)
visible_bias = np.log(frequency / (1 - frequency)).astype(np.float32)
hidden_bias = np.zeros(hidden_units, dtype=np.float32)
weights = rng.normal(
    0, 0.01, size=(256, hidden_units)
).astype(np.float32)

folder = root / "models"
folder.mkdir(exist_ok=True)
destination = folder / "rbm_initial.npz"
if destination.exists():
    raise SystemExit("Initial model already exists; left unchanged.")

np.savez_compressed(
    destination,
    weights=weights,
    visible_bias=visible_bias,
    hidden_bias=hidden_bias,
)
manifest_bytes = (root / "data/symbols/split_manifest.json").read_bytes()
metadata = {
    "architecture": "binary RBM",
    "visible_units": 256,
    "hidden_units": hidden_units,
    "seed": seed,
    "status": "initialized, not trained",
    "training_patterns": len(training),
    "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
}
(folder / "rbm_initial.json").write_text(json.dumps(metadata, indent=2))

print("Weight matrix:", weights.shape)
print("Total parameters:", weights.size + visible_bias.size + hidden_bias.size)
print("Saved:", destination)
print("Status: initialized, not trained.")
