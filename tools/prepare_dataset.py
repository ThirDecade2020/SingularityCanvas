import hashlib
import json
from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parents[1] / "data" / "symbols"
source = (folder / "catalogue.json").read_bytes()
catalogue = json.loads(source)
patterns = catalogue["patterns"]

bits = np.array([p["bits"] for p in patterns], dtype=np.uint8)
ids = np.array([p["id"] for p in patterns])

assert bits.shape == (len(patterns), 256)
assert np.all((bits == 0) | (bits == 1))
assert len(set(ids.tolist())) == len(patterns)

seed = 42
order = np.random.default_rng(seed).permutation(len(patterns))
boundary = int(len(patterns) * 0.8)
splits = {
    "train": order[:boundary],
    "validation": order[boundary:],
}
assert set(splits["train"]).isdisjoint(set(splits["validation"]))

for name, indices in splits.items():
    np.savez_compressed(
        folder / f"{name}.npz",
        bits=bits[indices],
        ids=ids[indices],
    )
    print(f"{name}: {len(indices)} patterns, 256 bits each")

manifest = {
    "catalogue_sha256": hashlib.sha256(source).hexdigest(),
    "seed": seed,
    "numpy_version": np.__version__,
    "split_unit": "unique binary pattern",
    "augmentation_rule": "All variants inherit their parent's split.",
    "splits": {
        name: ids[indices].tolist()
        for name, indices in splits.items()
    },
}
(folder / "split_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)
print("Saved datasets and reproducible split manifest to:", folder)
