import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
metadata = json.loads((folder / "results.json").read_text())
ids = metadata["training_ids"]
catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
lookup = {p["id"]: p for p in catalogue["patterns"]}
examples = np.array([lookup[i]["bits"] for i in ids], dtype=np.float64)
names = [lookup[i]["names"][0] for i in ids]

with np.load(root / "models/rbm_diagnostic_16.npz",
             allow_pickle=False) as data:
    weights = data["weights"].astype(np.float64)
    visible_bias = data["visible_bias"].astype(np.float64)
    hidden_bias = data["hidden_bias"].astype(np.float64)

def log_weight(states):
    # Exact visible-state weight at temperature 1, summing hidden states.
    return (
        states @ visible_bias
        + np.logaddexp(0, states @ weights + hidden_bias).sum(axis=-1)
    )

def logsumexp(values):
    maximum = values.max()
    return float(maximum + np.log(np.exp(values - maximum).sum()))

def normalize(log_weights):
    return np.exp(log_weights - logsumexp(log_weights))

distances = np.count_nonzero(
    examples[:, None, :] != examples[None, :, :], axis=-1
)
np.fill_diagonal(distances, examples.shape[1] + 1)
minimum_distance = int(distances.min())

# Radius-one neighborhoods are disjoint if centers differ by >2 pixels.
radius = 1 if minimum_distance > 2 else 0
center_logs = log_weight(examples)
region_logs = []

for example in examples:
    if radius == 1:
        states = np.repeat(example[None, :], 257, axis=0)
        rows = np.arange(1, 257)
        columns = np.arange(256)
        states[rows, columns] = 1 - states[rows, columns]
    else:
        states = example[None, :]
    region_logs.append(logsumexp(log_weight(states)))

region_logs = np.array(region_logs)
center_shares = normalize(center_logs)
region_shares = normalize(region_logs)

print(f"Minimum distance between training symbols: {minimum_distance} pixels")
print(f"Neighborhood radius: {radius}")
print("Shares below are conditional on the enumerated states ONLY.")
print("They are not probabilities over the full 256-bit image space.\n")
print(f"{'Symbol':36} {'Centers only':>13} {'Neighborhoods':>15}")

records = []
for index in np.argsort(-region_shares):
    print(
        f"{names[index]:36} "
        f"{center_shares[index]:12.3%} "
        f"{region_shares[index]:14.3%}"
    )
    records.append({
        "id": ids[index],
        "name": names[index],
        "center_log_weight": float(center_logs[index]),
        "region_log_weight": float(region_logs[index]),
        "share_among_centers": float(center_shares[index]),
        "share_among_enumerated_regions": float(region_shares[index]),
    })

spread = float((region_logs.max() - region_logs.min()) / np.log(10))
print(f"\nLargest/smallest neighborhood weight ratio: 10^{spread:.2f}")
print("Exact relative weights for these regions; no sampling involved.")
print("More distant variations and other images are not included.")

(folder / "exact_local_weights.json").write_text(json.dumps({
    "model": "rbm_diagnostic_16.npz",
    "temperature": 1.0,
    "neighborhood_radius": radius,
    "minimum_center_distance": minimum_distance,
    "normalization_scope": "enumerated disjoint neighborhoods only",
    "records": records,
}, indent=2))
print("Saved exact_local_weights.json; model unchanged.")
