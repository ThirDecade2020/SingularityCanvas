import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
diagnostic = json.loads((folder / "results.json").read_text())
catalogue = json.loads((root / "data/symbols/catalogue.json").read_text())
lookup = {p["id"]: p for p in catalogue["patterns"]}
ids = diagnostic["training_ids"]
patterns = np.array([lookup[i]["bits"] for i in ids], dtype=np.uint8)

files = {
    "16 units": ("supported_clue16_results.json", "supported_clue16_samples.npz"),
    "8 units": ("supported_clue_results.json", "supported_clue_samples.npz"),
    "12 units": ("supported_clue12_results.json", "supported_clue12_samples.npz"),
    "12 refined": ("supported_clue12_refined_results.json", "supported_clue12_refined_samples.npz"),
}
reports = {}
previous_clues = {}

for model in ("8 units", "12 units", "12 refined", "16 units"):
    metadata_file, samples_file = files[model]
    metadata = json.loads((folder / metadata_file).read_text())
    records = []
    print(f"\n{model}")
    print(f"{'Clue source':35} {'Valid':>5} {'Mean errors':>12} {'<=8 errors':>12}")

    with np.load(folder / samples_file, allow_pickle=False) as saved:
        for identifier in ids:
            case = metadata["results"][identifier]
            positions = np.array(case["locked_indices"], dtype=int)
            values = np.array(case["clue_values"], dtype=np.uint8)
            signature = (positions.tolist(), values.tolist())
            if model == "8 units":
                previous_clues[identifier] = signature
            else:
                assert signature == previous_clues[identifier]

            compatible = np.all(patterns[:, positions] == values, axis=1)
            candidates = patterns[compatible]
            assert len(candidates) > 0

            samples = saved[identifier]
            assert np.all(samples[:, positions] == values)
            free = np.ones(256, dtype=bool)
            free[positions] = False
            errors = np.count_nonzero(
                samples[:, None, free] != candidates[None, :, free],
                axis=-1,
            ).min(axis=1)

            record = {
                "id": identifier,
                "name": lookup[identifier]["names"][0],
                "compatible_training_symbols": len(candidates),
                "mean_unlocked_pixel_errors": float(errors.mean()),
                "fraction_within_8_errors": float((errors <= 8).mean()),
                "fraction_exact_matches": float((errors == 0).mean()),
            }
            records.append(record)
            print(
                f"{record['name']:35} {len(candidates):5d} "
                f"{errors.mean():12.2f} {(errors <= 8).mean():12.1%}"
            )

    reports[model] = records
    print(
        "Across-case mean pixel errors:",
        round(float(np.mean([r["mean_unlocked_pixel_errors"] for r in records])), 2),
    )
    print(
        "Across-case fraction within 8 errors:",
        f"{np.mean([r['fraction_within_8_errors'] for r in records]):.1%}",
    )

(folder / "supported_capacity_comparison.json").write_text(
    json.dumps(reports, indent=2)
)
print("\nErrors count only unlocked pixels; lower is better.")
print("The eight-error threshold is a diagnostic choice, not a recognition score.")
print("All cases use training symbols; this does not measure generalization.")
