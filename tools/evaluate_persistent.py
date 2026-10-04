import json
from html import escape
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from engine.rbm import sample_chain

ROOT = Path(__file__).resolve().parents[1]
with np.load(ROOT / "models/rbm_v2_pcd.npz", allow_pickle=False) as data:
    params = {name: jnp.asarray(data[name]) for name in data.files}
with np.load(ROOT / "data/symbols/validation.npz", allow_pickle=False) as data:
    patterns, ids = data["bits"], data["ids"]

catalogue = json.loads((ROOT / "data/symbols/catalogue.json").read_text())
names = {p["id"]: p["names"][0] for p in catalogue["patterns"]}
rng = np.random.default_rng(46)
selected = rng.choice(len(patterns), 4, replace=False)


@eqx.filter_jit
def generate(initials, keys, mask, clues):
    return jax.vmap(
        lambda initial, key: sample_chain(
            params, initial, mask, clues, key, steps=500
        )[0]
    )(initials, keys)


def picture(bits, label):
    colors = {-1: "#34343f", 0: "#7028ba", 1: "#62ef62"}
    cells = "".join(
        f'<rect x="{i % 16}" y="{i // 16}" width="1" height="1" '
        f'fill="{colors[int(value)]}"/>'
        for i, value in enumerate(bits)
    )
    return (
        f'<figure><svg viewBox="0 0 16 16" role="img" '
        f'aria-label="{escape(label)}">{cells}</svg>'
        f'<figcaption>{escape(label)}</figcaption></figure>'
    )


sections, records = [], []
for case, index in enumerate(selected):
    target = patterns[index]
    clue_order = rng.permutation(256)
    for count in (5, 32, 128):
        mask = np.zeros(256, dtype=bool)
        mask[clue_order[:count]] = True
        clues = np.where(mask, target, 0).astype(np.float32)
        initial = rng.integers(0, 2, size=(4, 256)).astype(np.float32)
        keys = jax.random.split(jax.random.key(1000 + case * 256 + count), 4)
        samples = np.asarray(generate(
            jnp.asarray(initial), keys, jnp.asarray(mask), jnp.asarray(clues)
        )).astype(np.uint8)
        assert np.all(samples[:, mask] == target[mask])

        accuracy = float((samples[:, ~mask] == target[~mask]).mean())
        baseline = float((target[~mask] == 0).mean())
        name = names[str(ids[index])]
        panels = picture(target, "Reference: withheld from sampler")
        panels += picture(np.where(mask, target.astype(np.int16), -1), f"{count} fixed clues")
        panels += "".join(
            picture(sample, f"Sample {i + 1}")
            for i, sample in enumerate(samples)
        )
        sections.append(
            f'<section><h2>{escape(name)} · {count} clues</h2>'
            f'<p>Unlocked-pixel agreement: {accuracy:.1%}. '
            f'All-purple baseline: {baseline:.1%}.</p>'
            f'<div class="row">{panels}</div></section>'
        )
        records.append({
            "target_id": str(ids[index]),
            "clue_count": count,
            "locked_indices": np.flatnonzero(mask).tolist(),
            "clue_values": target[mask].tolist(),
            "samples": samples.tolist(),
            "unlocked_agreement": accuracy,
            "purple_baseline": baseline,
        })
        print(f"{name}, {count} clues: agreement {accuracy:.1%}, "
              f"purple baseline {baseline:.1%}", flush=True)

output = ROOT / "data/evaluation"
output.mkdir(parents=True, exist_ok=True)
(output / "rbm_v2_pcd_samples.json").write_text(json.dumps({
    "model": "rbm_v2_pcd",
    "seed": 46,
    "sweeps_per_chain": 500,
    "temperature": 1.0,
    "records": records,
}))

page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Singularity Canvas — Sample Evaluation</title>
<style>
body{background:#101014;color:#e4e4eb;font:16px system-ui;
max-width:1200px;margin:32px auto;padding:0 24px}
h1{font-size:28px}h2{font-size:20px}p{line-height:1.5}
section{border-top:1px solid #444452;padding:20px 0}
.row{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:16px}
figure{margin:0}svg{width:100%;shape-rendering:crispEdges}
figcaption{font-size:14px;margin-top:8px}
@media(max-width:800px){.row{grid-template-columns:repeat(3,minmax(0,1fr))}}
</style>
<h1>Singularity Canvas · Actual Torx Samples</h1>
<p>Four random starts per case, 500 sweeps each. Green = 1;
purple = 0; gray = unknown clue.</p>
<p>No best-sample selection. With few clues, a valid alternative can differ
from the reference. Pixel agreement alone does not measure recognizability
or establish that sampling has converged.</p>
""" + "".join(sections) + "</html>"
preview = output / "rbm_v2_pcd_samples.html"
preview.write_text(page)
print("Saved:", preview)
