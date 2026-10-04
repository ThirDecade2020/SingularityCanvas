import json
from pathlib import Path
from html import escape
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / "data/evaluation/small_catalogue"
metadata = json.loads((folder / "direct_torx_results.json").read_text())

def picture(bits, label):
    colors = {-1: "#34343f", 0: "#7028ba", 1: "#62ef62"}
    cells = "".join(
        f'<rect x="{i % 16}" y="{i // 16}" width="1" height="1" '
        f'fill="{colors[int(bit)]}"/>'
        for i, bit in enumerate(bits)
    )
    return (
        f'<figure><svg viewBox="0 0 16 16">{cells}</svg>'
        f'<figcaption>{escape(label)}</figcaption></figure>'
    )

sections = []
with np.load(folder / "direct_torx_samples.npz", allow_pickle=False) as data:
    for case, title in (
        ("unconstrained", "No clues"),
        ("five_clues", "Three green and two purple clues"),
    ):
        record = metadata["results"][case]
        clues = np.full(256, -1, dtype=np.int16)
        positions = np.array(record["locked_indices"], dtype=int)
        clues[positions] = np.array(record["clue_values"], dtype=np.int16)

        panels = "".join(
            picture(sample, f"Sample {i + 1}")
            for i, sample in enumerate(data[case][:16])
        )
        sections.append(
            f"<section><h2>{escape(title)}</h2>"
            f"<div class='clues'>{picture(clues, 'Locked clues; gray = unknown')}</div>"
            f"<div class='grid'>{panels}</div></section>"
        )

page = """<!doctype html><meta charset="utf-8">
<title>Singularity Canvas — Direct Torx Samples</title>
<style>
body{background:#101014;color:#eee;font:16px system-ui;margin:32px}
h1{color:#62ef62}p{max-width:900px;line-height:1.5;color:#ccc}
section{border-top:1px solid #45454f;margin-top:32px;padding-top:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:24px}
figure{margin:0}svg{display:block;width:100%;max-width:210px}
figcaption{margin-top:8px;color:#ccc}
.clues{width:160px;margin-bottom:24px}
</style>
<h1>Direct Torx Samples</h1>
<p>Local CPU simulation · 16-symbol diagnostic model ·
8 active hidden units · temperature 1.</p>
<p>Torx draws a hidden state from its calculated distribution, then
samples the pixels. Each section shows the first 16 of 4,096 saved draws.
No best-sample selection, cleanup, or replacement with catalogue icons.</p>
<p>The five clues are a fixed diagnostic input, not clues selected from
a target symbol. Probability checks passed; image quality remains under evaluation.
This is not a demonstration of physical thermodynamic hardware advantage.</p>
"""
destination = folder / "direct_torx_preview.html"
destination.write_text(page + "".join(sections))
print("Saved:", destination)
