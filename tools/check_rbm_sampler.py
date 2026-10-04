from itertools import product
import numpy as np
import jax
import jax.numpy as jnp
from engine.rbm import sample_chain, VISIBLE, HIDDEN

# Only four visible and three hidden units interact.
W = np.array([
    [0.8, -0.4, 0.3],
    [-0.5, 0.7, 0.2],
    [0.4, 0.2, -0.6],
    [-0.3, 0.5, 0.7],
], dtype=np.float32)
vb = np.array([-0.3, 0.2, -0.1, 0.1], dtype=np.float32)
hb = np.array([0.1, -0.2, 0.3], dtype=np.float32)

params = {
    "weights": jnp.zeros((VISIBLE, HIDDEN)).at[:4, :3].set(W),
    "visible_bias": jnp.zeros(VISIBLE).at[:4].set(vb),
    "hidden_bias": jnp.zeros(HIDDEN).at[:3].set(hb),
}
visible = np.array(list(product((0, 1), repeat=4)), dtype=np.float64)
hidden = np.array(list(product((0, 1), repeat=3)), dtype=np.float64)

cases = (
    ("Unconstrained", 1.0, {}),
    ("Green and purple locked", 1.0, {0: 1, 2: 0}),
    ("Lower temperature", 0.7, {}),
)

for case, (label, temperature, fixed) in enumerate(cases):
    # Enumerate joint probabilities, then sum over hidden states.
    scores = (
        (visible @ vb)[:, None]
        + (hidden @ hb)[None, :]
        + visible @ W @ hidden.T
    ) / temperature
    joint = np.exp(scores - scores.max())
    expected = joint.sum(axis=1)

    mask = np.zeros(VISIBLE, dtype=bool)
    clues = np.zeros(VISIBLE, dtype=np.float32)
    for index, value in fixed.items():
        mask[index] = True
        clues[index] = value
        expected[visible[:, index] != value] = 0
    expected /= expected.sum()

    _, history = sample_chain(
        params,
        jnp.zeros(VISIBLE),
        jnp.asarray(mask),
        jnp.asarray(clues),
        jax.random.key(730 + case),
        temperature=temperature,
        steps=21_000,
    )
    draws = np.asarray(history)[1_000:]
    assert np.all((draws == 0) | (draws == 1))
    assert np.all(draws[:, mask] == clues[mask])

    codes = draws[:, :4].astype(np.int32) @ np.array([8, 4, 2, 1])
    measured = np.bincount(codes, minlength=16) / len(codes)
    largest_error = float(np.max(np.abs(measured - expected)))
    total_variation = float(np.abs(measured - expected).sum() / 2)

    print(f"\n{label} — temperature {temperature}")
    print(f"Largest state-probability error: {largest_error:.2%}")
    print(f"Total variation distance: {total_variation:.4f}")
    assert total_variation < 0.03, (
        "Outside diagnostic tolerance; inspect before changing training."
    )

print("\nPASS: sampled distributions match the tiny-model reference.")
print("PASS: both green and purple locks are respected.")
print("This does not establish convergence of the trained symbol model.")
