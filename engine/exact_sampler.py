"""Exact hidden-enumeration reference backend, not a hardware backend."""
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from torx import JaxPRNGSampler
from engine.rbm import ConditionalUpdate, VISIBLE, HIDDEN


class ExactRBMSampler:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        metadata = json.loads(self.model_path.with_suffix(".json").read_text())
        self.active = int(metadata["active_hidden_units"])
        if not 1 <= self.active <= 16:
            raise ValueError("This reference backend supports 1–16 active hidden units.")

        with np.load(self.model_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        if arrays["weights"].shape != (VISIBLE, HIDDEN):
            raise ValueError("Unexpected weight dimensions.")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite model parameters.")
        if np.any(arrays["weights"][:, self.active:] != 0):
            raise ValueError("Inactive hidden units must have zero weights.")
        if np.any(arrays["hidden_bias"][self.active:] != 0):
            raise ValueError("Inactive hidden units must have zero biases.")

        self.params = {name: jnp.asarray(value) for name, value in arrays.items()}
        integers = np.arange(2 ** self.active, dtype=np.uint32)
        shifts = np.arange(self.active - 1, -1, -1)
        hidden = ((integers[:, None] >> shifts) & 1).astype(np.float64)
        W = arrays["weights"][:, :self.active].astype(np.float64)
        vb = arrays["visible_bias"].astype(np.float64)
        hb = arrays["hidden_bias"][:self.active].astype(np.float64)

        # Cached temperature-1 quantities.
        self.pixel_logits = hidden @ W.T + vb
        self.base_log_weights = (
            hidden @ hb
            + np.logaddexp(0, self.pixel_logits).sum(axis=1)
        )
        # Stable sigmoid for every pixel given each hidden configuration.
        self.pixel_probabilities = np.exp(
            -np.logaddexp(0, -self.pixel_logits)
        )
        padded = jnp.zeros((len(hidden), HIDDEN)).at[:, :self.active].set(
            jnp.asarray(hidden, dtype=jnp.float32)
        )
        categorical = JaxPRNGSampler()
        visible_update = ConditionalUpdate(True)
        params = self.params

        @jax.jit
        def draw(key, log_weights, mask, clues):
            hidden_key, visible_key = jax.random.split(key)
            index = categorical.categorical(hidden_key, log_weights)
            return visible_update.sample(
                visible_key,
                {
                    "state": padded[index],
                    "temperature": jnp.array(1.0),
                    "locked": mask,
                    "clues": clues,
                },
                params,
            )
        self._draw = draw

    def sample(self, locked, values, seed, count=8):
        mask = np.asarray(locked)
        clues = np.asarray(values)
        if mask.shape != (VISIBLE,) or mask.dtype != np.bool_:
            raise ValueError("locked must be a Boolean array of 256 cells.")
        if clues.shape != (VISIBLE,) or not np.all((clues == 0) | (clues == 1)):
            raise ValueError("values must contain 256 binary values.")
        if not isinstance(count, int) or not 1 <= count <= 64:
            raise ValueError("count must be an integer from 1 to 64.")

        selected = self.pixel_logits[:, mask]
        log_weights = (
            self.base_log_weights
            + selected @ clues[mask]
            - np.logaddexp(0, selected).sum(axis=1)
        )
        logits = jnp.asarray(log_weights - log_weights.max(), dtype=jnp.float32)
        mask_jax = jnp.asarray(mask)
        clues_jax = jnp.asarray(clues, dtype=jnp.float32)
        keys = jax.random.split(jax.random.key(seed), count)

        # Bounded memory: one categorical draw at a time.
        samples = np.stack([
            np.asarray(self._draw(key, logits, mask_jax, clues_jax))
            for key in keys
        ]).astype(np.uint8)

        # Exact conditional pixel marginals, up to floating-point precision.
        posterior = np.exp(log_weights - log_weights.max())
        posterior /= posterior.sum()
        green_probabilities = posterior @ self.pixel_probabilities
        green_probabilities = np.clip(green_probabilities, 0.0, 1.0)
        green_probabilities[mask] = clues[mask]

        return {
            "green_probabilities": green_probabilities.tolist(),
            "probability_method": "exact-hidden-enumeration",
            "probability_scope": "Conditional pixel marginals under the learned model.",
            "samples": samples,
            "backend": "torx-exact-hidden-enumeration",
            "model": self.model_path.name,
            "active_hidden_units": self.active,
            "enumerated_hidden_states": 2 ** self.active,
            "temperature": 1.0,
            "seed": int(seed),
            "locked_count": int(mask.sum()),
        }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sampler = ExactRBMSampler(
        root / "models/rbm_diagnostic_exact16_refined.npz"
    )
    locked = np.zeros(VISIBLE, dtype=bool)
    values = np.zeros(VISIBLE, dtype=np.uint8)
    positions = [17, 54, 119, 180, 238]
    locked[positions] = True
    values[positions] = [1, 0, 1, 0, 1]
    result = sampler.sample(locked, values, seed=120)
    assert result["samples"].shape == (8, VISIBLE)
    assert np.all(result["samples"][:, locked] == values[locked])
    print("Samples:", result["samples"].shape)
    print("Backend:", result["backend"])
    print("Model:", result["model"])
    print("PASS: reusable sampler preserves all five clues.")
