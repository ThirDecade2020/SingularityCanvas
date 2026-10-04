from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from torx import AbstractReferenceFactor, DFG, Site

VISIBLE = 256
HIDDEN = 128
PIXELS = jax.ShapeDtypeStruct((VISIBLE,), jnp.float32)
FEATURES = jax.ShapeDtypeStruct((HIDDEN,), jnp.float32)
MASK = jax.ShapeDtypeStruct((VISIBLE,), jnp.bool_)
TEMPERATURE = jax.ShapeDtypeStruct((), jnp.float32)


class ConditionalUpdate(AbstractReferenceFactor):
    visible: bool = eqx.field(static=True)
    input_ports: dict = eqx.field(static=True)
    output_spec: object = eqx.field(static=True)

    def __init__(self, visible):
        self.visible = visible
        self.input_ports = {
            "state": FEATURES if visible else PIXELS,
            "temperature": TEMPERATURE,
        }
        if visible:
            self.input_ports.update(locked=MASK, clues=PIXELS)
        self.output_spec = PIXELS if visible else FEATURES

    def init_params(self, key):
        return {
            "weights": 0.01 * jax.random.normal(key, (VISIBLE, HIDDEN)),
            "visible_bias": jnp.zeros(VISIBLE),
            "hidden_bias": jnp.zeros(HIDDEN),
        }

    def sample(self, key, inputs, params, info=None,
               site_info=None, return_aux=False):
        if self.visible:
            logits = inputs["state"] @ params["weights"].T
            logits = logits + params["visible_bias"]
        else:
            logits = inputs["state"] @ params["weights"]
            logits = logits + params["hidden_bias"]

        probability = jax.nn.sigmoid(logits / inputs["temperature"])
        result = jax.random.bernoulli(key, probability).astype(jnp.float32)
        if self.visible:
            result = jnp.where(inputs["locked"], inputs["clues"], result)
        return (result, None) if return_aux else result


def hidden_inputs(values):
    return {"state": values[0], "temperature": values[1]}


def visible_inputs(values):
    return {
        "state": values[0], "temperature": values[1],
        "locked": values[2], "clues": values[3],
    }


SWEEP = DFG(
    sites=(
        Site(
            "hidden", ConditionalUpdate(False),
            parents=("pixels", "temperature"),
            porting_fn=hidden_inputs, param_key="model",
            info_key=None, site_info=None,
        ),
        Site(
            "output", ConditionalUpdate(True),
            parents=("hidden", "temperature", "locked", "clues"),
            porting_fn=visible_inputs, param_key="model",
            info_key=None, site_info=None,
        ),
    ),
    input_ports={
        "pixels": PIXELS, "temperature": TEMPERATURE,
        "locked": MASK, "clues": PIXELS,
    },
    output_name="output",
)


@eqx.filter_jit
def sample_chain(params, initial, locked, clues, key,
                 temperature=1.0, steps=100):
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    initial = jnp.where(locked, clues, initial)

    def advance(state, draw_key):
        result = SWEEP.sample(
            draw_key,
            {
                "pixels": state,
                "temperature": jnp.asarray(temperature, dtype=jnp.float32),
                "locked": locked,
                "clues": clues,
            },
            {"model": params},
        )
        return result, result

    return jax.lax.scan(advance, initial, jax.random.split(key, steps))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "models/rbm_initial.npz", allow_pickle=False) as data:
        params = {name: jnp.asarray(data[name]) for name in data.files}

    positions = jnp.array([17, 54, 119, 180, 238])
    locked = jnp.zeros(VISIBLE, dtype=jnp.bool_).at[positions].set(True)
    clues = jnp.zeros(VISIBLE).at[positions].set(
        jnp.array([1., 0., 1., 0., 1.])
    )
    initial = jnp.zeros(VISIBLE, dtype=jnp.float32)
    final, history = sample_chain(
        params, initial, locked, clues, jax.random.key(44)
    )
    history.block_until_ready()

    assert bool(jnp.all(history[:, positions] == clues[positions]))
    assert bool(jnp.all((history == 0) | (history == 1)))
    assert bool(jnp.any(history[:, ~locked] != history[0, ~locked]))
    print("Recorded states:", history.shape)
    print("PASS: Torx executes both RBM conditional updates.")
    print("PASS: all five clues remain fixed, including purple clues.")
    print("Model remains untrained; recognizable output is not expected yet.")
