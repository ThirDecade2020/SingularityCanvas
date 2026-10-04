import equinox as eqx
import jax
import jax.numpy as jnp
from torx import AbstractReferenceFactor, DFG, Site

SIZE = 16
GRID = jax.ShapeDtypeStruct((SIZE, SIZE), jnp.int32)
MASK = jax.ShapeDtypeStruct((SIZE, SIZE), jnp.bool_)


class GridUpdate(AbstractReferenceFactor):
    color: int = eqx.field(static=True)
    input_ports: dict = eqx.field(static=True)
    output_spec: object = eqx.field(static=True)

    def __init__(self, color):
        self.color = color
        self.input_ports = {"grid": GRID, "locked": MASK}
        self.output_spec = GRID

    def init_params(self, key):
        return {"strength": jnp.array(0.3, dtype=jnp.float32)}

    def sample(self, key, inputs, params, info=None,
               site_info=None, return_aux=False):
        grid = inputs["grid"]
        spins = 2 * grid - 1
        padded = jnp.pad(spins, 1)
        neighbors = (
            padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        probability = jax.nn.sigmoid(2 * params["strength"] * neighbors)
        draws = jax.random.bernoulli(key, probability).astype(jnp.int32)
        row, col = jnp.indices((SIZE, SIZE))
        update = ((row + col) % 2 == self.color) & ~inputs["locked"]
        result = jnp.where(update, draws, grid)
        return (result, None) if return_aux else result


def route(values):
    return {"grid": values[0], "locked": values[1]}


sweep = DFG(
    sites=tuple(
        Site(
            name,
            GridUpdate(color),
            parents=(parent, "locked"),
            porting_fn=route,
            param_key="physics",
            info_key=None,
            site_info=None,
        )
        for color, name, parent in (
            (0, "first", "grid"),
            (1, "second", "first"),
        )
    ),
    input_ports={"grid": GRID, "locked": MASK},
    output_name="second",
)
params = {"physics": {"strength": jnp.array(0.3, dtype=jnp.float32)}}
locked = jnp.zeros((SIZE, SIZE), dtype=jnp.bool_)
for row, col in ((6, 7), (7, 6), (7, 7), (7, 8), (8, 7)):
    locked = locked.at[row, col].set(True)

initial = jax.random.bernoulli(
    jax.random.key(1), 0.5, (SIZE, SIZE)
).astype(jnp.int32)
initial = jnp.where(locked, 1, initial)


@jax.jit
def run(grid, mask, key):
    def step(state, draw_key):
        updated = sweep.sample(
            draw_key, {"grid": state, "locked": mask}, params
        )
        return updated, updated
    return jax.lax.scan(step, grid, jax.random.split(key, 100))


final, history = run(initial, locked, jax.random.key(2))
history.block_until_ready()
assert bool(jnp.all(history[:, locked] == 1))
assert bool(jnp.any(history != initial))
print("Grid:", final.shape)
print("Recorded sweeps:", history.shape[0])
print("Locked cells:", int(locked.sum()))
print("PASS: the grid evolves and all five clues remain fixed.")
