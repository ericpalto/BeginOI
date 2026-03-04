from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybnn
import matplotlib.pyplot as plt
from pybnn import ThreeNodeXORParams


def _build_heatmap_jax(
    *,
    grid_n: int,
    t_final: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import jax
    import diffrax
    import jax.numpy as jnp
    from pybnn.formulations import get_formulation

    params = ThreeNodeXORParams()
    formulation = get_formulation("moorman")
    y0 = jnp.zeros((7,), dtype=jnp.float32)

    x1_jax = jnp.linspace(0.0, 1.0, int(grid_n), dtype=jnp.float32)
    x2_jax = jnp.linspace(0.0, 1.0, int(grid_n), dtype=jnp.float32)
    x1_grid, x2_grid = jnp.meshgrid(x1_jax, x2_jax, indexing="xy")
    batch_inputs = jnp.stack(
        [x1_grid.reshape(-1), x2_grid.reshape(-1)],
        axis=-1,
    )

    steps = int(np.ceil(float(t_final) / float(dt)))
    t1 = float(steps * float(dt))

    def solve_one(inputs: jax.Array) -> jax.Array:
        rhs = formulation.build_jax_rhs("three_node_xor", params, inputs)

        def vector_field(
            t_value: jax.Array,
            state: jax.Array,
            args: None,
        ) -> jax.Array:
            del args
            return rhs(t_value, jnp.maximum(state, 0.0))

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=0.0,
            t1=t1,
            dt0=float(dt),
            y0=y0,
            args=None,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=steps + 1,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            throw=False,
        )
        return jnp.maximum(solution.ys[0, 6], 0.0)

    y_flat = jax.jit(jax.vmap(solve_one))(batch_inputs)

    x1 = np.asarray(x1_jax, dtype=float)
    x2 = np.asarray(x2_jax, dtype=float)
    y = np.asarray(y_flat, dtype=float).reshape(int(grid_n), int(grid_n))
    return x1, x2, y


def _build_heatmap_numpy(
    *,
    grid_n: int,
    t_final: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = ThreeNodeXORParams()
    circuit = pybnn.create_circuit(
        "moorman",
        "three_node_xor",
        backend="numpy",
        params=params,
    )

    x1 = np.linspace(0.0, 1.0, int(grid_n), dtype=float)
    x2 = np.linspace(0.0, 1.0, int(grid_n), dtype=float)
    y = np.zeros((len(x1), len(x2)), dtype=float)
    for i, u1 in enumerate(x1):
        for j, u2 in enumerate(x2):
            y[i, j] = float(
                circuit.steady_state_output(
                    np.array([u1, u2], dtype=float),
                    t_final=float(t_final),
                    dt=float(dt),
                )
            )
    return x1, x2, y


def build_heatmap(
    *,
    grid_n: int,
    t_final: float,
    dt: float,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backend == "jax":
        return _build_heatmap_jax(grid_n=grid_n, t_final=t_final, dt=dt)
    if backend == "numpy":
        return _build_heatmap_numpy(grid_n=grid_n, t_final=t_final, dt=dt)
    raise ValueError(f"Unsupported backend: {backend!r}")


def save_heatmap(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(
        y.T,
        origin="lower",
        extent=(float(x1[0]), float(x1[-1]), float(x2[0]), float(x2[-1])),
        aspect="equal",
        cmap="viridis",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Moorman three_node_xor steady-state output y")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("y (steady-state)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a steady-state heatmap for Moorman three_node_xor with original "
            "(default) ThreeNodeXORParams weights."
        )
    )
    parser.add_argument("--grid-n", type=int, default=41)
    parser.add_argument("--t-final", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/three_node_xor_heatmap.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_n < 2:
        raise ValueError("--grid-n must be >= 2.")
    if args.t_final <= 0.0:
        raise ValueError("--t-final must be positive.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")

    x1, x2, y = build_heatmap(
        grid_n=args.grid_n,
        t_final=args.t_final,
        dt=args.dt,
        backend=args.backend,
    )
    save_heatmap(x1=x1, x2=x2, y=y, out=args.out)

    print(f"Saved heatmap: {args.out}")
    print(f"y-range: [{float(y.min()):.6f}, {float(y.max()):.6f}]")


if __name__ == "__main__":
    main()
