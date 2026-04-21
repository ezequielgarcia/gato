"""End-to-end Phase 1 driver: the hydrogen atom.

Run with the imaginary-time solver by default (fastest and most reliable for
the grid ansatz). Also exposes a VQE entry point for the neural ansatz.

Expected result: E₀ ≈ -0.5 Hartree on a reasonably converged grid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from .. import enable_x64
from ..ansatz import grid_ansatz, init_grid_ansatz, neural_ansatz
from ..ansatz.neural import NeuralAnsatz
from ..grid import Grid3D, normalize
from ..hamiltonian import Hamiltonian
from ..observables import (
    kinetic_energy,
    potential_energy,
    virial_ratio,
)
from ..potentials import softened_coulomb
from ..solvers import imaginary_time, vqe


@dataclass
class HydrogenResult:
    grid: Grid3D
    psi: jax.Array
    V: jax.Array
    energy: float
    kinetic: float
    potential: float
    virial_ratio: float
    history: list[tuple[int, float]]


def solve_hydrogen(
    N: int = 48,
    L: float = 10.0,
    Z: float = 1.0,
    epsilon: float | None = None,
    solver: str = "imag_time",
    n_steps: int = 3000,
    order: int = 2,
    seed: int = 0,
    verbose: bool = True,
) -> HydrogenResult:
    """Solve the hydrogen-like ground state for a one-electron atom with
    nuclear charge Z at the origin.

    Analytic ground state energy: E₀(Z) = -Z²/2 in Hartree.

    Parameters
    ----------
    N : grid points per axis.
    L : box side length in Bohr. Scale with Z: the orbital shrinks as 1/Z.
    Z : nuclear charge (1 = H, 2 = He⁺, 3 = Li²⁺, …).
    epsilon : Coulomb softening; default h/2.
    solver : "imag_time" or "vqe_neural".
    n_steps : optimizer / propagator steps.
    order : finite-difference order of the kinetic stencil (2 or 4).
    """
    grid = Grid3D(N=N, L=L)
    V = softened_coulomb(grid, Z=Z, epsilon=epsilon)
    H = Hamiltonian(grid=grid, V=V, boundary="dirichlet", order=order)

    key = jax.random.PRNGKey(seed)

    if solver == "imag_time":
        psi0 = init_grid_ansatz(grid, key, init="hydrogenic", Z=Z)
        psi, history = imaginary_time(H, psi0, n_steps=n_steps, log_every=max(1, n_steps // 20))
        hist = list(zip(history.steps, history.energies))
    elif solver == "vqe_neural":
        # One nucleus at the origin, charge Z. The cusp factor
        # exp(-Z |r|) imposes Kato's exact boundary condition at the nucleus.
        model = NeuralAnsatz(
            key=key,
            nuclei_positions=((0.0, 0.0, 0.0),),
            nuclei_charges=(Z,),
            hidden=32,
            n_layers=3,
        )
        opt = optax.adam(1e-3)
        model_final, history = vqe(
            H, neural_ansatz, model, opt,
            n_steps=n_steps, log_every=max(1, n_steps // 20),
        )
        psi = neural_ansatz(model_final, grid)
        hist = list(zip(history.steps, history.energies))
    else:
        raise ValueError(f"Unknown solver {solver!r}")

    psi = normalize(psi, grid)

    T = float(kinetic_energy(psi, grid, order=order))
    V_exp = float(potential_energy(psi, V, grid))
    E = T + V_exp
    vr = float(virial_ratio(psi, V, grid, order=order))

    result = HydrogenResult(
        grid=grid, psi=psi, V=V,
        energy=E, kinetic=T, potential=V_exp,
        virial_ratio=vr, history=hist,
    )

    if verbose:
        E0 = -0.5 * Z * Z
        print(f"Hydrogenic atom Z={Z} on {N}^3 grid, L = {L}  (h = {grid.h:.4f}, stencil order {order})")
        print(f"Solver        : {solver}   steps = {n_steps}")
        print(f"Energy        : {E:+.6f} Ha     (target: {E0:+.6f})")
        print(f"⟨T⟩           : {T:+.6f} Ha     (analytic: {-E0:+.6f})")
        print(f"⟨V⟩           : {V_exp:+.6f} Ha     (analytic: {2*E0:+.6f})")
        print(f"Virial 2T/|V| : {vr:+.6f}        (target:  1.000000)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the hydrogen-like atom.")
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--Z", type=float, default=1.0, help="Nuclear charge")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--solver", choices=["imag_time", "vqe_neural"], default="imag_time")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--order", type=int, choices=[2, 4], default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    enable_x64()
    solve_hydrogen(
        N=args.N, L=args.L, Z=args.Z, epsilon=args.epsilon,
        solver=args.solver, n_steps=args.steps, order=args.order, seed=args.seed,
    )


if __name__ == "__main__":
    main()
