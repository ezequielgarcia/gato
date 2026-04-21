"""Extrapolate the hydrogen ground-state energy to zero Coulomb softening.

The softened Coulomb potential V_ε(r) = -1/√(r² + ε²) introduces a positive
energy shift that is leading-order O(ε²) in the first-order perturbative
correction. Sweeping ε at fixed grid resolution and fitting

    E(ε) = E₀ + b·ε² + O(ε⁴)

lets us extrapolate to ε → 0 and separate the softening error from the
grid-discretization error.

Run:
    uv run python -m benchmarks.softening_extrapolation
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import gato
from gato.ansatz import init_grid_ansatz
from gato.grid import Grid3D
from gato.hamiltonian import Hamiltonian
from gato.potentials import softened_coulomb
from gato.solvers import lanczos


def run_sweep(
    N: int = 64,
    L: float = 12.0,
    epsilons: list[float] | None = None,
    order: int = 4,
    seed: int = 0,
):
    """Run Lanczos at a fixed (N, L) for several ε values; return E(ε)."""
    if epsilons is None:
        # span one decade, well above the grid spacing
        h = L / N
        epsilons = [0.5 * h, 0.75 * h, 1.0 * h, 1.5 * h, 2.0 * h]

    g = Grid3D(N=N, L=L)
    key = jax.random.PRNGKey(seed)
    psi0 = init_grid_ansatz(g, key, init="hydrogenic", Z=1.0)

    results = []
    for eps in epsilons:
        V = softened_coulomb(g, Z=1.0, epsilon=eps)
        H = Hamiltonian(grid=g, V=V, order=order)
        # Lanczos converges much faster than imag-time for the ground state
        res = lanczos(H, psi0, n_iters=60, n_eigenstates=1)
        E = float(res.eigenvalues[0])
        results.append((float(eps), E))
        print(f"  ε = {eps:.5f}   E = {E:+.7f} Ha   (Δ = {E + 0.5:+.5f})")
    return results


def extrapolate_epsilon_to_zero(
    results: list[tuple[float, float]], model: str = "linear_quadratic",
):
    """Fit and extrapolate E(ε → 0).

    Models:
      "quadratic"           : E₀ + b ε²
      "linear"              : E₀ + a ε
      "linear_quadratic"    : E₀ + a ε + b ε²   (default)

    For V_ε = -1/√(r² + ε²) the leading perturbative correction is
    O(ε) with a logarithmic coefficient (from the 1/r integrable
    singularity), not O(ε²) -- linear_quadratic is the safer default.
    """
    eps = np.array([r[0] for r in results])
    E = np.array([r[1] for r in results])
    if model == "quadratic":
        A = np.column_stack([np.ones_like(eps), eps**2])
    elif model == "linear":
        A = np.column_stack([np.ones_like(eps), eps])
    elif model == "linear_quadratic":
        A = np.column_stack([np.ones_like(eps), eps, eps**2])
    else:
        raise ValueError(f"unknown model {model!r}")
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    E0 = float(coef[0])
    return E0, coef


def main():
    parser = argparse.ArgumentParser(
        description="Extrapolate hydrogen E₀ to zero softening",
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=12.0)
    parser.add_argument("--order", type=int, choices=[2, 4], default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gato.enable_x64()
    print(f"Hydrogen softening extrapolation, N = {args.N}, L = {args.L}, order = {args.order}")
    results = run_sweep(N=args.N, L=args.L, order=args.order, seed=args.seed)

    print()
    for model in ("linear", "quadratic", "linear_quadratic"):
        E0, coef = extrapolate_epsilon_to_zero(results, model=model)
        residual = E0 + 0.5
        print(
            f"model={model:18s}  E₀ = {E0:+.7f} Ha   "
            f"residual {residual:+.5f} ({abs(residual)/0.5*100:.3f}%)"
        )
    return results


if __name__ == "__main__":
    main()
