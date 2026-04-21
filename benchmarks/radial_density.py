"""Plot the converged hydrogen radial probability density.

Solves the H atom to convergence, then histograms the ground-state density
|ψ(r)|² · 4π r² against the analytic P(r) = 4 r² e^{-2r}.

Output: docs/figures/hydrogen_radial_density.png

Run:
    uv run python -m benchmarks.radial_density
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import gato
from gato.ansatz import init_grid_ansatz
from gato.grid import Grid3D, normalize
from gato.hamiltonian import Hamiltonian
from gato.observables import radial_density
from gato.potentials import softened_coulomb
from gato.solvers import lanczos


def analytic_P(r):
    """Analytic hydrogen 1s radial probability density P(r) = 4 r² e^{-2r}."""
    return 4.0 * r * r * np.exp(-2.0 * r)


def main():
    parser = argparse.ArgumentParser(description="Plot hydrogen radial density.")
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--L", type=float, default=12.0)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument("--out", type=str, default="docs/figures/hydrogen_radial_density.png")
    args = parser.parse_args()

    gato.enable_x64()

    g = Grid3D(N=args.N, L=args.L)
    V = softened_coulomb(g, Z=1.0)
    H = Hamiltonian(grid=g, V=V, order=4)

    key = jax.random.PRNGKey(0)
    psi0 = init_grid_ansatz(g, key, init="hydrogenic", Z=1.0)

    # Lanczos gets the ground state faster than imag-time at this size
    res = lanczos(H, psi0, n_iters=80, n_eigenstates=1)
    psi = normalize(res.eigenstates[..., 0], g)
    E = float(res.eigenvalues[0])
    print(f"converged E = {E:+.6f} Ha  (analytic -0.5)")

    r_centers, P_numeric = radial_density(psi, g, n_bins=args.n_bins)
    r_centers = np.asarray(r_centers)
    P_numeric = np.asarray(P_numeric)
    r_analytic = np.linspace(0, float(g.L / 2), 400)
    P_analytic = analytic_P(r_analytic)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(r_analytic, P_analytic, "-", lw=1.5, label="analytic $4r^2 e^{-2r}$", color="black")
    ax.plot(r_centers, P_numeric, "o", ms=4, label=f"numerical ($N={args.N}^3$, $L={args.L}\\,a_0$)", color="tab:blue")
    ax.set_xlabel("$r\\ (a_0)$")
    ax.set_ylabel("radial probability $P(r)$")
    ax.set_title(f"Hydrogen 1s radial density   (E = {E:+.4f} Ha)")
    ax.set_xlim(0, 8)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
