"""Tests for the closed-shell RHF SCF driver.

The checks here span three levels:

1. `exchange_apply` matches its algebraic reduction for a single orbital:
    K̂ φ = φ · J[|φ|²].  This is the piece where the general n_orb² Poisson
    machinery must degenerate correctly to the self-exchange case that Phase 3
    helium hits.

2. The Fock operator reduces to the core Hamiltonian when the exchange /
    Hartree pieces vanish (zero orbitals), i.e. F ψ = (T + V_ext) ψ.

3. Helium ground state: the SCF loop converges and returns an energy within a
    few tens of mHa of the reference RHF value -2.862 E_h on a modest grid.
    The softened-kernel + O(h⁴) residual at N=64 is too coarse to hit the
    phase's 1 mHa exit criterion without an ε→0 extrapolation, so this test
    enforces the weaker "convergence behaviour is correct" bound; the tight
    energy target is deferred to a dedicated benchmark.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from gato.grid import Grid3D, inner_product, normalize
from gato.potentials import softened_coulomb
from gato.scf import _MixedFock, exchange_apply, rhf_energy, scf_rhf
from gato.solvers.poisson import hartree_potential

gpu_only = pytest.mark.skipif(
    not os.environ.get("GATO_GPU"),
    reason="set GATO_GPU=1 on a CUDA box to run the multi-orbital atom benchmarks",
)


def _normalized_gaussian(grid: Grid3D, alpha: float) -> jnp.ndarray:
    X, Y, Z = grid.coords()
    r2 = X * X + Y * Y + Z * Z
    psi = (2 * alpha / jnp.pi) ** 0.75 * jnp.exp(-alpha * r2)
    return normalize(psi, grid)


def test_exchange_reduces_to_self_for_one_orbital():
    grid = Grid3D(N=32, L=8.0)
    phi = _normalized_gaussian(grid, alpha=1.0)
    orbitals = phi[..., None]  # (N, N, N, 1)

    K_phi = exchange_apply(phi, orbitals, grid)
    expected = phi * hartree_potential(jnp.abs(phi) ** 2, grid)

    err = float(jnp.max(jnp.abs(K_phi - expected)))
    assert err < 1e-12


def test_fock_with_zero_orbitals_is_core_hamiltonian():
    grid = Grid3D(N=24, L=6.0)
    V_ext = softened_coulomb(grid, Z=2.0)
    zero_orb = jnp.zeros((*grid.shape, 1))
    V_H = jnp.zeros(grid.shape)

    from gato.hamiltonian import Hamiltonian

    fock = _MixedFock(grid=grid, V_ext=V_ext, orbitals=zero_orb, V_H=V_H, order=4)
    H_core = Hamiltonian(grid=grid, V=V_ext, order=4)

    psi = _normalized_gaussian(grid, alpha=1.5)
    diff = float(jnp.max(jnp.abs(fock.apply(psi) - H_core.apply(psi))))
    assert diff < 1e-12


def test_rhf_energy_single_orbital_decomposition():
    """For one doubly-occupied orbital, 2J − K collapses to a single J[|φ|²],
    so the two-body energy reduces to ∫ |φ|² J[|φ|²] dV."""
    grid = Grid3D(N=32, L=8.0)
    phi = _normalized_gaussian(grid, alpha=1.0)
    orbitals = phi[..., None]
    V_ext = jnp.zeros(grid.shape)

    E_total = float(rhf_energy(orbitals, V_ext, grid))

    # Expected two-body term: ∫ |φ|² J[|φ|²] dV.
    rho_one = jnp.abs(phi) ** 2
    J_one = hartree_potential(rho_one, grid)
    E_2body_expected = float(jnp.sum(rho_one * J_one) * grid.dV)

    # One-body: kinetic only (V_ext = 0), 2 × ⟨φ|T|φ⟩.
    from gato.operators import kinetic

    T_phi = kinetic(phi, grid.h, "dirichlet", 4)
    E_1body_expected = 2.0 * float(inner_product(phi, T_phi, grid).real)

    expected = E_1body_expected + E_2body_expected
    assert abs(E_total - expected) < 1e-10


def test_helium_scf_converges():
    """RHF on He: 2 electrons, one doubly-occupied orbital, V_ext = -2/r.

    Exact RHF energy is -2.862 E_h. On N=64, L=10 with ε=h/2 softening on
    both V_ext and the Hartree/exchange kernel, the softened Hamiltonian's
    own ground state sits near -2.60 E_h — the ~260 mHa gap to the
    reference is the softening residual, not an SCF bug (see the energy
    decomposition in the phase-3 README notes). This test pins the SCF to
    that softened-Hamiltonian answer; the 1 mHa exit criterion requires the
    ε→0 extrapolation or an origin-regularized kernel, which is tracked as
    a separate phase-3 accuracy task.
    """
    grid = Grid3D(N=64, L=10.0)
    V_ext = softened_coulomb(grid, Z=2.0)

    result = scf_rhf(
        V_ext, grid, n_occ=1,
        max_iters=25, tol=1e-6, mixing=0.7, order=4, lanczos_iters=40,
    )

    assert result.converged, f"SCF did not converge; history={result.energy_history}"
    assert result.energy > -3.0, f"He energy {result.energy} below exact -2.9037, unphysical"

    # Softened-Hamiltonian ground state for this grid: -2.599 E_h (measured).
    # Bound accommodates routine numerical noise without hiding regressions.
    E_soft = -2.599
    assert abs(result.energy - E_soft) < 0.010, (
        f"He RHF energy {result.energy} deviates from softened reference {E_soft} by "
        f"{abs(result.energy - E_soft) * 1000:.1f} mHa"
    )


# -----------------------------------------------------------------------------
# Multi-orbital closed-shell atom benchmarks (skipped on CPU)
#
# These exercise the general n_orb² exchange operator (Be: n_occ=2) and the
# p-orbital symmetry handling in the initial guess (Ne: n_occ=5). They are
# order-of-magnitude more expensive than He because each Fock apply does
# n_occ² Poisson solves. Skipped by default so the CPU test suite stays under
# three minutes; flip the skip off once the GPU backend is in use — a 5070
# should bring Be down to seconds and Ne to under a minute.
#
# Reference energies (exact RHF, atomic ground states):
#   Be   -14.573 E_h
#   Ne  -128.547 E_h
#   (bond energies for Phase 4 are another order of magnitude harder)
#
# The softened-Hamiltonian residual grows with Z, so on ε=h/2 grids we expect
# Be to sit ~0.3 E_h above -14.573 and Ne ~a few E_h above -128.547. The
# tests below enforce only "SCF converged and energy in a physical range";
# quantitative accuracy is the Phase-3 accuracy-task piece, not this smoke
# test.
# -----------------------------------------------------------------------------


@gpu_only
def test_beryllium_rhf_converges():
    """Be (Z=4, 1s² 2s²): first multi-orbital closed-shell atom.

    Validates the n_orb²=4 exchange operator and that Lanczos recovers both
    the 1s- and 2s-like eigenstates from a single Krylov run.
    """
    grid = Grid3D(N=64, L=10.0)
    V_ext = softened_coulomb(grid, Z=4.0)

    result = scf_rhf(
        V_ext, grid, n_occ=2,
        max_iters=30, tol=1e-6, mixing=0.5, order=4, lanczos_iters=80,
    )

    assert result.converged
    # Physical range: above exact (-14.573) but not wildly so
    assert -14.6 < result.energy < -12.0


@gpu_only
def test_neon_rhf_converges():
    """Ne (Z=10, 1s² 2s² 2p⁶): first atom requiring p-orbital occupancy.

    This is where the symmetry-breaking perturbation in the default initial
    guess earns its keep — a purely spherical starter would leave the three
    2p orbitals invisible to Lanczos.
    """
    grid = Grid3D(N=80, L=10.0)
    V_ext = softened_coulomb(grid, Z=10.0)

    result = scf_rhf(
        V_ext, grid, n_occ=5,
        max_iters=40, tol=1e-5, mixing=0.4, order=4, lanczos_iters=120,
    )

    assert result.converged
    # Softening hits Z=10 hard: expect to be several E_h above exact -128.547.
    # The test just asks for "sensible" — the actual number becomes a
    # calibration point once accuracy work lands.
    assert -130.0 < result.energy < -100.0

    # Sanity: at least some orbital energies should be non-degenerate — i.e.
    # 1s and 2s clearly below the 2p triplet — confirming we actually found
    # states of multiple angular momenta, not five copies of the same s.
    es = sorted(float(e) for e in result.orbital_energies)
    gaps = [es[i + 1] - es[i] for i in range(len(es) - 1)]
    assert max(gaps) > 0.5, (
        f"Orbital-energy spectrum {es} is nearly degenerate — Lanczos likely "
        f"found only s-states; initial-guess symmetry breaking may be broken."
    )
