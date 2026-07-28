"""Tests for the matrix-free Lanczos eigenvalue solver.

The key validations:

- The Lanczos ground state of the 3D harmonic oscillator matches the analytic
  E₀ = (3/2) ω, and the first three excited states (2p-like, 3-fold degenerate)
  cluster near the analytic E₁ = (5/2) ω.
- Returned eigenstates are orthonormal under the discrete inner product.
- Converged energies agree with imaginary-time propagation on hydrogen.
"""
import jax
import jax.numpy as jnp
import pytest

from gato.ansatz import init_grid_ansatz
from gato.grid import Grid3D, inner_product, normalize
from gato.hamiltonian import Hamiltonian
from gato.potentials import harmonic_oscillator, softened_coulomb
from gato.solvers import imaginary_time, lanczos


def test_lanczos_harmonic_oscillator_ground_state():
    """3D HO ground state E = (3/2) ω for ω = 1. Expect E₀ ≈ 1.5."""
    omega = 1.0
    g = Grid3D(N=32, L=10.0)
    V = harmonic_oscillator(g, omega=omega)
    H = Hamiltonian(grid=g, V=V, order=4)

    # a generic Gaussian seed overlaps every HO state
    key = jax.random.PRNGKey(0)
    X, Y, Z = g.coords()
    psi0 = jnp.exp(-(X**2 + Y**2 + Z**2) / 2)

    res = lanczos(H, psi0, n_iters=60, n_eigenstates=4)
    E0 = float(res.eigenvalues[0])
    assert abs(E0 - 1.5 * omega) < 1e-2, f"E₀ = {E0}"


def test_lanczos_harmonic_oscillator_spectrum():
    """3D HO eigenvalue ladder: distinct eigenvalues are E_n = (n + 3/2) ω.

    Single-vector Lanczos with reorthogonalization sometimes returns one or
    two copies per degenerate manifold; we therefore look at the set of
    distinct eigenvalues the solver discovers and check the first four rungs
    of the ladder are present.
    """
    omega = 1.0
    g = Grid3D(N=40, L=10.0)
    V = harmonic_oscillator(g, omega=omega)
    H = Hamiltonian(grid=g, V=V, order=4)

    key = jax.random.PRNGKey(1)
    psi0 = jax.random.normal(key, g.shape, dtype=jnp.float64)

    res = lanczos(H, psi0, n_iters=120, n_eigenstates=12)
    E = sorted(float(e) for e in res.eigenvalues)

    # extract distinct rungs (merge values within 1e-2 of each other)
    rungs = []
    for e in E:
        if not rungs or abs(e - rungs[-1]) > 1e-2:
            rungs.append(e)

    expected = [1.5, 2.5, 3.5, 4.5]
    assert len(rungs) >= len(expected), f"only found rungs {rungs}"
    for got, want in zip(rungs, expected):
        assert abs(got - want) < 5e-3, f"rung {got}, want {want}"


def test_lanczos_eigenstates_orthonormal():
    """⟨ψᵢ|ψⱼ⟩ = δᵢⱼ for Lanczos Ritz vectors."""
    omega = 1.0
    g = Grid3D(N=24, L=10.0)
    V = harmonic_oscillator(g, omega=omega)
    H = Hamiltonian(grid=g, V=V, order=4)

    X, Y, Z = g.coords()
    psi0 = jnp.exp(-(X**2 + Y**2 + Z**2) / 2) * (1 + 0.1 * (X + Y + Z))

    res = lanczos(H, psi0, n_iters=40, n_eigenstates=3)
    # eigenstates is (N, N, N, K); form the K×K overlap matrix
    K = res.eigenstates.shape[-1]
    overlap = jnp.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ov = inner_product(
                res.eigenstates[..., i], res.eigenstates[..., j], g
            ).real
            overlap = overlap.at[i, j].set(ov)
    I = jnp.eye(K)
    err = float(jnp.max(jnp.abs(overlap - I)))
    assert err < 1e-6


@pytest.mark.slow
def test_lanczos_vs_imag_time_hydrogen_ground_state():
    """Lanczos and imaginary-time should agree on the hydrogen ground state."""
    g = Grid3D(N=40, L=10.0)
    V = softened_coulomb(g, Z=1.0)
    H = Hamiltonian(grid=g, V=V, order=4)

    key = jax.random.PRNGKey(2)
    psi0 = init_grid_ansatz(g, key, init="hydrogenic", Z=1.0)

    res_lz = lanczos(H, psi0, n_iters=60, n_eigenstates=1)
    E_lz = float(res_lz.eigenvalues[0])

    psi_it, hist = imaginary_time(H, psi0, n_steps=1500, log_every=500)
    E_it = hist.energies[-1]

    # both should be within softening-limited accuracy of each other
    assert abs(E_lz - E_it) < 1e-3, f"Lanczos {E_lz} vs imag-time {E_it}"


def test_default_krylov_dim_scales_with_grid():
    """The default Krylov dimension must grow with N, not sit at a constant.

    A fixed dimension that is adequate on a coarse grid silently fails on a
    fine one: the Fock spectral width grows like h⁻² while the physical gap
    is h-independent, so the number of Lanczos steps needed to resolve the
    occupied manifold grows like 1/h ∝ N. Regression guard for the fine-grid
    failure documented in `default_krylov_dim` (at N=96 with the old fixed
    default of 80, water's highest occupied orbital came out at -0.11 Ha
    instead of -0.52 Ha and the SCF never converged).
    """
    from gato.solvers.lanczos import default_krylov_dim

    # Strictly increasing across the range the PP drivers actually use.
    dims = [default_krylov_dim(N, 4) for N in (32, 48, 64, 96, 128)]
    assert dims == sorted(dims)
    assert dims[-1] > dims[0]

    # Calibrated pass/fail points for H2O + HGH (see default_krylov_dim).
    assert default_krylov_dim(64, 4) >= 80
    assert default_krylov_dim(96, 4) >= 120
    assert default_krylov_dim(128, 4) >= 160

    # A floor keeps coarse grids from under-provisioning: separating a
    # near-degenerate manifold costs a fixed number of steps regardless of h.
    # HCl at L=14 needs >=120 steps at N=40 and N=48, where the 1/h term
    # alone would ask for only 50-60; below that its Cl 3π pair splits
    # spuriously and the SCF stalls.
    #
    # NOTE: this floor is *not* enough for HCl on finer grids — N=80 needs
    # ~200 and this rule returns 120. That is a structural limit of
    # single-vector Lanczos on degenerate levels, not a constant to retune;
    # see `default_krylov_dim` for why, and pass lanczos_iters explicitly for
    # molecules with π/e/t degeneracies.
    assert default_krylov_dim(40, 4) >= 120
    assert default_krylov_dim(48, 4) >= 120
    assert default_krylov_dim(8, 12) > 12
