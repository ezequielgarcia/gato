"""Tests for the Hamiltonian class."""
import jax
import jax.numpy as jnp

from gato.grid import Grid3D, normalize
from gato.hamiltonian import Hamiltonian
from gato.potentials import constant, harmonic_oscillator


def test_free_particle_zero_potential():
    """With V = 0, H = T. Check against kinetic alone."""
    g = Grid3D(N=16, L=5.0)
    V = constant(g, 0.0)
    H = Hamiltonian(grid=g, V=V)
    key = jax.random.PRNGKey(0)
    psi = jax.random.normal(key, g.shape)
    from gato.operators import kinetic
    assert float(jnp.max(jnp.abs(H.apply(psi) - kinetic(psi, g.h)))) < 1e-12


def test_rayleigh_scale_invariant():
    """E[cψ] = E[ψ] for any non-zero c."""
    g = Grid3D(N=16, L=6.0)
    V = harmonic_oscillator(g, omega=1.0)
    H = Hamiltonian(grid=g, V=V)
    X, Y, Z = g.coords()
    psi = jnp.exp(-(X**2 + Y**2 + Z**2) / 2)
    a = float(H.rayleigh(psi))
    b = float(H.rayleigh(7.3 * psi))
    assert abs(a - b) < 1e-10


def test_harmonic_oscillator_ground_state():
    """Analytic 3D HO ground state ψ ∝ exp(-ω r² / 2) has E = 3ω/2."""
    omega = 1.0
    g = Grid3D(N=48, L=10.0)
    V = harmonic_oscillator(g, omega=omega)
    H = Hamiltonian(grid=g, V=V)
    X, Y, Z = g.coords()
    psi = jnp.exp(-omega * (X**2 + Y**2 + Z**2) / 2)
    psi = normalize(psi, g)
    E = float(H.rayleigh(psi))
    E_exact = 1.5 * omega
    assert abs(E - E_exact) / E_exact < 0.01
