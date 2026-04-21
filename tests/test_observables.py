"""Tests for the observables module."""
import jax.numpy as jnp

from gato.grid import Grid3D, normalize
from gato.observables import (
    kinetic_energy,
    potential_energy,
    radial_density,
    virial_ratio,
)
from gato.potentials import harmonic_oscillator


def test_gaussian_kinetic_energy():
    """⟨T⟩ = 3/4 for unit-variance 3D Gaussian (analytic)."""
    g = Grid3D(N=48, L=10.0)
    X, Y, Z = g.coords()
    psi = jnp.exp(-(X**2 + Y**2 + Z**2) / 2)
    T = float(kinetic_energy(psi, g))
    assert abs(T - 0.75) < 1e-2


def test_harmonic_oscillator_virial():
    """For HO ground state, ⟨T⟩ = ⟨V⟩ = E/2, so 2⟨T⟩/|⟨V⟩| ≈ 1."""
    omega = 1.0
    g = Grid3D(N=48, L=10.0)
    V = harmonic_oscillator(g, omega)
    X, Y, Z = g.coords()
    psi = normalize(jnp.exp(-omega * (X**2 + Y**2 + Z**2) / 2), g)
    T = float(kinetic_energy(psi, g))
    V_exp = float(potential_energy(psi, V, g))
    # HO virial: ⟨T⟩ = ⟨V⟩ exactly
    assert abs(T - V_exp) / abs(V_exp) < 0.02
    # Our Coulomb-style virial ratio is 2T/|V|; for HO this is 2.
    ratio = float(virial_ratio(psi, V, g))
    assert abs(ratio - 2.0) < 0.05


def test_radial_density_normalization():
    """If ψ is normalized, ∫ P(r) dr ≈ 1."""
    g = Grid3D(N=48, L=10.0)
    X, Y, Z = g.coords()
    psi = normalize(jnp.exp(-jnp.sqrt(X**2 + Y**2 + Z**2)), g)
    r, P = radial_density(psi, g, n_bins=80)
    dr = r[1] - r[0]
    integral = float(jnp.sum(P) * dr)
    # large box + discretization: not exactly 1, but close
    assert abs(integral - 1.0) < 0.05
