"""End-to-end Phase 1 test: the hydrogen 1s ground state.

Keeps the grid small and the solver short so the test runs in under a minute
on CPU. A tight benchmark is done outside the test suite via
`python -m gato.physics.hydrogen --N 96 --L 12 --steps 4000`.
"""
import pytest

from gato.physics.hydrogen import solve_hydrogen


@pytest.mark.slow
def test_hydrogen_imag_time_converges():
    """On a coarse grid, imaginary time should recover E₀ within 10% of -0.5."""
    res = solve_hydrogen(N=40, L=10.0, solver="imag_time", n_steps=1500, verbose=False)
    # Grid coarse + softening ε=h/2 ≈ 0.125 => ~5-10% above -0.5
    assert res.energy < -0.44, f"E = {res.energy} should be below -0.44"
    assert res.energy > -0.55, f"E = {res.energy} should not undershoot the exact value"
    # virial ratio should be close to 1 on this state
    assert abs(res.virial_ratio - 1.0) < 0.08


@pytest.mark.slow
def test_hydrogen_vqe_neural_improves_from_random_init():
    """Neural VQE should drop below a random-init baseline.

    We don't require it to match imag_time on such a small budget; we just
    require that optimization is doing something useful.
    """
    res = solve_hydrogen(N=32, L=8.0, solver="vqe_neural", n_steps=400, verbose=False)
    # Initial neural output is essentially noise; energies start positive.
    # After 400 Adam steps on a 32^3 grid, we should at least be in bound-state
    # territory.
    assert res.energy < 0.0, f"E = {res.energy} expected to go negative"
    # Loose upper bound: if this passes we know the gradient flow works.
    assert res.energy < -0.2, f"E = {res.energy} expected to be in O(-0.3) range"
