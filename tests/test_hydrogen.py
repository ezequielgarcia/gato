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
    res = solve_hydrogen(N=40, L=10.0, n_steps=1500, verbose=False)
    # Grid coarse + softening ε=h/2 ≈ 0.125 => ~5-10% above -0.5
    assert res.energy < -0.44, f"E = {res.energy} should be below -0.44"
    assert res.energy > -0.55, f"E = {res.energy} should not undershoot the exact value"
    # virial ratio should be close to 1 on this state
    assert abs(res.virial_ratio - 1.0) < 0.08


@pytest.mark.slow
def test_hydrogenic_ions_scale_as_minus_Z_squared_over_two():
    """For a one-electron atom with nuclear charge Z, E₀ = -Z²/2. Test the
    scaling across Z ∈ {1, 2, 3}. Each run uses a box shrunk proportionally
    to the orbital size (~1/Z) to keep the relative grid resolution comparable.
    """
    configs = [
        (1.0, 10.0),  # H
        (2.0,  6.0),  # He+
        (3.0,  4.0),  # Li2+
    ]
    for Z, L in configs:
        res = solve_hydrogen(
            N=48, L=L, Z=Z,
            n_steps=1500, order=4, verbose=False,
        )
        expected = -0.5 * Z * Z
        # softening-limited; accept within 8% for a quick test
        rel = abs((res.energy - expected) / expected)
        assert rel < 0.08, f"Z={Z}: E = {res.energy:.4f}, expected {expected:.4f}"
        # virial should be close to unity
        assert abs(res.virial_ratio - 1.0) < 0.08


