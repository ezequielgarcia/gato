"""Tests for the Kato nuclear-cusp factor in the neural ansatz.

Kato's cusp condition for an eigenstate of a Coulomb Hamiltonian:

    ⟨(1/ψ) dψ/dr⟩_Ω  →  -Z     as r → R_k,

where ⟨·⟩_Ω denotes the spherical average over the angular variables. The
NeuralAnsatz is written as ψ_θ(r) = g_θ(r) · Π_k exp(-Z_k |r - R_k|). The
exponential factor contributes exactly -Z_k to the radial log-derivative at
each nucleus. The MLP g_θ is smooth there, so its linear term
(∇g_θ · r̂) averages to zero over the angular directions r̂. The cusp is
therefore exact *in the spherically-averaged sense*, by construction.
"""
import jax
import jax.numpy as jnp

from gato.ansatz.neural import NeuralAnsatz


def _spherical_avg_log_derivative(psi_fn, R, eps=1e-3, n_dirs=64, seed=42):
    """⟨(1/ψ) dψ/dr⟩_Ω at r→R, estimated by Monte Carlo over directions."""
    key = jax.random.PRNGKey(seed)
    dirs = jax.random.normal(key, (n_dirs, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    slopes = []
    for d in dirs:
        r_a = R + eps * d
        r_b = R + 2 * eps * d
        slope = (
            jnp.log(jnp.abs(psi_fn(r_b))) - jnp.log(jnp.abs(psi_fn(r_a)))
        ) / eps
        slopes.append(slope)
    return float(jnp.mean(jnp.array(slopes)))


def test_cusp_factor_alone_has_exact_slope():
    """Test the architecture directly: just the exp(-Z|r-R|) factor.

    Any MLP-induced directional noise is absent here, so the slope should be
    -Z to very high precision from any direction.
    """
    def psi(r, Z=1.0, R=jnp.zeros(3)):
        return jnp.exp(-Z * jnp.linalg.norm(r - R))

    for Z in (1.0, 2.0, 3.0):
        slope = _spherical_avg_log_derivative(
            lambda r: psi(r, Z=Z), R=jnp.zeros(3), eps=1e-4, n_dirs=16,
        )
        assert abs(slope - (-Z)) < 1e-3


def test_neural_ansatz_cusp_single_center():
    """Full ψ = MLP · cusp: the spherical average recovers -Z at the nucleus."""
    key = jax.random.PRNGKey(0)
    for Z in (1.0, 2.0):
        model = NeuralAnsatz(
            key=key,
            nuclei_positions=((0.0, 0.0, 0.0),),
            nuclei_charges=(Z,),
            hidden=16,
            n_layers=2,
        )
        slope = _spherical_avg_log_derivative(
            model, R=jnp.zeros(3), eps=5e-4, n_dirs=128,
        )
        # Monte-Carlo of 128 directions; tolerance accounts for residual
        # MLP-gradient noise at finite eps.
        assert abs(slope - (-Z)) < 0.1, (
            f"spherical-avg log-slope {slope} vs expected {-Z} for Z={Z}"
        )


def test_neural_ansatz_cusp_two_centers():
    """For H₂⁺ with nuclei at ±R/2·x̂, the spherical-avg log-slope at each
    nucleus is -Z, contributed by that nucleus's cusp term alone. The OTHER
    nucleus's cusp contributes a smooth (non-singular) factor at this point,
    so its radial-gradient contribution averages away."""
    key = jax.random.PRNGKey(1)
    R = 2.0
    Z = 1.0
    model = NeuralAnsatz(
        key=key,
        nuclei_positions=((R / 2, 0.0, 0.0), (-R / 2, 0.0, 0.0)),
        nuclei_charges=(Z, Z),
        hidden=16,
        n_layers=2,
    )
    for nucleus in [
        jnp.array([R / 2, 0.0, 0.0]),
        jnp.array([-R / 2, 0.0, 0.0]),
    ]:
        slope = _spherical_avg_log_derivative(
            model, R=nucleus, eps=5e-4, n_dirs=128,
        )
        assert abs(slope - (-Z)) < 0.1, (
            f"spherical-avg log-slope {slope} at nucleus {nucleus}"
        )


def test_cusp_factor_is_finite_everywhere():
    """ψ = MLP · exp(-Σ Z_k |r-R_k|) is finite at the nucleus (unlike 1/r)."""
    key = jax.random.PRNGKey(2)
    model = NeuralAnsatz(
        key=key,
        nuclei_positions=((0.0, 0.0, 0.0),),
        nuclei_charges=(1.0,),
    )
    for r in [
        jnp.array([0.0, 0.0, 0.0]),     # exactly at the nucleus
        jnp.array([1e-8, 0.0, 0.0]),    # infinitesimally close
        jnp.array([10.0, 10.0, 10.0]),  # far away
    ]:
        val = float(model(r))
        assert jnp.isfinite(val), f"ψ({r}) = {val} is non-finite"
