"""Force double precision for all tests. Quantum energies need x64."""
import jax

jax.config.update("jax_enable_x64", True)
