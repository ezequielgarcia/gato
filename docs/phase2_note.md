# GATO Phase 2: H₂⁺ with differentiable geometry

> **Read this on <https://ezequielgarcia.github.io/gato/phase2_note/>** — GitHub's markdown viewer does not render display math reliably.

**Companion note to `phase1_paper.md`.** Phase 2 re-uses every numerical primitive of Phase 1 (cell-centered grid, finite-difference kinetic operator, softened Coulomb, imaginary-time propagation, Rayleigh-quotient energy) without modification. Readers are assumed familiar with those; this note describes only what is new.

## Abstract

Phase 2 extends the GATO solver from single-centre atoms to multi-nucleus single-electron systems, with nuclear coordinates as first-class differentiable parameters. We implement the multi-center softened Coulomb potential, expose nuclear positions as a JAX pytree, and obtain Born–Oppenheimer forces by a single `jax.grad` call on the energy functional evaluated at the converged electronic wavefunction. The Hellmann–Feynman theorem makes back-propagation through the imaginary-time solver unnecessary. Geometry optimization is a standard optax loop alternating with electronic solves. On a $56^3$ grid with 4th-order stencils and $\varepsilon = h/2$ softening, 30 Adam steps starting from $R = 2.4\,a_0$ recover the H$_2^+$ bond length $R_e = 1.9986\,a_0$ (Burrau 1927 analytic value $1.9972\,a_0$, deviation $0.07\%$); the total energy $-0.5749\,E_h$ remains softening-limited, ~28 mHa above the analytic $-0.6026\,E_h$, the same $O(\varepsilon)$ bias documented in Phase 1 §5.5.

## 1. Scope

Phase 2 is framed as a capability milestone rather than a major work phase. It introduces exactly two new structural elements — a multi-center potential with differentiable nuclear positions, and a gradient-descent geometry-optimization loop — and re-uses the Phase 1 imaginary-time solver, finite-difference stencils, and softening prescription unchanged. The only new physics concepts beyond Phase 1 are the Born–Oppenheimer approximation and the Hellmann–Feynman theorem. The purpose is to validate the multi-nucleus and nuclear-force machinery on the simplest possible system (one electron, two identical nuclei, with an exact reference in prolate spheroidal coordinates) before they become entangled with many-electron self-consistency in Phase 3.

## 2. Multi-center potential

With $K$ nuclei at positions $\lbrace \mathbf R_k\rbrace $ carrying charges $\lbrace Z_k\rbrace $, the softened nuclear potential is

$$
V(\mathbf r; \lbrace \mathbf R_k\rbrace ) \;=\; -\sum_{k=1}^{K} \frac{Z_k}{\sqrt{|\mathbf r - \mathbf R_k|^2 + \varepsilon^2}}. \tag{1}
$$

The implementation in `src/gato/potentials.py::multi_center_softened_coulomb` is a `jax.vmap` over the nuclei, built on the single-center form already in Phase 1. Setting $K=1$ reduces to the Phase 1 potential by construction (`tests/test_multicenter_potential.py::test_reduces_to_single_center_when_K_equals_1`). Linearity in $Z$ is verified (`test_charge_scaling`), as is superposition of single-center contributions (`test_two_centers_superposition`). Translation of the nuclei by a vector $\mathbf t$ translates $V$ by the same vector up to a grid-resolution artifact: exact equality holds on a cell-centered grid only for shifts commensurate with the spacing $h$.

## 3. Nuclei as a JAX pytree

Nuclear coordinates flow through `jax.grad` cleanly only if their container is a registered pytree. We use a `typing.NamedTuple`,

```python
class Nuclei(NamedTuple):
    positions: jax.Array   # (K, 3) in Bohr
    charges:   jax.Array   # (K,)
```

which JAX treats automatically as a pytree with leaves (positions, charges). Forces on nuclei are then `-jax.grad(E)(positions)`, returning an array of the same shape $(K, 3)$. Keeping charges as a separate leaf — rather than as static metadata — is deliberate: it means `jax.grad` can also differentiate with respect to $Z_k$, which is the prerequisite for the differentiable-pseudopotential experiment proposed in `IDEAS.md` for Phase 3.5. It is not exercised in Phase 2 itself.

The geometry module additionally exposes `bond_length`, `bond_angle`, the classical nuclear-repulsion energy

$$
E_{\rm nn}(\lbrace \mathbf R_k\rbrace ) \;=\; \sum_{i<j} \frac{Z_i Z_j}{|\mathbf R_i - \mathbf R_j|}, \tag{2}
$$

and a charge-weighted centroid / recentre utility. A consistency test (`test_forces_from_nuclear_repulsion_match_analytic`) verifies that `jax.grad` on $E_{\rm nn}$ for two unit charges on the $z$-axis at $\pm R/2$ returns exactly the analytic Coulomb force $\pm 1/R^2 \hat{\mathbf z}$.

## 4. Born–Oppenheimer energy and Hellmann–Feynman forces

The electronic energy at fixed nuclear geometry is obtained variationally: the Phase 1 imaginary-time solver is applied to $\hat H(\lbrace \mathbf R_k\rbrace ) = \hat T + V(\,\cdot\,;\lbrace \mathbf R_k\rbrace )$ with the LCAO initial guess

$$
\psi_0(\mathbf r) \;=\; \sum_k Z_k^{3/2}\,\exp\!\bigl(-Z_k\,|\mathbf r - \mathbf R_k|\bigr), \tag{3}
$$

the bonding $\sigma_g$ combination for a homonuclear diatomic. After convergence, the total Born–Oppenheimer energy is

$$
E(\lbrace \mathbf R_k\rbrace ) \;=\; \frac{\langle \psi_{\rm el} \,|\, \hat T + V \,|\, \psi_{\rm el}\rangle}{\langle \psi_{\rm el} \,|\, \psi_{\rm el}\rangle} \;+\; E_{\rm nn}(\lbrace \mathbf R_k\rbrace ). \tag{4}
$$

Direct differentiation of $E$ with respect to $\mathbf R_k$, propagating through the converged $\psi_{\rm el}$, is the content of the Hellmann–Feynman theorem [Hellmann 1937, Feynman 1939]. Because $\psi_{\rm el}(\lbrace \mathbf R_k\rbrace )$ is stationary with respect to electronic variations $\delta\psi$ at fixed nuclei — it is the variational minimum — the total derivative collapses to the partial:

$$
\frac{d E}{d\mathbf R_k} \;=\; \left\langle \psi_{\rm el} \,\Big|\, \frac{\partial V}{\partial \mathbf R_k} \,\Big|\, \psi_{\rm el}\right\rangle \;+\; \frac{\partial E_{\rm nn}}{\partial \mathbf R_k}. \tag{5}
$$

Pulay corrections [Pulay 1969], which arise whenever the basis functions depend on $\mathbf R_k$, vanish identically here because our representation is a fixed Cartesian grid, not atom-centred basis functions. This is a genuine simplification unique to the real-space-grid approach: common in production DFT codes with Gaussian bases, Pulay terms are a nontrivial bookkeeping burden.

Operationally, eq. (5) is obtained from

```python
force = -jax.grad(bo_energy, argnums=0)(positions, charges, psi_el, grid, eps)
```

where `bo_energy` is the RHS of eq. (4) written as a pure function of `positions`. No back-propagation through the imaginary-time solver is performed, and no checkpointing of intermediate $\psi$ arrays is required. Variational stationarity guarantees the result is exact when $\psi_{\rm el}$ has converged; away from convergence, the computed force carries a residual $O(|\delta E/\delta \psi|)$ bias that vanishes as imaginary time continues.

The construction is validated against a symmetric finite difference of $E$ at fixed $\psi_{\rm el}$ in `tests/test_h2_plus.py::test_hellmann_feynman_matches_finite_difference`: the autodiff force agrees with the finite-difference reference to $\sim 10^{-4}\,E_h/a_0$ at $R = 2.2\,a_0$ on a $32^3$ grid.

## 5. Geometry-optimization loop

Relaxation proceeds by alternating an electronic solve with a nuclear step:

1. Given $\mathbf R^{(n)}$, converge $\psi_{\rm el}^{(n)}$ by $n_{\rm el}$ imaginary-time steps initialised from eq. (3).
2. Compute $\mathbf g^{(n)} = \nabla_{\mathbf R}\,E$ via `jax.grad` on eq. (4), evaluated at the converged $\psi_{\rm el}^{(n)}$.
3. Advance $\mathbf R^{(n+1)}$ by one Adam step [Kingma & Ba 2015] on $\mathbf g^{(n)}$.
4. Re-centre: subtract the charge-weighted centroid to pin the translational zero-mode to the origin.

The outer loop is driven from Python; only the energy-and-gradient evaluation is JIT-compiled. Step (4) is redundant in exact arithmetic (Newton's third law forces the net translational gradient to zero for a closed system) but suppresses cumulative numerical drift over many tens of steps.

Adam is used in preference to plain gradient descent despite the smooth landscape: the imaginary-time solve at each geometry step only approximately converges $\psi_{\rm el}$, and the resulting small-amplitude noise in $\mathbf g^{(n)}$ is absorbed by Adam's first-moment smoothing. A learning rate of $0.04$–$0.05$ with 20–30 geometry steps is empirically sufficient for H$_2^+$.

## 6. Results: H$_2^+$

### 6.1 Born–Oppenheimer curve

`bo_curve(R_values, grid)` computes $E(R)$ at arbitrary separations for a homonuclear diatomic aligned with the $z$-axis, nuclei at $\pm R/2$. On a $40^3$ grid ($L = 10\,a_0$, $h = 0.25$, 4th-order stencil, $\varepsilon = h/2$, 1500 imag-time steps per point) the curve sampled at $R \in \lbrace 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0\rbrace \,a_0$ exhibits a bound minimum in $[1.5, 2.5]\,a_0$ with stabilisation below the dissociation asymptote $E(R \to \infty) \to -0.5\,E_h$ (a single hydrogen atom with an infinitely distant proton); see `test_bo_curve_has_bound_minimum`.

### 6.2 Bond length via gradient descent

**Table 1.** H$_2^+$ equilibrium bond length recovered by Adam on $E(\lbrace \mathbf R_k\rbrace )$. All runs: 4th-order stencil, $\varepsilon = h/2$, LCAO initial guess, `gato-h2plus` CLI. Reference: Burrau 1927, $R_e = 1.9972\,a_0$, $E(R_e) = -0.6026\,E_h$.

| $N$ | $L$ ($a_0$) | $h$ | $R_0$ | $n_{\rm el}$ | geom steps | $R_e$ (measured) | $E$ (measured) | $|R_e - R_e^{\rm ref}|$ | $E - E^{\rm ref}$ |
|-----|-------------|------|-------|---------------|------------|-------------------|----------------|--------------------------|----------------------|
| 40  | 10          | 0.250 | 2.6   | 1500          | 25         | $2.267$           | $-0.5606$      | $0.270$                  | $+42\,$mHa           |
| 56  | 10          | 0.179 | 2.4   | 2000          | 30         | $\mathbf{1.9986}$ | $-0.5749$      | $0.0014$                 | $+28\,$mHa           |

The bond length converges to within $0.0014\,a_0$ of the analytic value at $N = 56$. The total energy remains softening-limited; reducing $\varepsilon$ or linearly extrapolating $\varepsilon \to 0$ along the Phase 1 §5.5 recipe is expected to close the ~28 mHa residual. We have not performed the extrapolation in this note because the geometry — not the total energy — is the Phase 2 acceptance target.

### 6.3 Cost

The $N = 56$ run completes in approximately four minutes of wall-clock time on a single CPU core in float64. The stack is pure JAX and is expected to run in single-digit seconds on an RTX 5070 (see `README.md` §9); this has not been measured in this note.

## 7. Acceptance criterion

The Phase 2 exit criterion from `README.md` §5 — recover the H$_2^+$ equilibrium bond length ($\approx 2.00\,a_0$) by pure gradient descent on the energy — is met by the $N = 56$ result ($R_e = 1.9986\,a_0$, deviation $0.07\%$).

## 8. Reproducibility

```bash
uv run pytest tests/test_geometry.py tests/test_multicenter_potential.py -v   # ~10 s
uv run pytest tests/test_h2_plus.py -v                                        # ~60 s (slow)
uv run gato-h2plus --N 40 --L 10 --R0 2.6 --electronic-steps 1500 --geom-steps 25 --lr 0.05
uv run gato-h2plus --N 56 --L 10 --R0 2.4 --electronic-steps 2000 --geom-steps 30 --lr 0.04
```

All results in this note were produced in float64 on a single CPU core.

## References

- Burrau, Ø. (1927), "Berechnung des Energiewertes des Wasserstoffmolekel-Ions (H$_2^+$) im Normalzustand", *Det Kongelige Danske Videnskabernes Selskab: Matematisk-Fysiske Meddelelser* **7**(14).
- Hellmann, H. (1937), *Einführung in die Quantenchemie*, Franz Deuticke, Leipzig.
- Feynman, R. P. (1939), "Forces in molecules", *Physical Review* **56**, 340.
- Pulay, P. (1969), "Ab initio calculation of force constants and equilibrium geometries in polyatomic molecules", *Molecular Physics* **17**, 197.
- Kingma, D. P. & Ba, J. (2015), "Adam: a method for stochastic optimization", *ICLR*.
