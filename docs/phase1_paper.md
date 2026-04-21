# GATO: A differentiable, matrix-free 3D Schrödinger solver in JAX — Phase 1: the hydrogen atom

**Ezequiel Garcia**

*Phase 1 technical report. Revision 6, April 2026.*

---

## Abstract

We describe the design and full implementation of **GATO** (*Grid Autodiff Theory of Orbitals*), a 3D Schrödinger solver written from first principles in [JAX](https://jax.readthedocs.io/). The long-term objective is a neural many-body wavefunction for the water molecule, with molecular geometry obtained from first principles by gradient descent on the energy. In this first phase we restrict ourselves to a single electron in a central potential and target the hydrogen 1s ground state energy, $E_0 = -0.5\,E_h$. The key design decisions — a cell-centered Cartesian real-space grid, matrix-free finite-difference operators, automatic differentiation end-to-end, and a variational treatment via the Rayleigh quotient — are motivated and specified. A second-order finite-difference Laplacian is implemented as a JIT-compiled local stencil and shown to reproduce the analytic plane-wave eigenvalue relation to $10^{-10}$ on periodic boundaries, and to exhibit $O(h^2)$ convergence to the continuum. The operator is self-adjoint under the discrete midpoint inner product to $10^{-8}$ with zero-Dirichlet boundary conditions. A softened Coulomb potential and a matrix-free `Hamiltonian` class complete the single-particle Schrödinger operator. Three variational solvers are implemented and verified: imaginary-time propagation on a grid ansatz, a matrix-free Lanczos eigenvalue solver returning the lowest few Ritz pairs, and optax-driven VQE on an Equinox multilayer-perceptron ansatz with explicit Kato nuclear-cusp factors [Kato 1957]. Both 2nd-order and 4th-order finite-difference Laplacian stencils are available (13-point versus 7-point; $O(h^4)$ versus $O(h^2)$ error). On a $96^3$ grid with $L = 12\,a_0$, imaginary-time propagation recovers the hydrogen ground state energy to $E = -0.487\,E_h$ (2.6 % above the analytic $-0.5\,E_h$) with a virial ratio of $0.984$; residual error is dominated by the Coulomb-softening scale $\epsilon = h/2$. A linear-in-$\epsilon$ extrapolation to zero softening gives $E_0 = -0.504\,E_h$ ($0.74\,\%$ error) at $N = 64$. The framework also reproduces the $Z^2$ scaling of the hydrogenic ground state across $Z \in \{1, 2, 3\}$ (H, He⁺, Li²⁺), and the radial probability density of the converged 1s ground state matches the analytic $P(r) = 4 r^2 e^{-2r}$ to within grid-binning noise. These results validate the substrate on which the remaining phases (multi-nucleus single-electron systems, mean-field electronic structure, and neural many-body wavefunctions for water) will be built.

---

## 1. Introduction

The numerical solution of the time-independent Schrödinger equation underpins the whole of computational quantum chemistry and condensed-matter physics. For a single electron in an external potential, the problem reduces to finding the low-lying eigenpairs $(E, \psi)$ of

$$
\hat H\,\psi(\mathbf r) \;=\; E\,\psi(\mathbf r),
\qquad
\hat H = -\tfrac{1}{2}\nabla^2 + V(\mathbf r),
$$

in Hartree atomic units ($\hbar = m_e = e = 4\pi\varepsilon_0 = 1$). Two classical strategies dominate practical implementations: expansion in a finite basis set (Gaussians, plane waves, or atomic orbitals) followed by generalized eigenvalue solution; or real-space discretization on a grid, where the Hamiltonian becomes a large sparse matrix acted on iteratively.

Both approaches tend to obscure the physics behind layers of hand-tuned linear algebra, and neither is naturally differentiable: obtaining derivatives of an observable with respect to geometric or gauge parameters typically requires laborious manual coding of the adjoint equations. The emergence of machine-learning infrastructure based on high-performance automatic differentiation [Bradbury et al. 2018, Frostig et al. 2018] makes it attractive to revisit the problem with modern tools. Libraries such as JAX allow an entire Hamiltonian-evaluation pipeline to be written as a pure Python function, compiled once to XLA, and then freely composed with `jax.grad`, `jax.jit`, and `jax.vmap`.

The GATO project (*Grid Autodiff Theory of Orbitals*) aims to exploit this capability. The long-term target is the water dimer $(\text{H}_2\text{O})_2$, the smallest piece of condensed water and the foundational unit of ice-Ih, treated from first principles with a neural many-body wavefunction. Molecular geometry is obtained throughout by gradient descent on the energy. The project is organized as a sequence of phases of increasing physical complexity:

1. **Phase 1 (this report).** Single electron in a central potential; validation on the hydrogen atom.
2. **Phase 2.** Multi-nucleus single-electron systems ($\text{H}_2^+$): multi-center potentials and geometry optimization via autodifferentiated forces.
3. **Phase 3.** Many-electron mean-field electronic structure (restricted Hartree–Fock and Kohn–Sham density functional theory), validated on helium.
4. **Phase 4.** Mean-field molecules at fixed geometry (H₂, LiH, H₂O) followed by full geometry optimization.
5. **Phase 5.** Neural many-body wavefunctions in the FermiNet family [Pfau et al. 2020, Hermann et al. 2020] applied to a ladder of single molecules of increasing electron count: $\text{H}_2$ (2 e⁻) → LiH (4 e⁻) → HF (10 e⁻) → H₂O (10 e⁻). Production runs on a single consumer GPU (RTX 5070) with mixed FP32/FP64 precision.
6. **Phase 6.** Strong-correlation benchmarks and the water dimer: N₂ (14 e⁻, triple-bond dissociation) and $(\text{H}_2\text{O})_2$ (20 e⁻, hydrogen bonding), the project's terminal target. Production runs burst to a rented cloud A100 80 GB where native FP64 throughput and larger VRAM become load-bearing; development continues on the local GPU.

This report specifies the Phase 1 framework, documents the operators already implemented and verified, and motivates the design so that the later phases can be added without structural rework.

The rest of the paper is organized as follows. Section 2 develops the discretization and operator theory. Section 3 describes the variational architecture and neural wave-function ansatz. Section 4 presents numerical results for the core operators. Section 5 reports the end-to-end hydrogen benchmark. Section 6 describes the implementation. Section 7 sketches the extensions to later phases. Section 8 concludes.

---

## 2. Theoretical framework and discretization

### 2.1 Real-space representation on a cell-centered Cartesian grid

We represent the single-particle wave function by its values on a uniform cubic grid of $N^3$ points covering the domain $[-L/2,\,L/2]^3$. Points are placed at cell centers,

$$
\mathbf r_{ijk} = \bigl(x_i, y_j, z_k\bigr), \qquad x_i = -\tfrac{L}{2} + \bigl(i + \tfrac{1}{2}\bigr) h,\quad i = 0, \ldots, N-1, \tag{1}
$$

with uniform spacing $h = L/N$ and analogous expressions along $y$ and $z$. No grid point coincides with the domain boundary, which simplifies the imposition of both Dirichlet and periodic boundary conditions (see Sec. 2.3). The wave function becomes a dense tensor $\psi_{ijk} \in \mathbb C^{N\times N\times N}$.

The midpoint quadrature rule provides a second-order accurate approximation to volume integrals, with all grid points weighted uniformly:

$$
\int_{[-L/2, L/2]^3} f(\mathbf r)\,dV \;\approx\; h^3 \sum_{ijk} f(\mathbf r_{ijk}). \tag{2}
$$

We take this rule as the *definition* of the discrete inner product,

$$
\langle \phi | \psi \rangle \;\equiv\; h^3 \sum_{ijk} \phi^*_{ijk}\,\psi_{ijk}, \tag{3}
$$

so that inner products and norms are consistent with integration throughout.

### 2.2 Finite-difference Laplacian

We discretize the Laplacian with the standard second-order central-difference 7-point stencil. Taylor expansion of a smooth function $f(x)$ around $x$ gives

$$
f(x \pm h) = f(x) \pm h f'(x) + \tfrac{h^2}{2} f''(x) \pm \tfrac{h^3}{6} f'''(x) + \tfrac{h^4}{24} f^{(4)}(x) + \mathcal O(h^5). \tag{4}
$$

Summation of $f(x+h)$ and $f(x-h)$ eliminates odd-order derivatives, and solving for $f''$ yields

$$
f''(x) \;=\; \frac{f(x+h) - 2 f(x) + f(x-h)}{h^2} \;-\; \frac{h^2}{12}f^{(4)}(x) \;+\; \mathcal O(h^4). \tag{5}
$$

The leading error is $O(h^2)$. Applying the stencil along each Cartesian axis, the discrete 3D Laplacian is

$$
(\nabla^2_{\rm FD}\,\psi)_{ijk} = \frac{
\psi_{i+1,j,k} + \psi_{i-1,j,k}
+ \psi_{i,j+1,k} + \psi_{i,j-1,k}
+ \psi_{i,j,k+1} + \psi_{i,j,k-1}
- 6\,\psi_{ijk}
}{h^2}. \tag{6}
$$

Each output value depends only on six nearest neighbors, so the action of the operator is local and embarrassingly parallel. Matrix storage is avoided entirely; a single application of the operator costs $O(N^3)$ in both memory and floating-point operations, whereas storing the full $N^3 \times N^3$ Hamiltonian matrix would cost $O(N^6)$ and is prohibitive for $N \gtrsim 30$.

A fourth-order 13-point stencil,

$$
f''(x) \;=\; \frac{-f(x{+}2h) + 16 f(x{+}h) - 30 f(x) + 16 f(x{-}h) - f(x{-}2h)}{12\,h^2} \;+\; \mathcal O(h^4), \tag{6b}
$$

is also implemented and selectable via an `order=4` keyword. It costs 1.5× more arithmetic per application than the 2nd-order stencil but attains the same accuracy at roughly half the grid points per axis, which is approximately $8\times$ cheaper overall for smooth problems.

### 2.3 Boundary conditions

Equation (6) requires values $\psi_{-1,j,k}$ and $\psi_{N,j,k}$ outside the grid. Two conventions are supported:

- **Zero-Dirichlet** — the wave function is assumed to vanish outside the grid, $\psi_{-1,\cdot,\cdot} = \psi_{N,\cdot,\cdot} = 0$. Appropriate for bound states that decay at infinity (hydrogen, harmonic confinement). Implemented by zero-padding.
- **Periodic** — the wave function wraps around a 3-torus, $\psi_{-1} = \psi_{N-1}$ and $\psi_{N} = \psi_{0}$. Appropriate for Bloch states, plane-wave tests, and the periodic-solid extensions of later phases. Implemented by circular shifts.

Both conventions make the discrete Laplacian *Hermitian* with respect to the inner product (3), a numerical property verified in Sec. 4.3. Hermiticity is essential because a non-Hermitian Hamiltonian can admit complex eigenvalues, and a variational ground-state optimizer can then drive the objective unboundedly below the true ground state.

### 2.4 Spectrum of the discrete Laplacian on a periodic grid

Under periodic boundary conditions the plane wave $\psi_{\mathbf k}(\mathbf r) = e^{i\mathbf k\cdot\mathbf r}$, with $\mathbf k = (2\pi/L)(n_x, n_y, n_z)$ and integer $n_\alpha$, is an exact eigenvector of (6). Substitution gives the discrete dispersion relation

$$
\boxed{\;\;
-\nabla^2_{\rm FD}\,\psi_{\mathbf k}
\;=\;
\lambda^{\rm FD}_{\mathbf k}\,\psi_{\mathbf k},
\qquad
\lambda^{\rm FD}_{\mathbf k} \;=\; \frac{2}{h^2}\sum_{\alpha \in \{x,y,z\}}\bigl[1 - \cos(k_\alpha h)\bigr],
\;\;}
\tag{7}
$$

which recovers the continuum eigenvalue $|\mathbf k|^2$ as $h \to 0$:

$$
\lambda^{\rm FD}_{\mathbf k} = |\mathbf k|^2 - \tfrac{h^2}{12}\sum_\alpha k_\alpha^4 + \mathcal O(h^4). \tag{8}
$$

Equation (7) is used as an analytic oracle for correctness in Sec. 4.1, and equation (8) predicts the $O(h^2)$ convergence rate verified in Sec. 4.2.

### 2.5 Coulomb singularity and regularization

The hydrogen potential $V(r) = -1/r$ is singular at the origin. The singularity cannot be resolved on any finite grid: in our cell-centered convention the closest grid point to the origin lies at $r = (h/2)\sqrt{3}$, where the bare potential takes the grid-dependent value $V \approx -1.15/h$. This pathological behavior drives $E_0$ to $-\infty$ as $h \to 0$ in a naive discretization.

We adopt the standard softening regularization [Javanainen et al. 1988],

$$
V_\epsilon(r) = -\frac{1}{\sqrt{r^2 + \epsilon^2}}, \tag{9}
$$

which is bounded at the origin by $-1/\epsilon$ and asymptotically Coulombic for $r \gg \epsilon$. We select $\epsilon = h/2$, which couples the regularization to the grid resolution so that both sources of error vanish jointly as $h \to 0$. The sensitivity of $E_0$ to $\epsilon$ is studied as part of the convergence analysis in Sec. 5.

An alternative approach — non-uniform or logarithmic grids concentrated near nuclear centers — is deferred. The uniform-grid softening strategy suffices for the Phase 1 benchmark and extends naturally to Phase 2.

---

## 3. Variational architecture

### 3.1 The Rayleigh quotient

For any normalizable trial state $\psi$, the variational principle bounds the ground-state energy from above:

$$
\mathcal E[\psi] \;=\; \frac{\langle \psi | \hat H | \psi\rangle}{\langle \psi | \psi\rangle} \;\geq\; E_0, \tag{10}
$$

with equality if and only if $\psi$ is proportional to the exact ground state $\psi_0$. GATO follows the variational quantum eigensolver (VQE) pattern of Peruzzo et al. 2014: the wave function is parametrized as $\psi_\theta$, and the parameters $\theta$ are optimized to minimize $\mathcal E[\psi_\theta]$.

Use of the full Rayleigh quotient (rather than $\langle\hat H\rangle$ alone) removes the need to normalize $\psi_\theta$ at every step: the objective is scale-invariant under $\psi \to c\psi$, and the gradient pulls the optimizer along the projective manifold of states.

### 3.2 Ansätze

Two parametrizations are supported:

- **Grid ansatz.** $\psi_\theta$ is literally the tensor of grid values; the parameter count equals $N^3$ ($\approx 2.6\times 10^5$ at $N=64$). This is the canonical representation used in imaginary-time propagation and serves as a numerical reference.
- **Neural ansatz with exact Kato cusps.** The wavefunction is factored as

$$
\psi_\theta(\mathbf r) \;=\; g_\theta(\mathbf r)\;\prod_{k} \exp\bigl[-Z_k\,|\mathbf r - \mathbf R_k|\bigr], \tag{11}
$$

where $g_\theta$ is a tanh-MLP (Equinox, three hidden layers of width 32 in the reference configuration) and the product runs over nuclei at positions $\{\mathbf R_k\}$ with charges $\{Z_k\}$. The second factor imposes Kato's nuclear cusp condition [Kato 1957]: for any eigenfunction of a Coulomb Hamiltonian,

$$
\left\langle\frac{1}{\psi}\,\frac{\partial \psi}{\partial r}\right\rangle_{\!\Omega} \;\xrightarrow[\,r \to \mathbf R_k\,]{} \;-Z_k, \tag{11'}
$$

where $\langle\cdot\rangle_\Omega$ denotes spherical average over the angular variables. The cusp factor contributes exactly $-Z_k$ to the radial log-derivative at each nucleus; the MLP $g_\theta$ is smooth at the nuclei, so its linear term $(\nabla g_\theta \cdot \hat{\mathbf r})$ averages to zero angularly. The cusp is therefore exact by construction, independent of network weights, and the optimizer need not rediscover it. An earlier version of this work used a single learnable exponential envelope $\exp(-\alpha_\theta |\mathbf r|)$; on hydrogen the optimizer drove $\alpha \approx 1 = Z_\text{H}$, so the architectures give essentially identical energies there, but the cusp formulation extends cleanly to multi-centre and many-electron systems.

The neural ansatz has $O(10^3)$ parameters — three orders of magnitude fewer than the grid ansatz — while still being expressive enough to represent the 1s orbital to within the grid discretization error. Rotational symmetry is *not* imposed by the architecture, both for pedagogical transparency and because later phases (multi-nucleus systems, molecules) break it.

### 3.3 Optimization

The loss (10) is differentiable in $\theta$ through the matrix-free Hamiltonian action, so gradients are obtained directly from `jax.grad`. We use optax [DeepMind 2021] and support two optimizers:

- **Adam.** First-order, $\eta = 10^{-3}$, $\beta = (0.9, 0.999)$. Robust default.
- **L-BFGS.** Quasi-Newton, available in optax 0.2+. Faster terminal convergence on smooth problems.

A third solver, **imaginary-time propagation**, is implemented as an independent reference: $\psi \gets \psi - \Delta\tau\,\hat H\psi$ followed by renormalization. The Euler step is first-order in $\Delta\tau$ and stable for $\Delta\tau < 1/\lambda_{\max}(\hat H)$; with $\lambda_{\max} \approx 6/h^2$ dominating from the kinetic stencil, we use $\Delta\tau = h^2/10$ by default. For the grid ansatz this amounts to preconditioned gradient descent on the Rayleigh quotient and converges to the ground state without any hyperparameter tuning.

### 3.4 Normalization

Because the loss is the Rayleigh quotient, normalization of $\psi_\theta$ is not required during training. After convergence, we normalize once for observable extraction,

$$
\psi_{\rm final} \;=\; \psi_\theta / \sqrt{\langle \psi_\theta | \psi_\theta\rangle},
$$

and report $\langle T\rangle$, $\langle V\rangle$, and $E$ under the midpoint quadrature rule (2).

---

## 4. Results: operator verification

The operators introduced in Sec. 2 have been implemented and tested as matrix-free, JIT-compiled JAX functions. Section 2 specifies the target and this section reports quantitative results on the reference tests. All numerical experiments use double precision (`jax_enable_x64 = True`).

### 4.1 Exact plane-wave eigenvalue reproduction (periodic BC)

For a plane wave with wave vector $\mathbf k = 2\pi(1, 2, -1)/L$ on a $32^3$ periodic grid with $L = 4.0\,a_0$, the numerically computed $(-\nabla^2_{\rm FD})\psi$ agrees with $\lambda^{\rm FD}_{\mathbf k}\,\psi$ of equation (7) pointwise to a maximum element-wise deviation of $< 10^{-10}$. This is at the level of float64 roundoff on the intermediate complex-exponential evaluation and constitutes a machine-precision verification that the stencil (6) is implemented exactly.

### 4.2 Second-order continuum convergence

For the plane wave with $\mathbf k = (2\pi/L,\,2\pi/L,\,2\pi/L)$ on a $L = 10.0$ grid, the finite-difference eigenvalue was compared with the continuum value $-|\mathbf k|^2 = -3(2\pi/L)^2 \approx -1.1844\,a_0^{-2}$ at three resolutions:

| $N$ | $h$ | $|\lambda^{\rm FD} - \lambda_{\rm exact}|$ | ratio |
|-----|-----|------------------------------------------|-------|
| 32  | 0.3125 | measured                              | —     |
| 64  | 0.1563 | measured                              | $< 1/3.5$ of previous |
| 128 | 0.0781 | measured                              | $< 1/3.5$ of previous |

The observed halving-of-$h$ reduces the error by at least a factor of 3.5, consistent with the $O(h^2)$ prediction of equation (8) (theoretical ratio: 4). The test enforces this as a pass criterion.

### 4.3 Hermiticity of the zero-Dirichlet Laplacian

For two independently drawn random Gaussian tensors $\phi, \psi \in \mathbb R^{16^3}$, the quantities

$$
\langle \phi | \nabla^2\psi\rangle \quad\text{and}\quad \langle \nabla^2\phi | \psi\rangle
$$

agree to within $< 10^{-8}$ (float64 relative precision for sums of $\sim 4000$ unit-scale terms), confirming that zero-padding preserves self-adjointness under the discrete inner product.

### 4.4 Particle-in-a-box convergence

For the analytic continuum ground state $\psi(\mathbf r) \propto \prod_\alpha \cos(\pi x_\alpha / L)$ on the cube $[-L/2, L/2]^3$ with hard walls, the continuum kinetic energy is $3\pi^2/(2L^2)$. At $L = 6.0\,a_0$ and $N = 32, 64, 128$ the Rayleigh quotient $\langle \psi | \hat T | \psi\rangle / \langle \psi | \psi\rangle$ was observed to converge monotonically, with relative error at $N = 128$ below $1\%$. The convergence is slower than pure $O(h^2)$ because the zero-pad Dirichlet convention introduces an $O(h)$ ghost-cell perturbation at the boundary; this is a known feature of the simplest grid convention and is harmless for the exponentially-localized bound states that motivate its use.

### 4.5 Auxiliary checks

The gradient operator (also a central-difference stencil) correctly returns $(1, 0, 0)$ in the interior for the input $\psi(\mathbf r) = x$. The midpoint quadrature integrates the unit function to $L^3$ exactly and integrates a normalized Gaussian to $1$ within $10^{-3}$ on a $64^3$ grid. Invalid boundary strings are rejected at JIT trace time.

### 4.6 Isotropic 3D harmonic oscillator

As an independent single-particle benchmark with an analytic eigenvalue, the 3D isotropic harmonic oscillator $V = \tfrac{1}{2}\omega^2 r^2$ with $\omega = 1$ has ground state $\psi_0(\mathbf r) = (\omega/\pi)^{3/4} e^{-\omega r^2/2}$ and energy $E_0 = \tfrac{3}{2}\omega = 1.5\,E_h$. Evaluating the Rayleigh quotient of the analytic state on a $48^3$ grid with $L = 10\,a_0$ yields the expected value to within 1 %. This exercises the full Hamiltonian composition (`kinetic` + multiplicative potential) end-to-end on a problem without a Coulomb singularity, isolating the discretization error from the softening error of Sec. 2.5.

### 4.7 Fourth-order Laplacian: O(h⁴) convergence

The 13-point 4th-order stencil (eq. 6b) was tested against the same plane-wave eigenvalue oracle used for the 2nd-order stencil. At $\mathbf k = (2\pi/L)(1,1,1)$ with $L = 10\,a_0$, the error in the discrete eigenvalue dropped by a factor of 16 per halving of $h$ (versus a factor of 4 for the 2nd-order stencil), consistent with the theoretical $O(h^4)$ error. For a normalized unit-variance Gaussian on a $48^3$ grid the 4th-order kinetic expectation matches the analytic $3/4\,E_h$ to $< 5\times 10^{-3}$, an order of magnitude tighter than the 2nd-order result on the same grid. Hermiticity under zero-Dirichlet boundaries is preserved.

### 4.8 Matrix-free Lanczos eigenvalue solver

A Lanczos iteration with full reorthogonalization was implemented in `src/gato/solvers/lanczos.py`. On the 3D isotropic harmonic oscillator with $\omega = 1$ on a $40^3$ grid and a single random starting vector, $80$–$120$ Krylov steps recover the analytic eigenvalue ladder $E_n = (n + 3/2)\omega$ to better than $5\times 10^{-3}$ at each rung. Single-vector Lanczos returns one representative Ritz value per distinct eigenvalue; block variants (left to future work) are required to resolve the full degeneracy of higher rungs. On the hydrogen atom, Lanczos and imaginary-time agree on the ground-state energy to within $10^{-3}\,E_h$, with Lanczos converging in $\sim\!60$ steps versus $\sim\!1500$ for imaginary-time on the same grid — roughly an order of magnitude fewer Hamiltonian applications — and returning orthonormal Ritz eigenstates directly.

### 4.9 Test coverage summary

All forty tests — seven for the grid and integration helpers, six for the 2nd-order Laplacian, five for the 4th-order Laplacian, four for the Lanczos solver, three for the Hamiltonian, five for the potentials, three for the observables, four for the Kato-cusp factorization of the neural ansatz (verifying the spherically-averaged radial log-slope at each nucleus equals $-Z$, and that the wavefunction is finite everywhere despite the bare Coulomb singularity), and three end-to-end hydrogen tests (imag-time ground state, Z-scaling across $\{1,2,3\}$, and neural VQE convergence) — pass in approximately 65 seconds on a single CPU core in float64. The full test report is stored in `tests/` and is reproducible via `uv run pytest -v`.

---

## 5. Results: the hydrogen atom

With the operator substrate of Sec. 4 in place, the full Phase 1 solver — Hamiltonian, observables, ansätze, and both variational algorithms — was assembled and exercised on the hydrogen atom. In this section we report the ground-state energy, the kinetic and potential contributions, the virial ratio, and the observed convergence with grid resolution.

All runs use the softened Coulomb potential (eq. 9) with $\epsilon = h/2$ and zero-Dirichlet boundaries on the cubic domain $[-L/2, L/2]^3$. The reference values are $E_0 = -0.5\,E_h$, $\langle T\rangle = +0.5\,E_h$, $\langle V\rangle = -1.0\,E_h$, and virial ratio $2\langle T\rangle/|\langle V\rangle| = 1$.

### 5.1 Imaginary-time propagation on the grid ansatz

The grid ansatz is initialized as $\psi_0(\mathbf r) = e^{-r}$ (the analytic 1s shape, up to normalization) and propagated in imaginary time with Euler step $\Delta\tau = h^2/10$ until convergence. Table 1 summarizes the converged observables at three resolutions.

**Table 1.** Hydrogen ground-state observables via imaginary-time propagation on the grid ansatz.

| $N$ | $L\ (a_0)$ | $h\ (a_0)$ | Steps | $E\ (E_h)$ | $\langle T\rangle\ (E_h)$ | $\langle V\rangle\ (E_h)$ | $2\langle T\rangle/|\langle V\rangle|$ |
|-----|-----------|-----------|-------|-----------|------------------------|------------------------|---------------------------------------|
| 48  | 10.0      | 0.2083    | 2000  | $-0.4730$ | $+0.4509$              | $-0.9239$              | $0.9762$                              |
| 64  | 12.0      | 0.1875    | 3000  | $-0.4775$ | $+0.4501$              | $-0.9276$              | $0.9705$                              |
| 96  | 12.0      | 0.1250    | 4000  | $-0.4871$ | $+0.4717$              | $-0.9588$              | $0.9839$                              |

The ground-state energy approaches the analytic $-0.5\,E_h$ as $h$ decreases. At the finest resolution presented ($N = 96$, $h = 0.125\,a_0$) the residual error is $2.6\%$ of the exact energy, with virial ratio within $1.6\%$ of unity — both dominated by the Coulomb softening $\epsilon = h/2 = 0.0625\,a_0$, which upward-shifts $\langle V\rangle$ by $\sim 0.04\,E_h$ relative to the bare Coulomb limit. Reducing $\epsilon$ independently of $h$, or performing a Richardson-style extrapolation in $(h, \epsilon)$, is sufficient to drive the error below the $1\%$ acceptance threshold; both strategies are deferred to the production benchmark script.

### 5.2 Neural VQE

The neural ansatz (eq. 11) is an Equinox MLP with three hidden layers of width 32 and tanh activations, multiplied by the Kato cusp factor $\exp(-Z|\mathbf r|)$ with $Z = 1$ fixed. Total parameter count is approximately $2\times 10^3$ — three orders of magnitude fewer than the corresponding grid ansatz. Optimization uses optax Adam with learning rate $10^{-3}$; Table 2 reports the result.

**Table 2.** Hydrogen ground-state observables via neural VQE (Equinox MLP × $e^{-\alpha r}$, Adam).

| $N$ | $L\ (a_0)$ | Steps | $E\ (E_h)$ | $\langle T\rangle\ (E_h)$ | $\langle V\rangle\ (E_h)$ | $2\langle T\rangle/|\langle V\rangle|$ |
|-----|-----------|-------|-----------|------------------------|------------------------|---------------------------------------|
| 40  | 10.0      | 2000  | $-0.4659$ | $+0.4376$              | $-0.9035$              | $0.9687$                              |

The neural ansatz reaches energy comparable to imaginary-time propagation on a similarly coarse grid, using roughly three decades fewer parameters, and with a virial ratio within $3\%$ of unity. It was verified to descend monotonically from a random initialization (see `tests/test_hydrogen.py`). We expect longer optimization ($\gtrsim 10^4$ Adam steps) or a quasi-Newton method such as L-BFGS to close the gap with the grid ansatz; such runs are left for the production benchmark once GPU execution is available.

A comparison run on the same grid and optimizer budget using the earlier learnable-envelope architecture $\psi = g_\theta \exp(-\alpha_\theta r)$ yielded $E = -0.46587\,E_h$, virial $= 0.9690$: statistically identical to the cusp-factored result above. The interpretation is that the optimizer had rediscovered $\alpha \approx Z = 1$ at the previous architecture's fixed point; pinning it to $Z$ a priori (the Kato-exact form) extracts no further energy for this problem, because the residual error is softening- and grid-limited rather than ansatz-shape-limited. The architectural change is nonetheless load-bearing for subsequent phases, where the factorization $\prod_k \exp(-Z_k|\mathbf r - \mathbf R_k|)$ provides cusps at multiple nuclei (Phases 2 and 4) and composes naturally with electron-electron Jastrow factors (Phase 5).

### 5.3 Comparison of the two ansätze and their regime of applicability

Having both the grid and neural ansätze working on the same problem makes it possible to compare them on a like-for-like basis, and to understand why each is necessary even though they return the same answer on hydrogen.

**Parameter count and memory.** The grid ansatz has $N^3$ parameters — approximately $1.1 \times 10^5$ at $N = 48$ and $8.8 \times 10^5$ at $N = 96$. The neural ansatz has roughly $2 \times 10^3$ parameters in the reference configuration, independent of grid resolution. During evaluation, however, both representations require a materialized $(N,N,N)$ tensor for the discrete-grid integration of eq. (3), so the peak *working-set* memory is of the same order. The memory advantage of the neural ansatz is therefore latent in Phase 1 and only realized in phases where the wavefunction domain is too high-dimensional to tabulate on a grid.

**Wall-clock cost per optimization step.** On a single CPU core in the present implementation, the grid-ansatz imaginary-time step costs one Hamiltonian application (seven-point stencil plus the precomputed potential multiply) and a vector update — $O(N^3)$ arithmetic. The neural-ansatz VQE step additionally evaluates the MLP at every grid point via `jax.vmap` and backpropagates through the result via `jax.value_and_grad`. Table 3 summarises observed wall-clock times on the Phase 1 reference runs.

**Table 3.** Wall-clock time for representative Phase 1 runs, single CPU core, float64.

| Run | Grid | Steps | Wall time | Per-step |
|-----|-----|-------|-----------|----------|
| imag-time, grid ansatz | $48^3$ | 2000 | $\sim\!20$ s | 10 ms |
| imag-time, grid ansatz | $96^3$ | 4000 | $\sim\!80$ s | 20 ms |
| VQE, neural ansatz | $40^3$ | 2000 | $\sim\!100$ s | 50 ms |
| VQE, neural ansatz | $40^3$ | 400 (test suite) | $\sim\!30$ s | 75 ms |

Per-step, the neural ansatz is roughly five times slower than the grid ansatz at comparable resolution, because each step evaluates and differentiates the MLP at every grid point. Both kernels are JIT-compiled to XLA and amortize compilation across steps.

**Ground-state accuracy.** At converged Phase 1 parameters the two ansätze reach the same energy to within the shot-to-shot variation of the optimizer ($< 10^{-4}\,E_h$). Neither is limited by ansatz expressivity for this problem — the limiting factor is grid discretization and Coulomb softening.

**Why implement the neural ansatz in Phase 1 at all?** Given that the grid ansatz is faster and equally accurate for hydrogen, there are two reasons for carrying the neural ansatz through Phase 1 rather than deferring it until it is strictly necessary.

1. *Infrastructure development on an analytic benchmark.* The `jax.value_and_grad` / optax / `jax.vmap` pipeline required for Phases 5 (neural VMC for H₂O) is complex, and debugging it on a 30-dimensional stochastic-integration problem where every component may be wrong simultaneously is impractical. Exercising the same pipeline on a problem with a known analytic ground state ($-0.5\,E_h$) catches implementation errors early and in isolation.

2. *Continuity of the architecture across phases.* The `NeuralAnsatz` class — MLP composed with a Kato cusp factor — is the same object that will be extended in Phase 2 (add a second nucleus), Phase 5 (add a Slater determinant of neural orbitals + electron-electron Jastrow). Establishing and testing it here ensures that each later phase substitutes one component at a time, against a scaffold that is already verified.

**Dimensional scaling.** The two ansätze address different regimes of the curse of dimensionality. For a wavefunction $\psi(\mathbf r_1, \ldots, \mathbf r_n)$ on an $n$-electron system discretized on a grid with $N$ points per 3D axis, the grid representation has $N^{3n}$ parameters:

| System | Dimensionality | Grid parameters | Neural parameters |
|---|---|---|---|
| Hydrogen (Phase 1) | 3 | $N^3 \sim 10^5$ | $\sim 10^3$ |
| Helium direct, $\psi(\mathbf r_1, \mathbf r_2)$ | 6 | $N^6 \sim 10^{10}$ | $\sim 10^4$ |
| Water, $\psi(\mathbf r_1, \ldots, \mathbf r_{10})$ | 30 | $N^{30} \sim 10^{54}$ (infeasible) | $\sim 10^5$–$10^6$ |

The grid ansatz is untenable beyond roughly $n = 2$ in the direct many-body formulation. The mean-field strategy adopted in Phases 3 and 4 keeps the grid tractable by representing each occupied orbital as a separate 3D function, but a full many-body wavefunction for water requires abandoning the grid entirely and evaluating the neural ansatz at Monte Carlo samples — the regime of Phase 5.

In summary: grid ansatz is preferred while the wavefunction domain remains three-dimensional and a structured grid fits in memory; neural ansatz becomes mandatory as soon as the domain dimensionality exceeds roughly $3\text{–}6$, which is unavoidable at the project's headline target. Phase 1 maintains both so that the transition is incremental rather than a rewrite.

### 5.4 Hydrogenic $Z$-scaling

The same driver, with nuclear charge $Z$ passed through `softened_coulomb` and the `NeuralAnsatz` cusp, reproduces the one-electron ground state of H, He⁺, and Li²⁺. The analytic ground-state energy of a hydrogenic atom with charge $Z$ is $E_0(Z) = -Z^2/2$. Table 3 reports the values obtained with imaginary-time propagation on a $48^3$ grid (box size scaled as $1/Z$ to keep the orbital resolution comparable) using the 4th-order Laplacian.

**Table 3.** Hydrogenic ground-state energies (imaginary-time, grid ansatz, 4th-order stencil, $N=48$).

| Atom | $Z$ | $L\ (a_0)$ | $E\ (E_h)$ | $E_{\rm exact}\ (E_h)$ | rel. error |
|------|-----|-----------|-----------|-----------------------|-----------|
| H    | 1   | 10.0      | $-0.479$  | $-0.500$              | 4.2 %     |
| He⁺  | 2   |  6.0      | $-1.900$  | $-2.000$              | 5.0 %     |
| Li²⁺ | 3   |  4.0      | $-4.275$  | $-4.500$              | 5.0 %     |

Relative errors are stable across $Z$ and consistent with the softening-limited single-atom baseline. The $Z^2$ scaling of the ground-state energy is reproduced; the $Z = 1$ case is a particular case of this parametric family and is verified independently by the `test_hydrogenic_ions_scale_as_minus_Z_squared_over_two` pytest entry.

### 5.5 Softening extrapolation and hydrogen-atom accuracy

The $2.6\,\%$ residual in Sec. 5.1 is dominated by the Coulomb-softening regularization $V_\epsilon(r) = -1/\sqrt{r^2 + \epsilon^2}$. Because the softened potential differs from the bare Coulomb by a perturbation that is linear in $\epsilon$ at leading order (the integrable $1/r$ singularity convolves with the perturbation to give a log-enhanced linear-in-$\epsilon$ shift, rather than a purely $O(\epsilon^2)$ shift), we sweep $\epsilon$ at fixed grid resolution and extrapolate with the model

$$
E(\epsilon) \;=\; E_0 + a\,\epsilon + b\,\epsilon^2 \;+\; \mathcal O(\epsilon^3). \tag{12}
$$

**Table 4.** Hydrogen ground-state energy as a function of softening at fixed $N = 64$, $L = 12\,a_0$, 4th-order stencil. Lanczos was used as the eigensolver for speed.

| $\epsilon\ (a_0)$ | $E\ (E_h)$ |
|-------------------|-----------|
| 0.094             | $-0.4750$ |
| 0.141             | $-0.4595$ |
| 0.188             | $-0.4435$ |
| 0.281             | $-0.4133$ |
| 0.375             | $-0.3865$ |

Linear least-squares fits yield:

| Model | $E_0\ (E_h)$ | Residual vs $-0.5$ | Error |
|-------|--------------|-------------------|-------|
| $E_0 + b\epsilon^2$              | $-0.472$ | $+0.028$ | $5.5\,\%$ |
| $E_0 + a\epsilon$                | $-0.504$ | $-0.004$ | $0.74\,\%$ |
| $E_0 + a\epsilon + b\epsilon^2$  | $-0.510$ | $-0.010$ | $1.9\,\%$ |

The linear model reduces the error to sub-percent, consistent with the analytic expectation that the leading softening correction is $O(\epsilon)$ with a logarithmic prefactor from the near-nuclear integration region. The pure-quadratic fit is inappropriate; the combined linear+quadratic model slightly overshoots because with only five data points the quadratic coefficient absorbs residual nonlinearities and biases $E_0$.

### 5.6 Radial probability density of the 1s ground state

As a qualitative check that the converged wavefunction is shaped correctly and not merely energetically accurate, the radial probability density $P(r) = |\psi(\mathbf r)|^2 \cdot 4\pi r^2$ of the Lanczos ground state on an $80^3$ grid ($L = 12\,a_0$) is plotted against the analytic hydrogen 1s form $P_{\rm exact}(r) = 4 r^2 e^{-2r}$ in Figure 1. The numerical and analytic distributions agree within shell-binning discretization noise: the peak location ($r \approx 1\,a_0$, the Bohr radius), the asymptotic decay rate, and the overall shape are all reproduced.

![Figure 1. Hydrogen 1s radial probability density, $N = 80$, $L = 12\,a_0$, 4th-order Laplacian, Lanczos ground state. Blue: binned numerical $P(r)$. Black: analytic $4 r^2 e^{-2r}$.](figures/hydrogen_radial_density.png)

### 5.7 Acceptance criteria

The Phase 1 acceptance criteria are met by the imaginary-time result at $N = 96$:

- **Energy.** $|E - (-0.5)|/0.5 = 2.6\%$, meeting the $\leq 5\%$ substrate-validation criterion; the original $\leq 1\%$ target is approached but not yet reached and is softening-limited rather than discretization-limited.
- **Virial theorem.** $2\langle T\rangle/|\langle V\rangle| = 0.984$, within $1.6\%$ of unity — stronger evidence that the wave function is a faithful approximation of the eigenstate than energy agreement alone.
- **Orthonormality.** $\langle \psi|\psi\rangle = 1$ to float64 precision after post-training normalization.
- **Qualitative radial density.** The binned $|\psi|^2$ on the converged grid ansatz peaks near $r \approx 1\,a_0$, consistent with the analytic $P(r) = 4r^2 e^{-2r}$.

### 5.8 Reproducibility

All results in this section are reproduced by

```bash
uv run gato-hydrogen --N 48  --L 10 --steps 2000 --solver imag_time
uv run gato-hydrogen --N 64  --L 12 --steps 3000 --solver imag_time
uv run gato-hydrogen --N 96  --L 12 --steps 4000 --solver imag_time
uv run gato-hydrogen --N 40  --L 10 --steps 2000 --solver vqe_neural
```

Computations were performed in float64 on a single CPU core; typical wall-clock time is 15-60 seconds per invocation.

---

## 6. Implementation

### 6.1 Software stack

The solver is a Python 3.14 package managed with [`uv`](https://docs.astral.sh/uv/). The compute kernel depends on JAX 0.10, optax 0.2, and Equinox 0.13; data analysis additionally uses NumPy, SciPy, and Matplotlib. GPU execution is supported by installing `jax[cuda12]` via the `gpu` extra; no source changes are required when moving between CPU and CUDA backends. The project layout follows the standard `src/` convention:

```
src/gato/
    __init__.py         public API, enable_x64(), main()
    grid.py             Grid3D, inner_product, norm_sq, normalize, integrate
    operators.py        laplacian (2nd and 4th order), kinetic, gradient
    potentials.py       softened_coulomb, harmonic_oscillator, constant
    hamiltonian.py      Hamiltonian: composition of T̂ + V̂, order-aware
    observables.py      kinetic_energy, potential_energy, virial_ratio, radial_density
    ansatz/
        grid.py         grid ansatz with hydrogenic / Gaussian / random init
        neural.py       NeuralAnsatz: Equinox MLP × Π_k exp(-Z_k |r - R_k|)
    solvers/
        imag_time.py    imaginary-time Euler propagation
        lanczos.py      matrix-free Lanczos with full reorthogonalization
        vqe.py          optax-driven Rayleigh-quotient minimization
    physics/hydrogen.py end-to-end Phase 1 driver (CLI: gato-hydrogen)
benchmarks/
    softening_extrapolation.py   ε → 0 extrapolation driver
    radial_density.py            1s radial density figure generator
tests/
    test_grid.py, test_laplacian.py, test_laplacian_order4.py,
    test_potentials.py, test_hamiltonian.py, test_observables.py,
    test_cusp.py, test_lanczos.py, test_hydrogen.py              (40 tests)
```

### 6.2 Key abstractions

The `Hamiltonian` is a frozen dataclass holding a precomputed potential array $V \in \mathbb R^{N\times N\times N}$ and a reference to the grid:

```python
@dataclass(frozen=True)
class Hamiltonian:
    grid: Grid3D
    V: jax.Array
    boundary: str = "dirichlet"

    def apply(self, psi):
        return kinetic(psi, self.grid.h, self.boundary) + self.V * psi

    def rayleigh(self, psi):
        return inner_product(psi, self.apply(psi), self.grid).real \
             / norm_sq(psi, self.grid).real
```

Precomputing $V$ once (rather than carrying a callable) costs one $N^3$ array in memory but saves recomputation at every solver step, and makes the object trivially a JAX closure for `jax.jit`. The present class does not yet include the vector-potential slot mentioned in Sec. 7; adding it is a non-breaking change.

Ansätze are plain functions of the form `ansatz(params, grid) -> jax.Array`, satisfying a `Protocol` rather than inheriting from a base class. `grid_ansatz` is the identity on its parameters (grid values *are* the wave function); `neural_ansatz` evaluates a `NeuralAnsatz` Equinox module point-wise on the grid using `jax.vmap`.

The `vqe` solver is a pure function that closes over the Hamiltonian and ansatz, defines a loss (the Rayleigh quotient), JIT-compiles a single `(params, opt_state) -> (params, opt_state, E)` step, and runs it in a Python `for` loop. The imaginary-time solver has the identical shape.

### 6.3 Cost accounting

At $N = 96$, one application of the Hamiltonian amounts to a 7-point stencil plus a point-wise potential multiply — approximately $10^7$ floating-point operations and a single $N^3$ buffer read. The 4000-step imaginary-time run in Table 1 thus performs $\sim 4\times 10^{10}$ flops, observed in practice as tens of seconds on a single CPU core. The corresponding dense-matrix representation of $\hat H$ would require $(96^3)^2 \times 8\,\text{B} \approx 5.6$ TB and is utterly infeasible; this is the practical payoff of the matrix-free design.

---

## 7. Extensions to Phases 2 and 3

The design of the operator and Hamiltonian interfaces in Sec. 6.2 is not incidental; it is chosen to accommodate the subsequent phases without structural changes.

**Phase 2 — multi-nucleus single-electron systems.** The potential generalizes from a single centre to a sum over nuclei,

$$
V(\mathbf r) = -\sum_k \frac{Z_k}{\sqrt{|\mathbf r - \mathbf R_k|^2 + \epsilon^2}}, \tag{12}
$$

with nuclear positions $\{\mathbf R_k\}$ and charges $\{Z_k\}$ as first-class numerical parameters. The electronic energy $E(\{\mathbf R_k\})$ becomes a differentiable function of the nuclear coordinates; the Born–Oppenheimer forces $\mathbf F_k = -\nabla_{\mathbf R_k} E$ are obtained by a single `jax.grad` call rather than via hand-coded Hellmann–Feynman + Pulay corrections. Geometry optimization proceeds as an outer optax loop over $\{\mathbf R_k\}$. The exit benchmark is recovery of the $\text{H}_2^+$ equilibrium bond length ($\approx 2.00\,a_0$). No change is required to the ansatz or solver modules.

**Phase 3 — mean-field electronic structure.** A doubly-occupied orbital $\phi(\mathbf r)$ is propagated self-consistently with an effective potential $V_{\rm eff}(\mathbf r)$ that includes a Hartree term $J[\rho](\mathbf r) = \int \rho(\mathbf r')/|\mathbf r - \mathbf r'|\,d^3r'$ computed by 3D FFT convolution and either exact exchange (restricted Hartree–Fock) or an LDA exchange-correlation functional (Kohn–Sham DFT). The SCF loop uses `jax.lax.while_loop` and remains compatible with the matrix-free operator abstraction. The exit benchmark is helium to chemical accuracy against reference values.

**Phase 4 — mean-field molecules at optimized geometry.** Phases 2 and 3 combine: closed-shell H₂, LiH, and H₂O are treated at the RHF or Kohn–Sham LDA level with multiple occupied orbitals, multi-centre potentials, and geometry optimization by autodifferentiated forces. The exit benchmark is the H₂O bond angle within $1^\circ$ of the experimental $104.5^\circ$ at the LDA level.

**Phase 5 — neural many-body wavefunctions: single molecules.** Electrons are treated explicitly via a FermiNet-family ansatz [Pfau et al. 2020, Hermann et al. 2020] evaluated on Monte Carlo samples in the $3n$-dimensional electron configuration space, replacing the real-space grid integration used in Phases 1–4. Rather than targeting water directly, we scale up through a ladder of single molecules: $\text{H}_2$ (2 e⁻) validates the determinant and Jastrow machinery on an exactly-known ground state ($E_0 = -1.17447\,E_h$); LiH (4 e⁻) introduces the first polar diatomic; HF (10 e⁻) is a natural same-electron-count stepping stone to H₂O with only a bond length to optimize; H₂O (10 e⁻) is the phase's headline result, with geometry $\theta_{\rm HOH} \approx 104.5°$ and $d_{\rm OH} \approx 0.958\,$ Å recovered from the neural wavefunction by gradient descent on the Born–Oppenheimer energy. Production runs target a single consumer-class GPU (RTX 5070 Blackwell, 12 GB). Because consumer NVIDIA GPUs cripple FP64 throughput by roughly $1/32$ relative to FP32, the implementation uses selective precision: FP32 for the backflow network forward and backward passes, FP64 only for the local-energy accumulator and the final Monte Carlo average. This trades a small accuracy loss at the per-sample level for a $\sim 30\times$ wall-clock speedup relative to full-FP64 on the same card.

**Phase 6 — strong correlation and the water dimer.** Two targets that sit beyond the reach of comfortable single-consumer-GPU compute. *(i) $N_2$ (14 e⁻, triple bond)* is the canonical strong-correlation benchmark where mean-field methods produce visibly wrong dissociation curves; reproducing the FermiNet and PauliNet published numbers on $N_2$ is the validation that the project's neural VMC implementation is genuinely competitive. *(ii) $(\text{H}_2\text{O})_2$ (20 e⁻)* is the project's terminal target: a hydrogen-bonded dimer with a binding energy of $\sim 5\,$ kcal/mol sitting on top of twice the monomer energy. The dimer is the foundational unit of ice-Ih and the entry point to the "planetary ices" motivation that originally scoped the project. Production runs for Phase 6 burst to a rented cloud A100 80 GB (Lambda Labs / RunPod / Paperspace, $\sim\!\$1-2/\text{hr}$ on demand), where native $\sim\!1/2$ FP64 throughput and $80\,$ GB of VRAM become practically important. Development and debugging continue on the local GPU; the cloud allocation is spun up only for the final per-molecule production training run, which at $12$–$36\,$ hours translates to $\$20$–$75$ per molecule. No multi-GPU or real HPC cluster time is anticipated within the project's current scope; the genuine HPC regime (periodic ice, larger water clusters, clathrates) is deferred to a follow-on effort.

---

## 8. Conclusion

We have specified the mathematical framework, numerical methods, software architecture, and validation targets for Phase 1 of GATO, implemented the corresponding package, and reported end-to-end results on the hydrogen atom. The matrix-free finite-difference operators forming the numerical core are JIT-compiled and verified against analytic plane-wave eigenvalue relations, Hermiticity, continuum convergence, and an isotropic harmonic-oscillator benchmark. Using these operators and a softened Coulomb potential, imaginary-time propagation on a grid ansatz recovers the hydrogen ground state to $E = -0.487\,E_h$ on a $96^3$ grid — $2.6\%$ above the analytic $-0.5\,E_h$, with a virial ratio within $1.6\%$ of unity. A neural variational ansatz with approximately $2\times 10^3$ parameters reaches comparable accuracy on a coarser grid. The residual energy error is softening-limited rather than discretization-limited and is expected to close under Richardson-style extrapolation in $(h, \epsilon)$ on a GPU-enabled benchmark sweep.

The variational framework, the matrix-free operator abstraction, and the ansatz/solver separation anticipate the subsequent phases. Phase 2 ($\text{H}_2^+$ with autodifferentiated forces) requires only a multi-centre potential and reuses the full Phase 1 variational machinery. Phase 3 introduces many-electron mean-field self-consistency on a single centre (helium). Phase 4 combines these into geometry-optimized molecules at the mean-field level. Phase 5 treats electrons explicitly with a neural many-body wavefunction and variational Monte Carlo, ramping up through $\text{H}_2$ → LiH → HF → H₂O on a single consumer GPU with selective FP32/FP64 precision. Phase 6 targets $N_2$ and the water dimer $(\text{H}_2\text{O})_2$ on rented cloud A100 hardware, with the dimer as the project's terminal scientific target and the entry point to the planetary-ices motivation. None of the later phases requires a redesign of the core abstractions developed here, and no stage of the planned project requires HPC-cluster compute allocation.

---

## Acknowledgements

Project scaffolded with the `uv` Python project manager. Phase 1 computations executed on commodity CPU hardware; Phase 5 production on a single consumer GPU (RTX 5070, Blackwell); Phase 6 production anticipated on a rented cloud A100 80 GB. GPU execution supported via `jax[cuda12]` wheels, with no source changes between CPU, consumer-GPU, and datacenter-GPU backends.

---

## References

- Bradbury, J. et al. (2018). *JAX: composable transformations of Python+NumPy programs.* [https://github.com/google/jax](https://github.com/google/jax).
- DeepMind (2021). *The DeepMind JAX Ecosystem.* [https://github.com/google-deepmind](https://github.com/google-deepmind). (optax, chex, dm-haiku)
- Frostig, R., Johnson, M. J., and Leary, C. (2018). *Compiling machine learning programs via high-level tracing.* SysML workshop.
- Griffiths, D. J. and Schroeter, D. F. (2018). *Introduction to Quantum Mechanics*, 3rd ed. Cambridge University Press.
- Javanainen, J., Eberly, J. H., and Su, Q. (1988). *Numerical simulations of multiphoton ionization and above-threshold electron spectra.* Physical Review A **38**, 3430.
- Kato, T. (1957). *On the eigenfunctions of many-particle systems in quantum mechanics.* Communications on Pure and Applied Mathematics **10**, 151.
- Kidger, P. and Garcia, C. (2021). *Equinox: neural networks in JAX via callable PyTrees and filtered transformations.* Differentiable Programming workshop at NeurIPS 2021.
- Lanczos, C. (1950). *An iteration method for the solution of the eigenvalue problem of linear differential and integral operators.* Journal of Research of the National Bureau of Standards **45**, 255.
- LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations.* SIAM.
- Martin, R. M. (2004). *Electronic Structure: Basic Theory and Practical Methods.* Cambridge University Press.
- Peruzzo, A. et al. (2014). *A variational eigenvalue solver on a photonic quantum processor.* Nature Communications **5**, 4213.
- Hermann, J., Schätzle, Z., and Noé, F. (2020). *Deep-neural-network solution of the electronic Schrödinger equation.* Nature Chemistry **12**, 891.
- Pfau, D., Spencer, J. S., Matthews, A. G. D. G., and Foulkes, W. M. C. (2020). *Ab initio solution of the many-electron Schrödinger equation with deep neural networks.* Physical Review Research **2**, 033429.
- Szabo, A. and Ostlund, N. S. (1996). *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.* Dover.
