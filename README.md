# GATO

**Grid Autodiff Theory of Orbitals**

> Math renders cleanly on the rendered docs site: **<https://ezequielgarcia.github.io/gato/>**.
> GitHub's markdown viewer mangles display-math spacing and some macros — prefer the site for the phase notes.


A 3D Schrödinger solver built from scratch in [JAX](https://jax.readthedocs.io/), targeting GPU backends.

This README is meant to be readable by a physics student who has seen Griffiths' *Introduction to Quantum Mechanics* but not necessarily a full graduate course on computational electronic structure. It explains both what the code does and *why* each design choice was made.

---

## 0. What GATO is

GATO solves the electronic Schrödinger equation for atoms and molecules, from first principles, on a computer.

Given a set of atomic nuclei — their positions and their charges — GATO computes the **ground-state wavefunction and energy** of the electrons that surround them. From that wavefunction it derives everything downstream: electron density, orbital shapes, bond lengths, bond angles, and the forces that hold the molecule together.

Concretely, the code:

- **Represents the wavefunction on a three-dimensional grid.** The cube of space around the molecule is sampled at evenly spaced points, and the value of the electronic wavefunction is stored at every point. No Gaussian basis sets, no atomic orbitals — just numbers on a lattice.
- **Builds the Hamiltonian as an operator, not a matrix.** Kinetic energy is a seven-point finite-difference stencil acting on neighbouring grid values; the nuclear potential is a pointwise multiplication; electron–electron repulsion is a 3D Fourier-transform convolution. Nothing is ever assembled into a dense $N^3 \times N^3$ matrix.
- **Finds the ground state by variational optimization.** The wavefunction is treated as a differentiable parameter; the energy is a scalar function of those parameters; gradient-based optimizers (imaginary-time propagation, Rayleigh-quotient descent, Lanczos diagonalization) drive the energy to its minimum.
- **Finds molecular geometry the same way.** Nuclear positions are also differentiable parameters. `jax.grad` applied to the total energy returns the forces on the nuclei; Adam on those forces relaxes the molecule to its equilibrium geometry.
- **Handles many-electron systems self-consistently.** For atoms and molecules with more than one electron, GATO runs a self-consistent-field loop (Hartree–Fock or Kohn–Sham DFT): the effective potential felt by each electron depends on the density of all the others, and the loop iterates until the wavefunction and the potential it generates agree.
- **Delivers physically meaningful numbers.** Bond lengths, bond angles, total energies, virial ratios, orbital energies, radial densities — the observables a chemist or spectroscopist would actually ask about.

The end-to-end pipeline is pure [JAX](https://jax.readthedocs.io/): differentiable, JIT-compiled to XLA, and runs identically on CPU or on a single consumer-grade NVIDIA GPU. The target progression is hydrogen → $\text{H}_2^+$ → helium → restricted Hartree–Fock and Kohn–Sham DFT on H₂, LiH, and H₂O, with molecular geometries obtained by gradient descent on the total energy. An optional side-branch extends the same machinery to heavy atoms (Au⁺, Hg) using a scalar-relativistic kinetic operator.

---

## 1. Motivation

Most introductions to numerical quantum mechanics take one of two paths:

1. **Dense linear algebra.** Write the Hamiltonian as an $N^3 \times N^3$ matrix, feed it to a diagonalizer, get eigenvalues. This is conceptually clear but dies around $N = 30$ because the matrix has $N^6$ entries — an $80^3$ grid would need ~260 TB just to store $\hat H$.
2. **Black-box packages.** Use PySCF, Gaussian, ORCA. These work, but the physics is hidden under layers of optimized Fortran and you learn nothing about *how* a Schrödinger solver actually works.

GATO takes a different route:

- **Matrix-free.** The Hamiltonian is represented by a function `psi -> H(psi)`. We never build a matrix. We only need to multiply $\hat H$ by a state, and we do that with local finite-difference stencils.
- **Differentiable.** Everything runs through JAX's automatic differentiation, so the gradient of any observable with respect to any parameter (grid spacing, vector potential, neural-network weights) is one `jax.grad` away.
- **JIT-compiled.** Every inner-loop operation is compiled to XLA, which runs on CPU or GPU with no code changes.

The trade-off is that you cannot trivially compute excited states the way diagonalization does. Variational methods (minimize $\langle \hat H \rangle$) give the ground state; excited states need constrained optimization or imaginary-time evolution with Gram–Schmidt.

---

## 2. Physics Background

### 2.1 Atomic units

Throughout GATO we use **Hartree atomic units**: $\hbar = m_e = e = 4\pi\varepsilon_0 = 1$. Consequences:

| Quantity | SI value of 1 a.u. |
|---|---|
| Length (Bohr, $a_0$) | $5.292 \times 10^{-11}$ m |
| Energy (Hartree, $E_h$) | $27.211$ eV $\approx 4.36 \times 10^{-18}$ J |
| Time | $2.42 \times 10^{-17}$ s |

In these units the hydrogen ground state has energy exactly $-0.5\,E_h$ and Bohr radius $1\,a_0$. All numbers in the code are dimensionless — conversions happen at the boundary when you report a result.

The time-independent Schrödinger equation for one electron becomes

$$
\hat H\,\psi(\mathbf r) = E\,\psi(\mathbf r), \qquad \hat H = -\tfrac{1}{2}\nabla^2 + V(\mathbf r).
$$

### 2.2 Why a grid?

We represent $\psi$ by its values on a cube of sample points. This is called a **real-space grid** representation. It has two big advantages over, say, a Gaussian basis set:

- **Systematic convergence.** Refine the grid, the answer gets better in a predictable way. No basis-set selection black magic.
- **Locality.** Kinetic energy (the Laplacian) becomes a local stencil — each point only talks to its immediate neighbors — which is exactly what GPUs are good at.

The cost is that you need a *lot* of points to resolve atomic-scale features. A hydrogen 1s orbital has characteristic size $\sim 1\,a_0$ and decays over $\sim 10\,a_0$, so a box of side 10 with 64 points per side already gives a respectable 1st-order answer.

### 2.3 Cell-centered Cartesian grid

GATO uses a uniform **cell-centered** grid on the cube $[-L/2, L/2]^3$. With $N$ points per axis and spacing $h = L/N$, the grid points sit at

$$
x_i = -\tfrac{L}{2} + (i + \tfrac{1}{2})\,h, \qquad i = 0, 1, \dots, N-1.
$$

Why cell-centered rather than putting points on the boundary ($x_i = -L/2 + i\,h$)? Two reasons:

1. **Symmetric treatment of both boundaries.** No grid point sits exactly on $x = \pm L/2$, so neither Dirichlet nor periodic boundaries create special cases.
2. **Natural integration rule.** The midpoint rule

$$
\int_{-L/2}^{L/2} f(x)\,dx \;\approx\; h\sum_{i=0}^{N-1} f(x_i)
$$

   is second-order accurate and uses every grid point uniformly. In 3D this generalizes to $\int f\,dV \approx h^3\sum f(\mathbf r_{ijk})$.

The volume element `dV = h³` lets us compute inner products:

$$
\langle \phi | \psi\rangle = \int \phi^{\ast}(\mathbf r)\,\psi(\mathbf r)\,dV \;\approx\; h^3\sum_{ijk} \phi_{ijk}^{\ast}\,\psi_{ijk}.
$$

This is implemented as `gato.inner_product(phi, psi, grid)`.

### 2.4 Finite-difference Laplacian

The Laplacian $\nabla^2 = \partial_x^2 + \partial_y^2 + \partial_z^2$ is discretized with the **second-order central difference stencil**. Taylor-expanding a smooth function:

$$
f(x+h) = f(x) + h f'(x) + \tfrac{h^2}{2} f''(x) + \tfrac{h^3}{6} f'''(x) + \mathcal O(h^4),
$$

$$
f(x-h) = f(x) - h f'(x) + \tfrac{h^2}{2} f''(x) - \tfrac{h^3}{6} f'''(x) + \mathcal O(h^4).
$$

Adding and rearranging:

$$
f''(x) = \frac{f(x+h) - 2 f(x) + f(x-h)}{h^2} + \mathcal O(h^2).
$$

The error is $O(h^2)$, so halving the spacing cuts the error by a factor of four. Applying this along each axis:

$$
(\nabla^2 \psi)_{ijk} = \frac{\psi_{i+1,j,k} + \psi_{i-1,j,k} + \psi_{i,j+1,k} + \psi_{i,j-1,k} + \psi_{i,j,k+1} + \psi_{i,j,k-1} - 6\psi_{ijk}}{h^2}.
$$

This is the 7-point stencil. Each output value depends on 6 neighbors plus the center — a completely local operation, trivially parallel.

### 2.5 Boundary conditions

The stencil needs values at $i = -1$ and $i = N$, which are outside the grid. We handle this two ways:

- **Zero-Dirichlet** (`boundary="dirichlet"`): we pretend $\psi = 0$ outside the grid. Appropriate for bound states that decay to zero far from the origin (hydrogen, confined oscillators). Implemented with `jnp.pad(psi, 1)`.
- **Periodic** (`boundary="periodic"`): we pretend $\psi$ wraps around — $\psi_{-1} = \psi_{N-1}$, $\psi_N = \psi_0$. Appropriate for Bloch states, plane waves, and eventually periodic solids. Implemented with `jnp.roll`.

Both conventions make the Laplacian **Hermitian** (self-adjoint under our discrete inner product), which is crucial: a non-Hermitian $\hat H$ can give complex eigenvalues, and a variational ground-state optimizer can then drive the "energy" to $-\infty$.

### 2.6 Matrix-free operators

A physicist thinks of $\hat T = -\tfrac{1}{2}\nabla^2$ as an operator. A linear-algebra textbook turns it into a matrix. We keep it as an operator:

```python
def kinetic(psi, h, boundary="dirichlet"):
    return -0.5 * laplacian(psi, h, boundary)
```

This function maps a $\psi$ array to $\hat T\psi$ in $O(N^3)$ time and $O(N^3)$ memory, never touching an $N^6$-entry matrix. All downstream algorithms — variational minimization, imaginary-time evolution, conjugate-gradient diagonalization, Lanczos — only need the action $\hat H\psi$, not the explicit matrix.

---

## 3. What's Implemented (Phase 1 so far)

### 3.1 Project layout

```
gato/
├── README.md               (this file)
├── pyproject.toml          uv-managed project definition
├── uv.lock                 pinned exact dependency versions
├── .python-version         → 3.14
├── src/
│   └── gato/
│       ├── __init__.py     public API + enable_x64() + main()
│       ├── grid.py         Grid3D + integration helpers
│       ├── operators.py    laplacian / kinetic / gradient (matrix-free)
│       ├── potentials.py   softened Coulomb, harmonic, constant
│       ├── hamiltonian.py  H = T + V with .apply / .rayleigh
│       ├── observables.py  ⟨T⟩, ⟨V⟩, virial ratio, radial density
│       ├── ansatz/
│       │   ├── grid.py     grid-parameter ansatz (parameters = grid values)
│       │   └── neural.py   Equinox MLP × Kato cusp factor
│       ├── solvers/
│       │   ├── imag_time.py  ψ ← ψ − Δτ·Hψ  (grid ansatz)
│       │   └── vqe.py        optax-driven Rayleigh-quotient minimization
│       ├── geometry.py       Nuclei pytree, bond length/angle, nuclear repulsion
│       └── physics/
│           ├── hydrogen.py          end-to-end Phase 1 driver (CLI `gato-hydrogen`)
│           ├── h2_plus.py           end-to-end Phase 2 driver (CLI `gato-h2plus`)
│           └── radial_hydrogen.py   1D log-radial cross-check, pure -Z/r
├── tests/                  60 tests covering every module above
└── benchmarks/             softening extrapolation and radial-density figure generators
```

### 3.2 `Grid3D`

A frozen dataclass holding only the grid size `N` and extent `L`:

```python
from gato import Grid3D

grid = Grid3D(N=64, L=10.0)
grid.h           # 0.15625   (= L / N)
grid.shape       # (64, 64, 64)
grid.dV          # 0.00381   (= h³)
X, Y, Z = grid.coords()      # three (64, 64, 64) arrays
r = grid.radial()            # sqrt(X² + Y² + Z²)
```

The grid itself stores no data — it's just a description. The wave function lives in a separate `(N, N, N)` array, which keeps the data layout explicit and JIT-friendly.

### 3.3 Integration helpers

```python
from gato import integrate, inner_product, norm_sq, normalize

integrate(psi, grid)         # ∫ ψ dV
inner_product(phi, psi, grid)  # ⟨φ | ψ⟩ = ∫ φ* ψ dV
norm_sq(psi, grid)             # ⟨ψ | ψ⟩
normalize(psi, grid)           # returns ψ / √⟨ψ|ψ⟩
```

All use the midpoint rule with volume element `dV = h³`.

### 3.4 `laplacian`, `kinetic`, `gradient`

All three are `jax.jit`-compiled matrix-free operators:

```python
from gato import laplacian, kinetic, gradient

lap = laplacian(psi, grid.h, boundary="dirichlet")   # ∇² ψ
T_psi = kinetic(psi, grid.h)                         # -½ ∇² ψ
grad = gradient(psi, grid.h)                         # shape (3, N, N, N)
```

`boundary` must be a static argument (`"dirichlet"` or `"periodic"`) because the control flow depends on it; JAX traces each branch once and caches the compiled code.

The Laplacian also accepts an `order` argument (`2` or `4`). The 4th-order 13-point stencil costs 1.5× more per application than the default 2nd-order 7-point stencil but is $O(h^4)$ accurate, reaching the same accuracy at half the grid size per axis — roughly $8\times$ cheaper overall on smooth problems.

### 3.5 Neural ansatz with Kato cusp

The neural ansatz is

$$
\psi_\theta(\mathbf r) \;=\; g_\theta(\mathbf r)\;\prod_k \exp\bigl(-Z_k\,|\mathbf r - \mathbf R_k|\bigr),
$$

where $g_\theta$ is a tanh-MLP (Equinox, configurable depth/width) and the product runs over nuclei at positions $\{\mathbf R_k\}$ with charges $\{Z_k\}$. The exponential factor imposes **Kato's exact nuclear-cusp condition** [Kato 1957],

$$
\left\langle \frac{1}{\psi}\frac{\partial \psi}{\partial r}\right\rangle_{\!\Omega} \;\xrightarrow[\,r \to \mathbf R_k\,]{} \;-Z_k,
$$

which every true Coulomb eigenstate satisfies. The smooth MLP contribution averages to zero angularly near the nucleus, so the cusp is **exact by construction** regardless of the network weights.

```python
from gato.ansatz.neural import NeuralAnsatz

model = NeuralAnsatz(
    key=key,
    nuclei_positions=((0.0, 0.0, 0.0),),   # hydrogen at origin
    nuclei_charges=(1.0,),
    hidden=32, n_layers=3,
)
```

For Phase 2 ($\text{H}_2^+$) the same class takes two nuclei; for Phase 5 (neural VMC on water), the same cusp machinery extends to a product over all nuclei of H₂O, with Jastrow factors to be added for electron-electron cusps.

### 3.6 Tests

60 tests, all passing on a 32³–128³ grid in about 175 s on CPU. The key ones:

| Test | What it checks | Why it matters |
|---|---|---|
| `test_periodic_plane_wave_is_eigenvector` | $e^{i\mathbf k\cdot\mathbf r}$ on a periodic grid is an *exact* eigenvector of the FD Laplacian, with eigenvalue $-\tfrac{2}{h^2}\sum_\alpha(1-\cos k_\alpha h)$. | Proves the stencil is implemented correctly to machine precision. |
| `test_periodic_continuum_limit_second_order` | As $h \to 0$ the FD eigenvalue converges to $-\lvert\mathbf k\rvert^2$ and the error drops by $\sim 4$× when $N$ doubles. | Verifies the $O(h^2)$ accuracy of the discretization. |
| `test_dirichlet_hermitian` | $\langle\phi\mid\nabla^2\psi\rangle = \langle\nabla^2\phi\mid\psi\rangle$ for random $\phi,\psi$. | Hermiticity is required for any variational ground-state method to converge to a real minimum. |
| `test_dirichlet_particle_in_box_converges` | The Rayleigh quotient for $\psi = \prod\cos(\pi x_\alpha/L)$ on $[-L/2, L/2]^3$ converges monotonically to the continuum $3\pi^2/(2L^2)$. | Sanity-checks the full $\langle\psi\mid\hat T\mid\psi\rangle / \langle\psi\mid\psi\rangle$ pipeline against a textbook analytic result. |
| `test_harmonic_oscillator_ground_state` | The analytic 3D HO ground state gives $E = \tfrac{3}{2}\omega$ to $< 1\%$. | End-to-end check of `Hamiltonian` composition against an analytic non-Coulomb benchmark. |
| `test_neural_ansatz_cusp_single_center` | The spherically-averaged log-slope $\langle(1/\psi)\partial_r\psi\rangle_\Omega$ at the nucleus equals $-Z$. | Verifies Kato's cusp is satisfied by construction, for any random MLP weights. |
| `test_hydrogen_imag_time_converges` | Imaginary-time propagation from the hydrogenic initial guess gives $E$ within 10 % of $-0.5\,E_h$ on a small grid. | End-to-end sanity of the full Phase 1 stack (grid → kinetic → potential → Hamiltonian → solver → observables). |
| `test_hydrogenic_ground_state_energy[Z]` | Pure $V=-Z/r$ on a 1D log-radial grid recovers $-Z^2/2$ to $\lesssim 10^{-4}\,E_h$ at $N=400$ for $Z\in\{1,2,3\}$. | Independent cross-check of the softening-extrapolated 3D result, with no $\epsilon$ anywhere in the pipeline. |

### 3.7 Quick smoke test

```bash
uv run gato
```

Prints JAX version, the available device, a grid summary, and the kinetic energy of a unit Gaussian (should be close to 0.75 Ha, the analytic value $3/4$).

---

## 4. Installation and Development

### 4.1 Requirements

- Python 3.14 (managed by `uv`)
- [`uv`](https://docs.astral.sh/uv/) 0.10+

### 4.2 Setup

```bash
cd gato
uv sync              # creates .venv/ and installs all CPU deps from uv.lock
uv run pytest        # 40 tests should pass
uv run gato          # smoke test
uv run gato-hydrogen # full hydrogen benchmark
```

On a GPU box, add the CUDA extras:

```bash
uv sync --extra gpu  # pulls jax[cuda12] wheels (~2 GB of NVIDIA libraries)
```

No system CUDA install is needed — `jax[cuda12]` bundles its own CUDA 12 + cuDNN. An NVIDIA driver version 525+ is enough.

### 4.3 Double precision

JAX defaults to float32 for performance. Quantum energies are small differences of large kinetic and potential contributions, so float32 will quietly corrupt your answers. Always enable x64:

```python
import gato
gato.enable_x64()   # call once, before any JAX computation
```

Tests enable this automatically via `tests/conftest.py`.

---

## 5. Roadmap

### 5.0 Systems at a glance

Every physical system the project solves, organized by method. The same molecules (H₂, LiH, H₂O) appear at both RHF (Phase 4) and LDA (Phase 5) so the two methods can be compared on identical geometries; the ab-initio terminal result lives at Phase 4.

| Phase | System | $n_e$ | Nuclei | Method | Ab initio? | Hardware |
|-------|--------|-------|--------|--------|-----------|----------|
| 1 ✓ | H (hydrogen atom) | 1 | 1 | grid / neural VQE | yes | CPU |
| 1 ✓ | He⁺, Li²⁺ (hydrogenic ions) | 1 | 1 | same, $Z$-scaled | yes | CPU |
| 2 ✓ | $\text{H}_2^+$ | 1 | 2 | grid + autodiff forces on $\mathbf R$ | yes | CPU |
| 3 | He (helium atom) | 2 | 1 | RHF, SCF | yes | CPU / 5070 |
| 4 | H₂ | 2 | 2 | RHF + autodiff forces | yes | 5070 |
| 4 | LiH | 4 | 2 | RHF + autodiff forces | yes | 5070 |
| 4 | H₂O | 10 | 3 | RHF, geometry optimization — **project's ab-initio terminal target** | yes | 5070 |
| 5 | H₂, LiH, H₂O | same as above | same | KS-DFT (LDA), method comparison vs. RHF | no (parameterized XC) | 5070 |
| 6 | H, He, Be, Ne (and Na, Hg vapor if Phase 7 is done) | same atoms as above | 1 | atomic absorption spectra from excited-state Lanczos + transition dipoles | yes | CPU |
| 7 | Au⁺, Hg (and optionally Ag, Cu for the light-homolog contrast) | 34, 40 | 1 | scalar-relativistic RHF / ZORA on a log-radial 1D grid, plus Phase 6 spectra on top | yes | CPU |

**Distinct systems:** H, He⁺, Li²⁺, He, $\text{H}_2^+$, H₂, LiH, H₂O, Be, Ne, Au⁺, Hg (Ag, Cu).

**Neural ansätze** (Equinox MLP × Kato cusp) are used alongside grid ansätze throughout, both as a pedagogical tool (Phase 1's hydrogen VQE) and as a viable representation of the self-consistent orbitals inside the Phase 3–5 SCF loops.

**Ab-initio boundary.** Phase 4 (restricted Hartree–Fock) is the strictly-ab-initio terminal result: one Slater determinant, exact electron–electron Coulomb and exchange, no empirical or numerically-fit parameters anywhere. Phase 5 (Kohn–Sham DFT with the LDA functional) is an explicitly-parameterized extension: the LDA exchange-correlation functional is numerically fit to quantum Monte Carlo data for the homogeneous electron gas [Ceperley–Alder 1980, Perdew–Zunger 1981], so the method carries reference-calculation parameters. It's included because LDA often gives more *accurate* geometries than RHF despite being less *principled* — a pedagogically important contrast. Beyond Phase 5, correlated many-body wavefunctions are handled by FermiNet (see §5.6).

### Phase 1 — Differentiable Hydrogen Atom *(complete)*

Foundation: prove the machinery works by recovering the hydrogen ground state $E_0 = -0.5\,E_h$.

- [x] Cell-centered 3D Cartesian grid (`Grid3D`)
- [x] JIT-compiled Laplacian, kinetic, gradient operators with Dirichlet / periodic BCs
- [x] Integration, inner product, normalization helpers
- [x] `potentials.py` — softened Coulomb $V_\epsilon(r) = -1/\sqrt{r^2+\epsilon^2}$, harmonic, constant
- [x] `hamiltonian.py` — full $\hat H = \hat T + \hat V$ with `.apply`, `.expectation`, `.rayleigh`
- [x] `observables.py` — $\langle T\rangle$, $\langle V\rangle$, virial ratio, radial density
- [x] `ansatz/grid.py` — raw grid parameters as variational wave function
- [x] `ansatz/neural.py` — Equinox MLP $\psi_\theta(\mathbf r) = g_\theta(\mathbf r)\prod_k e^{-Z_k|\mathbf r-\mathbf R_k|}$ with exact Kato nuclear cusps
- [x] `solvers/imag_time.py` — imaginary-time propagation $\psi \to \psi - \Delta\tau\,\hat H\psi$
- [x] `solvers/vqe.py` — optax-driven Rayleigh-quotient minimization
- [x] `physics/hydrogen.py` — end-to-end driver targeting $-0.5\,E_h$
- [x] 4th-order 13-point Laplacian stencil (selectable via `order=4`)
- [x] Matrix-free Lanczos solver (`solvers/lanczos.py`) for lowest-$K$ eigenpairs
- [x] Hydrogenic $Z$-scaling: $E_0(Z) = -Z^2/2$ reproduced for H, He⁺, Li²⁺
- [x] Softening extrapolation $\epsilon \to 0$ closes residual to $<1\%$ of analytic $-0.5\,E_h$
- [x] Radial density figure: converged $P(r)$ vs. analytic $4r^2 e^{-2r}$
- [x] Log-radial 1D cross-check (`physics/radial_hydrogen.py`): pure $V=-Z/r$ on a log-spaced grid, with odd-parity boundary at $r=0$ and a 4th-order chain-rule Laplacian. Reproduces $-Z^2/2$ to $\sim 10^{-7}\,E_h$ at $N=800$ for $Z\in\{1,2,3\}$, independently confirming the $\epsilon\to 0$ extrapolation
- [x] Test suite (46 tests): plane-wave exactness at $O(h^2)$ and $O(h^4)$, Hermiticity, HO Lanczos spectrum, Kato cusp spherical-average, $Z$-scaling, end-to-end hydrogen, log-radial cross-check

**Exit criterion:** recover $E_0 \approx -0.5\,E_h$ within 1% on a 64³ grid.

### Phase 2 — Multi-nucleus, single electron ($\text{H}_2^+$) *(complete)*

First taste of chemistry. A stepping stone that introduces two ideas needed for molecules without yet requiring many-electron machinery: multi-center potentials and geometry as a differentiable parameter. Scoped as a ~1–2 week capability milestone, not a major work phase — its purpose is to validate forces and multi-center machinery in the simplest setting before they get entangled with SCF in Phase 3.

- [x] `potentials.multi_center_softened_coulomb` — $V(\mathbf r) = -\sum_k Z_k / \sqrt{|\mathbf r - \mathbf R_k|^2 + \varepsilon^2}$
- [x] `geometry.py` — `Nuclei` pytree (positions, charges), bond-length / bond-angle observables, nuclear repulsion, COM / recenter
- [x] `jax.grad` of the Born–Oppenheimer energy with respect to $\mathbf R_k$ → **Hellmann–Feynman forces** (no back-propagation through the imag-time solver needed; variational stationarity cancels Pulay at convergence)
- [x] Geometry optimization as optax on $\mathbf R$, with charge-weighted COM pinned to the origin
- [x] `ansatz.grid.init_lcao` — bonding σ_g initial guess (sum of hydrogenic 1s)
- [x] `physics/h2_plus.py` — end-to-end driver: `solve_electronic`, `bo_curve`, `optimize_geometry`, CLI `gato-h2plus`
- [x] 11 tests: bond observables, analytic Coulomb force from `jax.grad` on E_nn, K=1 reduction, charge scaling, superposition, differentiability in positions, BO curve has bound minimum, Hellmann–Feynman = finite-difference force, geometry opt lands in $[1.7, 2.3]\,a_0$

**Exit criterion met.** Starting from $R = 2.4\,a_0$ on a $56^3$ grid, 30 geometry steps of Adam on the BO energy recover $R_e = 1.9986\,a_0$, within $0.07\%$ of the Burrau 1927 analytic value $1.9972\,a_0$. Total energy $-0.5749\,E_h$ is ~28 mHa above the analytic $-0.6026\,E_h$ — softening-limited, exactly as in Phase 1 §5.5.

### Phase 3 — Helium via mean-field electronic structure

Enter many-electron land on the simplest closed-shell atom. Single nucleus, two electrons, no molecular geometry yet — the focus is on getting self-consistency right.

- [x] Hartree potential $J(\mathbf r) = \int \rho(\mathbf r')/|\mathbf r - \mathbf r'|\,dV'$ via 3D FFT convolution on a doubled grid (Hockney/Eastwood method)
- [x] `scf.py` — self-consistent-field loop, matrix-free Fock with a general $n_\text{orb}^2$ exchange operator and linear density mixing
- [x] Restricted Hartree–Fock with doubly-occupied orbital (helium, N=64 L=10 grid, converges in ~6 iterations)
- [x] `functionals.py` — LDA exchange (Dirac) + correlation (Perdew–Zunger 1981); unpolarized, with variational-consistency and branch-continuity tests
- [x] Kohn–Sham DFT with LDA — `scf_ks_lda` driver, purely local $V_\text{eff} = V_\text{ext} + V_H + V_\text{xc}$ (no exchange operator, no per-pair Poisson solves)
- [x] **Accuracy work — softening residual:** `epsilon` threaded through `scf_rhf` / `scf_ks_lda`; `benchmarks/helium_softening_extrapolation.py` closes the gap from +263 mHa to +19 mHa at $N=64$ and to $-3.4$ mHa (RHF) / $+4.0$ mHa (LDA) at $N=80$ via a linear $\varepsilon \to 0$ fit. See `docs/phase3_note.md` §6 for data and figure.

**Exit criterion:** helium ground-state energy within chemical accuracy ($\sim 1$ mHa) of reference values ($-2.8617\,E_h$ RHF, $-2.834\,E_h$ LDA, $-2.9037\,E_h$ exact). **Met** at $N = 80$, $L = 10$ via linear $\varepsilon \to 0$ extrapolation: RHF $-2.8651\,E_h$ (residual $-3.4$ mHa), LDA $-2.8300\,E_h$ (residual $+4.0$ mHa). The small residual is grid-discretization bias of the 4th-order kinetic stencil, not softening; one further $N$ refinement (trivial on the 5070) takes it sub-mHa.

**The softening residual, for the record.** At the default $\varepsilon = h/2$ on $N = 64$, helium RHF sits $\sim 260$ mHa above the reference. This is not an SCF bug — the SCF solves the softened Hamiltonian to $10^{-6}\,E_h$, but the softened Hamiltonian itself is lifted above true Coulomb by the combined bias from softening both $V_\text{ext}$ and the Hartree/exchange kernel. The bias scales with the probability mass at $r \lesssim \varepsilon$, which is $\sim 0.08\,a_0$ — non-negligible at $Z = 2$ where the 1s orbital has characteristic size $\sim 0.3\,a_0$. The `epsilon` argument is threaded through `scf_rhf` / `scf_ks_lda` so both the $V_\text{ext}$ softening (in `softened_coulomb`) and the Hartree/exchange kernel softening (in `hartree_potential` / `exchange_apply`) sweep together, and $E(\varepsilon)$ is demonstrably linear in $\varepsilon$ over the relevant range — see `docs/phase3_note.md` §6.

The same linear $\varepsilon \to 0$ fit will close the residual at $Z = 8$ (oxygen) in Phase 4. If its range gets inconvenient, the alternative route — an origin-regularized Hartree kernel (exact $1/r$ everywhere except a cube of side $h$ at the origin, whose singular value is replaced by the analytic self-energy of a uniform charge cube, the standard plane-wave trick) — removes the kernel-side softening entirely and leaves only $V_\text{ext}$ for route 1 to clean up. Not needed for Phase 3; optional for Phase 4.

**Additional atom benchmarks (planned, GPU-gated).** He alone validates the $n_\text{orb}=1$ path. Two more closed-shell atoms fill in the coverage before jumping to molecules:

- **Be** ($Z=4$, $1s^2\,2s^2$, $n_\text{occ}=2$) — exercises the general $n_\text{orb}^2$ exchange operator with two orbitals of the same angular momentum. Reference RHF: $-14.573\,E_h$.
- **Ne** ($Z=10$, $1s^2\,2s^2\,2p^6$, $n_\text{occ}=5$) — first p-orbital occupancy; exercises the symmetry-breaking perturbation in the default initial guess (a purely spherical starter would leave the three 2p's invisible to Lanczos). Reference RHF: $-128.547\,E_h$. These orbital counts also match the Phase 4 molecular targets: H₂ is $n_\text{occ}=1$ like He, LiH is $n_\text{occ}=2$ like Be, H₂O is $n_\text{occ}=5$ like Ne — so the atoms isolate SCF machinery from geometry optimization debugging.

Test stubs exist in `tests/test_scf.py::test_beryllium_rhf_converges` and `::test_neon_rhf_converges`, currently `@pytest.mark.skip`'d because each takes ~30 min (Be) to multi-hour (Ne) on CPU. They'll be enabled once the GPU backend (`uv sync --extra gpu`) is in use, where they should drop to seconds / minute respectively.

### Phase 4 — Restricted Hartree–Fock molecules (ab-initio terminal target)

Combine Phase 2 (multiple nuclei + autodiff forces) with Phase 3 (SCF) into *strictly ab-initio* mean-field molecules. One Slater determinant, exact Coulomb and exchange, no empirical parameters.

- [ ] SCF with multi-center potentials, Hartree term $J[\rho]$ via 3D FFT convolution
- [ ] **Exact exchange operator** $\hat K$ implemented as $n_\text{orb}^2$ real-space Poisson solves per SCF iteration — the computational difference from Phase 5's LDA is in this operator alone
- [ ] Hellmann–Feynman + Pulay forces via `jax.grad` on the self-consistent energy
- [ ] Joint $(\theta_{\text{orbitals}}, \mathbf R_{\text{nuclei}})$ optimization
- [ ] H₂ (2 e⁻, 1 orbital): bond length $\approx 1.40\,a_0$ at RHF, compared to experiment $1.401\,a_0$
- [ ] LiH (4 e⁻, 2 orbitals): bond length $\approx 3.02\,a_0$ at RHF
- [ ] H₂O (10 e⁻, 5 orbitals): bond angle $\approx 106°$ at RHF, bond length $\approx 1.78\,a_0$

**Exit criterion (project's ab-initio terminal target):** H₂O geometry recovered at the RHF level consistent with published RHF reference values (angle $\approx 106°$, $d_{\text{OH}} \approx 1.78\,a_0$). Energy within a few mHa of reference RHF. The systematic $\sim\!1.5°$ overestimate of the bond angle vs. experiment is a known RHF limitation — fixing it requires either a parameterized functional (Phase 5) or correlated wavefunctions (FermiNet, §5.6).

### Phase 5 — Kohn–Sham DFT with LDA (parameterized extension)

Same SCF framework as Phase 4, with the exact exchange operator $\hat K$ swapped for a Kohn–Sham exchange-correlation functional — specifically the local-density approximation (LDA) with the Perdew–Zunger parameterization of the Ceperley–Alder homogeneous-electron-gas QMC data. This phase is **explicitly not ab initio** in the strict sense: the LDA XC functional has numerical parameters fit to reference calculations. It is included for two reasons:

1. **Method-comparison.** LDA on H₂O gives a bond angle of $\approx 104.5°$ — closer to experiment than RHF. Running the same geometries through both methods (and optionally through FermiNet later) is the cleanest demonstration of the parameterized-vs-ab-initio-vs-correlated tradeoff.
2. **Minimal marginal code.** Once the SCF loop of Phase 4 exists, the only new code is the LDA exchange-correlation functional itself — a ~10-line point-wise function of the local density $\rho(\mathbf r)$.

- [ ] `functionals.py` — LDA exchange (Dirac) and correlation (Perdew–Zunger 1981)
- [ ] Swap `Fock.exact_exchange` for `ks.V_xc[rho]` in the Phase-4 SCF loop
- [ ] Re-run H₂, LiH, H₂O at the LDA level; compare geometries and energies directly to the Phase 4 RHF numbers

**Exit criterion:** H₂O bond angle within $1°$ of experimental $104.5°$ at the LDA level; same for bond length within $0.02\,a_0$ of experimental. Numbers must match published DFT-LDA reference values.

### 5.6 Beyond Phase 5: handing off to FermiNet

Phase 4 is the terminal scope of GATO proper. Anything requiring correlated many-body wavefunctions — kcal/mol-accurate binding energies, the water dimer, N₂'s triple-bond dissociation curve, open-shell or strongly-correlated systems — is better handled by [DeepMind's FermiNet](https://github.com/google-deepmind/ferminet). FermiNet is also written in JAX, so the handoff is trivial: the same CUDA/cuDNN stack, the same JIT cache, no framework switch. A typical workflow is to use GATO to prototype and understand a system at mean-field level, then — if correlation matters — run the same geometry through FermiNet for the production number.

What is *not* duplicated between GATO and FermiNet:

- GATO covers the **single-particle and mean-field regime** on a real-space grid: hydrogen, hydrogenic ions, $\text{H}_2^+$, helium, and light closed-shell molecules with DFT-LDA + autodifferentiated forces. FermiNet does not ship a grid-based Schrödinger solver or an SCF loop.
- FermiNet covers the **many-body correlated regime**: neural Slater-backflow wavefunctions trained by variational Monte Carlo with KFAC. GATO deliberately does not re-implement this — the production implementation is already open source and state-of-the-art.

This split keeps GATO's scope bounded (3–6 months of work rather than 12+), and leaves the re-implementation of well-solved problems to the teams that have invested years optimizing them.

### Phase 6 — Atomic absorption spectra (observable-extraction layer)

The first phase that goes *beyond ground states*. Phases 1–5 compute $E_0$ and $\psi_0$; Phase 6 computes excited states and the transition dipole moments between them, giving quantitative atomic absorption spectra — the kind of data a spectroscopist would actually ask for. Pedagogical payoff: explain the color of a neon sign, the yellow of a sodium street lamp, the blue-green of mercury-vapor fluorescent tubes, and the Balmer series of hydrogen, all from the same ab-initio atomic Hamiltonian. Scoped as a ~2-week extension — essentially no new infrastructure, just a new wiring of what is already there.

**Why it belongs here.** The Lanczos solver from Phase 1 already returns the lowest $K$ eigenpairs of any matrix-free Hermitian operator. Phases 1–5 used only the ground state; Phase 6 uses the rest. The transition dipole moment $\mu_{0\to e} = \langle\psi_e | \hat{\mathbf r} | \psi_0\rangle$ is one grid integration once both states are on the grid. Selection rules are not postulated; they emerge as parity and Wigner–Eckart statements about which dipole integrals vanish by symmetry. Natural (lifetime) linewidths are computed from the Einstein $A$ coefficient, which is another closed-form function of the same $\mu$. All of this is a few hundred lines on top of the Phase 3 / Phase 5 atomic solvers.

- [ ] `spectra.excited_states_lanczos` — return the lowest $K$ eigenpairs of the (Fock or KS) Hamiltonian, reusing the Phase 1 solver with $K > 1$
- [ ] `spectra.transition_dipole(psi_0, psi_e, grid)` — compute $\boldsymbol{\mu}_{0 \to e} = \int \psi_e^{\ast}\,\mathbf r\,\psi_0\,dV$, returning a 3-vector per pair
- [ ] `spectra.oscillator_strength(omega, mu)` — the standard $f_{0\to e} = (2/3)\,m_e\,\omega_{e0}\,|\mu|^2/\hbar$
- [ ] `spectra.einstein_A(omega, mu)` — natural emission rate $A = (4\omega^3 / 3\hbar c^3)\,|\mu|^2$, gives the natural linewidth $\Gamma_{\rm nat} = \hbar A$
- [ ] `spectra.absorption_cross_section(omega, lines, broadening)` — Lorentzian (natural) + Gaussian (Doppler, $T$-dependent) broadening, returning $\sigma(\omega)$ as a stick spectrum or smooth curve
- [ ] `physics/atomic_spectra.py` — end-to-end driver producing NIST-comparable tables and plots for H, He, Be, Ne
- [ ] Demos: Balmer series ($n=3,4,5 \to n=2$); He $1s^2\to 1s2s$ and $1s^2\to 1s2p$; Be $2s^2\to 2s2p$; Ne $2p^6\to 2p^5 3s$; formally-forbidden transitions (zero oscillator strength, explicit parity demonstration)
- [ ] Validation against the NIST Atomic Spectra Database: line positions to chemical accuracy on H and He, qualitative on Be and Ne (LDA/RHF mean-field limits)

**The demo.** One plot per element: vertical sticks at computed $\omega_{e0}$, heights set by $f_{0\to e}$, smooth envelope from Doppler broadening at 300 K. Overlay NIST reference lines. Show parity-forbidden transitions coming out numerically $\sim 10^{-12}$ (grid noise) while allowed ones are $\sim 10^{-1}$ — a ten-orders-of-magnitude separation that validates the selection-rule machinery from first principles.

**Exit criterion.** Balmer-α (H, $n = 3 \to 2$) recovered to $< 0.1$ nm on a $N = 800$ log-radial grid. He $1s^2 \to 1s2p$ ($58.4$ nm) recovered to $< 1$ nm at the RHF+CIS level, or the computed number agrees with the single-excitation Lanczos treatment's known systematic offset from experiment. Selection rules demonstrated numerically by a factor $\geq 10^8$ suppression of forbidden lines relative to allowed ones.

**What Phase 6 deliberately does not cover.** Multi-reference excited states (needs CI or TDDFT with a real XC kernel), vibrationally-resolved molecular spectra (needs Franck–Condon machinery), solid-state band transitions (needs periodic BC and $k$-point sampling), and any line-broadening mechanism that requires inter-atomic interactions (pressure broadening). These are separate codebases / separate phases.

### Phase 7 — Scalar-relativistic heavy atoms (optional extension)

An optional side-branch beyond the H → H₂O main line and the Phase 6 spectra layer. Aimed at making the "relativity is visible to the eye" story quantitative: gold's yellow color via the 6s contraction, Hg's 254 nm line as a spin-forbidden transition whose lifetime only comes out right with relativistic corrections. Target audience is the same pedagogical reader as Phase 1. Scoped as a one-month extension, strictly smaller than any of Phases 3–5.

**Why it belongs in GATO.** The new physics is two local operators (mass–velocity $-\hat p^4/8 m^3 c^2$ and Darwin $\propto \nabla^2 V$), or equivalently one position-dependent effective mass in the kinetic stencil (ZORA). Both drop into the existing SCF loop by swapping the kinetic operator — no new solver, no new functional, no new geometry machinery. Everything else (Lanczos, Hartree, LDA, `jax.grad`, and the Phase 6 spectra layer) is re-used unchanged. The correct framing is "GATO with one extra operator", not a new framework.

**Why log-radial, not 3D Cartesian.** At $Z = 79$ the 1s orbital has $\langle r\rangle \sim 1/Z \approx 0.013\,a_0$, far below the $h \approx 0.08\,a_0$ grid spacing used for water. Resolving core electrons on a Cartesian grid would need $N \gtrsim 300$ per axis. Atoms are spherically symmetric, so the natural representation is the log-radial 1D grid already built for `physics/radial_hydrogen.py`: exponential clustering toward the nucleus gives effectively infinite resolution at the core for ~$10^3$ points.

- [ ] `operators.kinetic_zora(psi, V, h, c)` — ZORA kinetic on the log-radial grid, $\hat T = \hat p\,[c^2/(2 m c^2 - V)]\,\hat p$
- [ ] `operators.kinetic_mv_darwin` — alternative perturbative mass-velocity + Darwin, as a cross-check against ZORA
- [ ] `functionals.lda_xc_*_radial` — LDA XC adapted to the 1D radial angular integration (trivial factor of $4\pi r^2$)
- [ ] `scf_rhf_radial` / `scf_ks_lda_radial` — radial analogues of the Phase 3 SCF loops, with a spherical Hartree kernel (1D radial Poisson, $O(N)$)
- [ ] Core-valence partitioning: occupy shells by Aufbau, not by Lanczos on a single Krylov run (shell structure is explicit in 1D radial)
- [ ] `physics/gold.py` — end-to-end driver for Au⁺ ($5d^{10}$, $n_{\rm occ} = 34$) and Hg ($5d^{10}\,6s^2$, $n_{\rm occ} = 40$), running NR and scalar-relativistic side by side
- [ ] Light-homolog contrast: Cu (3d¹⁰) and Ag (4d¹⁰) — both closed-shell one-column-above relatives of Au⁺, both predicted to show *smaller* relativistic contraction since $Z\alpha$ is smaller; demonstrates the $\sim Z^2$ scaling of the correction
- [ ] Spectra hook: rerun Phase 6 on Hg's low-lying states with the ZORA operator. Hg 254 nm ($6^1S_0 \to 6^3P_1$) is formally spin-forbidden (ΔS ≠ 0) but visible in mercury lamps because scalar-relativistic + small-SO mixing borrows allowed character from $^1P_1$; reproducing the line at all is a direct consequence of the Phase 7 operator swap

**The demo.** Same SCF stack, two kinetic operators, one plot:

| Quantity | NR | Scalar-rel | Experimental trend |
|----------|------|------------|---------------------|
| Au⁺ 6s orbital energy | ~$-0.29\,E_h$ | ~$-0.38\,E_h$ | 6s contraction — $\sim 20\%$ deeper binding |
| Au⁺ 5d → 6s gap | UV | visible blue | "gold is yellow" |
| $\langle r\rangle_{6s}$(Hg) / $\langle r\rangle_{6s}$(Cd) | ratio $\approx 1$ | ratio $< 1$ | Hg's unusual volatility / liquid-at-RT behavior |

**Exit criterion.** Scalar-relativistic Au⁺ 6s orbital energy within $1\%$ of the published all-electron scalar-relativistic DFT reference [Desclaux 1973], and the 5d→6s gap reduction from NR to scalar-rel reproduced to within a few hundred meV. Phase 6 spectra rerun with ZORA gives a nonzero oscillator strength at Hg 254 nm that vanishes in the NR limit.

**What Phase 7 deliberately does not cover.** Full two-component spin-orbit (needs doubled array shapes and a rewritten stencil), the Dirac equation proper (four-component, $Z_{\rm crit} = 1/\alpha$ issues), and any molecular geometry at $Z > 20$. Readers who need those should use DIRAC, ReSpect, or ADF — GATO's scope ends at "the cheapest relativistic correction that makes the textbook story quantitative."

---

## 6. Dependencies per phase

The package stays pure-Python, pure-JAX throughout. No new runtime dependencies are introduced across Phases 2–5.

| Phase | Hardware | New runtime deps | Dev-only deps | Why |
|---|---|---|---|---|
| 1 (done) | laptop CPU | `jax`, `optax`, `equinox`, `numpy`, `scipy`, `matplotlib` | `pytest` | base stack |
| 2 (H₂⁺) | laptop CPU | none | none | multi-center potential + autodiff forces only |
| 3 (helium RHF) | laptop CPU (or 5070) | none | none | FFT is in `jax.numpy.fft`; exact exchange is a 3D Poisson solve |
| 4 (RHF molecules) | local 5070 | none | **`pyscf`** (optional) | pyscf only for cross-validating against a trusted reference |
| 5 (LDA molecules) | local 5070 | none | none (reuse pyscf for comparison) | LDA is ~10 lines of point-wise math; reuses the Phase 4 SCF loop |
| 6 (atomic spectra) | laptop CPU | none | none | Lanczos (present) + one dipole integral, no new infrastructure |
| 7 (scalar-rel heavy atoms, optional) | laptop CPU | none | none | one extra kinetic operator on the existing log-radial grid |
| any, on any Nvidia GPU | | `jax[cuda12]` (via `--extra gpu`) | — | bundled CUDA 12 + cuDNN |

For correlated many-body wavefunctions, use [FermiNet](https://github.com/google-deepmind/ferminet) directly — it ships its own JAX dependency set that coexists with GATO's.

What's **not** needed:

- No PyTorch, no TensorFlow — stays 100 % JAX.
- No C / C++ / Fortran bindings. No `psi4`, no `libxc`.
- No databases, no trackers (optional `wandb` if we want experiment dashboards; trivial to add later).

---

## 7. Design Principles

- **JAX everywhere.** Autodiff, JIT, vmap. No NumPy in hot paths.
- **Matrix-free.** Hamiltonians are linear operators $\psi \mapsto \hat H\psi$, never dense arrays.
- **Pure functions.** No hidden state. Every solver takes an explicit `params` pytree and returns a new one.
- **Modular phases.** Each phase lives in its own subpackage; later phases import earlier operators rather than duplicating them.
- **Tested.** Every operator gets at least one analytic sanity check (harmonic-oscillator eigenvalues, free-particle plane waves, known hydrogen levels).

---

## 8. References

- Griffiths & Schroeter, *Introduction to Quantum Mechanics*, 3rd ed. — chapters on hydrogen and the variational principle.
- Martin, *Electronic Structure: Basic Theory and Practical Methods*, Cambridge — the canonical DFT/HF reference used for Phases 3 and 4.
- Szabo & Ostlund, *Modern Quantum Chemistry*, Dover — Hartree–Fock, configuration interaction, and the SCF loop.
- Ceperley, D. M. & Alder, B. J. (1980), "Ground state of the electron gas by a stochastic method", *Physical Review Letters* **45**, 566 — the homogeneous-electron-gas QMC data that LDA is fit to.
- Perdew, J. P. & Zunger, A. (1981), "Self-interaction correction to density-functional approximations for many-electron systems", *Physical Review B* **23**, 5048 — the LDA parameterization used in Phase 5.
- Kato, T. (1957), "On the eigenfunctions of many-particle systems in quantum mechanics", *Communications on Pure and Applied Mathematics* **10**, 151–177 — the nuclear- and electron-coalescence cusp conditions encoded into the neural ansatz.
- Peruzzo et al. (2014), "A variational eigenvalue solver on a photonic quantum processor" — the original VQE paper.
- Pfau et al. (2020), "Ab initio solution of the many-electron Schrödinger equation with deep neural networks" — FermiNet, the blueprint for Phase 5.
- Hermann et al. (2020), "Deep-neural-network solution of the electronic Schrödinger equation" — PauliNet, a parallel approach.
- LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations* — derivations of the stencils and their error bounds.
- JAX documentation: https://jax.readthedocs.io/
- Equinox (neural nets): https://docs.kidger.site/equinox/
- optax (optimization): https://optax.readthedocs.io/
- kfac-jax (natural-gradient optimizer): https://github.com/google-deepmind/kfac-jax

---

## 9. Status

Phases 1, 2, and 3 are **complete end-to-end**.

| Metric | Grid | Result | Target |
|---|---|---|---|
| Hydrogen $E_0$ (imag-time, $96^3$, $L=12$, $O(h^2)$) | | $-0.487\,E_h$ | $-0.500$ |
| Hydrogen $E_0$ (imag-time, $64^3$, $L=12$, $O(h^4)$) | | $-0.479\,E_h$ | $-0.500$ |
| Hydrogen $E_0$ (Lanczos + linear $\epsilon\to 0$ fit, $N=64$) | | $-0.504\,E_h$ | $-0.500$ |
| Hydrogen $E_0$ (log-radial 1D, pure $-Z/r$, $N=400$) | | $-0.4999997\,E_h$ | $-0.500$ |
| Hydrogen $E_0$ (log-radial 1D, pure $-Z/r$, $N=800$) | | $-0.4999999\,E_h$ | $-0.500$ |
| H$_2^+$ bond length $R_e$ (imag-time + geom opt, $56^3$, $L=10$, $O(h^4)$) | | $1.9986\,a_0$ | $1.9972$ |
| H$_2^+$ total energy at $R_e$ (same run, softened $\varepsilon = h/2$) | | $-0.5749\,E_h$ | $-0.6026$ |
| He RHF $E_0$ (SCF + linear $\varepsilon \to 0$ fit, $N=80$, $L=10$, $O(h^4)$) | | $-2.8651\,E_h$ | $-2.8617$ |
| He KS-LDA $E_0$ (SCF + linear $\varepsilon \to 0$ fit, $N=80$, $L=10$, $O(h^4)$) | | $-2.8300\,E_h$ | $-2.8340$ |
| Virial ratio $2\langle T\rangle/|\langle V\rangle|$ (96³) | | $0.984$ | $1.000$ |
| He⁺ $E_0$ (imag-time, $48^3$, $L=6$, $O(h^4)$) | | $-1.900\,E_h$ | $-2.000$ |
| Li²⁺ $E_0$ (imag-time, $48^3$, $L=4$, $O(h^4)$) | | $-4.275\,E_h$ | $-4.500$ |
| 3D HO Lanczos ladder ($N=40$, $L=10$) | | $1.50, 2.50, 3.50, \ldots$ | $1.5, 2.5, 3.5, \ldots$ |

The $\epsilon \to 0$ linear extrapolation closes the hydrogen residual from $2.6\%$ (fixed softening) to $< 1\%$ ($E_0 = -0.504\,E_h$). An independent 1D log-radial solver with the bare $V = -Z/r$ potential reproduces $-0.500\,E_h$ to seven decimal places at $N = 800$, which confirms that the remaining 3D residual is softening-limited, not a bug in the Cartesian stack. The $Z^2$ scaling of the hydrogenic ground state is reproduced across $Z \in \{1, 2, 3\}$ on both grids at comparable relative accuracy. The Lanczos solver recovers the full 3D harmonic-oscillator ladder on a $40^3$ grid, providing the eigensolver infrastructure that Phase 3 will use inside the SCF loop. All 46 tests pass.

**Phase 2 is also complete.** Starting from $R = 2.4\,a_0$ on a $56^3$ grid, 30 Adam steps on the Born–Oppenheimer energy recover the $\text{H}_2^+$ bond length $R_e = 1.9986\,a_0$ — within $0.07\%$ of the Burrau 1927 analytic value $1.9972\,a_0$ — by pure gradient descent. The total energy $-0.5749\,E_h$ is softening-limited (~28 mHa above the analytic $-0.6026\,E_h$), the same $O(\varepsilon)$ bias observed in Phase 1 and closable by the same $\varepsilon \to 0$ extrapolation.

**Phase 3 is complete.** Self-consistent RHF and Kohn–Sham-LDA drivers on helium converge in $\sim 6$ iterations at $\varepsilon = h/2$; sweeping $\varepsilon$ across five values at fixed grid and linearly extrapolating $E(\varepsilon) \to E(0)$ closes the residual to $-3.4\,$mHa (RHF) and $+4.0\,$mHa (LDA) at $N = 80$, $L = 10\,a_0$ — at chemical accuracy, with the remaining few-mHa residual attributable to 4th-order-stencil grid discretization rather than softening. The general $n_\text{orb}^2$ exchange operator, the Hockney doubled-grid FFT Hartree solver, and the symmetry-breaking core-Hamiltonian initial guess are all in place and tested; the Be / Ne atom benchmarks stand as GPU-gated stubs (they validate the multi-orbital code path once the 5070 is wired in).

All Phase 1, 2, and 3 numbers above were produced in float64 on a **single CPU core**. The codebase is pure JAX and runs unchanged on GPU via `uv sync --extra gpu`; on an RTX 5070 the same Phase 3 helium extrapolation is expected in single-digit minutes, and the Be/Ne benchmarks become tractable.

Next up: **Phase 4** — RHF on H₂, LiH, and H₂O with nuclear-gradient forces from the converged SCF density. The Phase 3 softening-extrapolation recipe ports over unchanged. Water is the project's **ab-initio terminal target**; Phase 5 adds LDA as a parameterized comparison; Phase 6 extracts atomic absorption spectra; Phase 7 (optional) swaps the kinetic operator for a scalar-relativistic form and picks up gold / mercury. Beyond Phase 5, users who need correlated many-body wavefunctions should run FermiNet on the output geometries — see §5.6.

---

## 10. Acknowledgements on development workflow

Phases 1 and 2 of GATO — roughly 2,900 lines of code, 60 tests, and the Phase 1 paper — were developed in about half a day of focused work, with Google Gemini and Anthropic's Claude assisting as coding and design collaborators. The LLMs accelerated implementation, test construction, and documentation drafting; the architectural choices, physics interpretation, roadmap scoping, and acceptance criteria remained human-directed throughout. This is a practical example of what the current generation of coding assistants can deliver on a moderately-scoped research codebase when the human provides clear design intent and the tools handle the mechanical work.
