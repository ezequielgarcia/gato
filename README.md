# GATO

**Grid Autodiff Theory of Orbitals**

A 3D Schrödinger solver built from scratch in [JAX](https://jax.readthedocs.io/), targeting GPU backends. The long-term trajectory goes hydrogen → $\text{H}_2^+$ → helium → mean-field molecules → a neural many-body wavefunction for H₂, LiH, HF, and H₂O → culminating in the water dimer $(\text{H}_2\text{O})_2$, the smallest piece of condensed water. Development and Phase 1–5 production run on a single consumer GPU (RTX 5070); Phase 6 (N₂, water dimer) bursts to a rented A100 for final training runs. The whole pipeline stays differentiable, matrix-free, and memory-efficient throughout, with molecular geometry obtained from first principles by gradient descent on the energy.

This README is meant to be readable by a physics student who has seen Griffiths' *Introduction to Quantum Mechanics* but not necessarily a full graduate course on computational electronic structure. It explains both what the code does and *why* each design choice was made.

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
\langle \phi | \psi\rangle = \int \phi^*(\mathbf r)\psi(\mathbf r)\,dV \;\approx\; h^3\sum_{ijk} \phi_{ijk}^*\,\psi_{ijk}.
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
│       └── physics/
│           └── hydrogen.py   end-to-end Phase 1 driver (CLI `gato-hydrogen`)
├── tests/                  40 tests covering every module above
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

40 tests, all passing on a 32³–128³ grid in about 70 s on CPU. The key ones:

| Test | What it checks | Why it matters |
|---|---|---|
| `test_periodic_plane_wave_is_eigenvector` | $e^{i\mathbf k\cdot\mathbf r}$ on a periodic grid is an *exact* eigenvector of the FD Laplacian, with eigenvalue $-\tfrac{2}{h^2}\sum_\alpha(1-\cos k_\alpha h)$. | Proves the stencil is implemented correctly to machine precision. |
| `test_periodic_continuum_limit_second_order` | As $h \to 0$ the FD eigenvalue converges to $-|\mathbf k|^2$ and the error drops by $\sim 4$× when $N$ doubles. | Verifies the $O(h^2)$ accuracy of the discretization. |
| `test_dirichlet_hermitian` | $\langle\phi\|\nabla^2\psi\rangle = \langle\nabla^2\phi\|\psi\rangle$ for random $\phi,\psi$. | Hermiticity is required for any variational ground-state method to converge to a real minimum. |
| `test_dirichlet_particle_in_box_converges` | The Rayleigh quotient for $\psi = \prod\cos(\pi x_\alpha/L)$ on $[-L/2, L/2]^3$ converges monotonically to the continuum $3\pi^2/(2L^2)$. | Sanity-checks the full $\langle\psi\|\hat T\|\psi\rangle / \langle\psi\|\psi\rangle$ pipeline against a textbook analytic result. |
| `test_harmonic_oscillator_ground_state` | The analytic 3D HO ground state gives $E = \tfrac{3}{2}\omega$ to $< 1\%$. | End-to-end check of `Hamiltonian` composition against an analytic non-Coulomb benchmark. |
| `test_neural_ansatz_cusp_single_center` | The spherically-averaged log-slope $\langle(1/\psi)\partial_r\psi\rangle_\Omega$ at the nucleus equals $-Z$. | Verifies Kato's cusp is satisfied by construction, for any random MLP weights. |
| `test_hydrogen_imag_time_converges` | Imaginary-time propagation from the hydrogenic initial guess gives $E$ within 10 % of $-0.5\,E_h$ on a small grid. | End-to-end sanity of the full Phase 1 stack (grid → kinetic → potential → Hamiltonian → solver → observables). |

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

Every physical system the project solves, across all six phases. A few molecules (H₂, LiH, H₂O) appear twice — intentionally — because they are solved first with mean-field and later with neural VMC, which gives us direct same-molecule method comparisons.

| Phase | System | $n_e$ | Nuclei | Method | Hardware |
|-------|--------|-------|--------|--------|----------|
| 1 ✓ | H (hydrogen atom) | 1 | 1 | grid / neural VQE | CPU |
| 1 ✓ | He⁺, Li²⁺ (hydrogenic ions) | 1 | 1 | same, Z-scaled | CPU |
| 2 | $\text{H}_2^+$ | 1 | 2 | grid + autodiff forces | CPU |
| 3 | He (helium atom) | 2 | 1 | RHF / KS-DFT (SCF) | CPU |
| 4 | H₂ | 2 | 2 | SCF + autodiff forces | CPU / 5070 |
| 4 | LiH | 4 | 2 | SCF + autodiff forces | CPU / 5070 |
| 4 | H₂O | 10 | 3 | SCF + autodiff forces at LDA level | 5070 |
| 5 | H₂ | 2 | 2 | neural VMC (FermiNet-class) | 5070 |
| 5 | LiH | 4 | 2 | neural VMC | 5070 |
| 5 | HF | 10 | 2 | neural VMC; diatomic warm-up for H₂O | 5070 |
| 5 | H₂O | 10 | 3 | neural VMC; headline single-molecule result | 5070 |
| 6 | N₂ | 14 | 2 | neural VMC; triple-bond dissociation benchmark | cloud A100 |
| 6 | $(\text{H}_2\text{O})_2$ water dimer | 20 | 6 | neural VMC; **terminal project target** | cloud A100 |
| 6 (stretch) | $(\text{H}_2\text{O})_4$ | 40 | 12 | neural VMC | cloud A100+ |
| 6 (stretch) | $\text{CH}_4 \cdot (\text{H}_2\text{O})_n$ | ~70 | ~12 | neural VMC (clathrate fragment) | cloud A100+ |

**Distinct molecular systems:** H, He⁺, Li²⁺, He, $\text{H}_2^+$, H₂, LiH, HF, H₂O, N₂, $(\text{H}_2\text{O})_2$, plus the two stretch goals. Each phase's Exit Criterion section below specifies the accuracy target for every member of that row.

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
- [x] Test suite (40 tests): plane-wave exactness at $O(h^2)$ and $O(h^4)$, Hermiticity, HO Lanczos spectrum, Kato cusp spherical-average, $Z$-scaling, end-to-end hydrogen

**Exit criterion:** recover $E_0 \approx -0.5\,E_h$ within 1% on a 64³ grid.

### Phase 2 — Multi-nucleus, single electron ($\text{H}_2^+$)

First taste of chemistry. A stepping stone that introduces two ideas needed for molecules without yet requiring many-electron machinery: multi-center potentials and geometry as a differentiable parameter.

- [ ] `potentials.multi_center_coulomb` — $V(\mathbf r) = -\sum_k Z_k / |\mathbf r - \mathbf R_k|$ (softened)
- [ ] `geometry.py` — nuclei data structure, bond-length / bond-angle observables
- [ ] `jax.grad` of the energy with respect to nuclear positions $\mathbf R_k$ → **forces**
- [ ] Geometry optimization as optax on $\mathbf R$
- [ ] $\text{H}_2^+$ Born–Oppenheimer curve $E(R_{\text{HH}})$ and bond length via gradient descent

**Exit criterion:** recover the $\text{H}_2^+$ equilibrium bond length ($\approx 2.00\,a_0$) by pure gradient descent on the energy. CPU-runnable.

### Phase 3 — Helium via mean-field electronic structure

Enter many-electron land on the simplest closed-shell atom. Single nucleus, two electrons, no molecular geometry yet — the focus is on getting self-consistency right.

- [ ] `scf.py` — self-consistent-field loop with `jax.lax.while_loop`
- [ ] `functionals.py` — LDA exchange-correlation (and later PBE)
- [ ] Hartree potential $J(\mathbf r) = \int \rho(\mathbf r')/|\mathbf r - \mathbf r'|\,dV'$ via 3D FFT convolution
- [ ] Restricted Hartree–Fock with doubly-occupied orbital
- [ ] Kohn–Sham DFT with LDA

**Exit criterion:** helium ground-state energy within chemical accuracy ($\sim 1$ mHa) of reference values ($-2.862\,E_h$ RHF, $-2.834\,E_h$ LDA, $-2.9037\,E_h$ exact).

### Phase 4 — Mean-field molecules and geometry optimization

Combine Phase 2 (multiple nuclei + autodiff forces) with Phase 3 (SCF) to do real chemistry at the mean-field level.

- [ ] SCF with multi-center potentials (H₂, LiH, H₂O at fixed geometry)
- [ ] Hellmann–Feynman + Pulay forces via `jax.grad` on the self-consistent energy
- [ ] Joint $(\theta_{\text{orbitals}}, \mathbf R_{\text{nuclei}})$ optimization
- [ ] Optional pseudopotentials to avoid resolving core electrons on the grid
- [ ] H₂O geometry optimization at the DFT-LDA level — should give bond angle $\approx 104^\circ$, $d_{\text{OH}} \approx 0.97$ Å

**Exit criterion:** H₂O bond angle within $1^\circ$ of the experimental $104.5^\circ$ at the LDA level; correlating with published DFT-LDA values. GPU helpful but not required.

### Phase 5 — Neural many-body wavefunction: single molecules

The neural-VMC milestone. Treat electrons explicitly; parametrize $\Psi(\mathbf r_1, \ldots, \mathbf r_n)$ as a neural Slater-backflow ansatz (FermiNet / PauliNet family) evaluated at Monte Carlo samples. Rather than jumping directly to water, we ramp up through a progression of single molecules of increasing electron count, so each step validates the infrastructure against a less complicated target.

**Core infrastructure (shared by every molecule in this phase):**

- [ ] `ansatz/determinant.py` — Slater determinants of learned orbitals
- [ ] `ansatz/fermi_net.py` — permutation-equivariant backflow network with per-electron features and pairwise streams
- [ ] Jastrow factor with explicit e–e and e–n cusp conditions (reuses the Kato cusp machinery from Phase 1)
- [ ] `sampling.py` — Metropolis–Hastings sampler over $|\Psi|^2$ in $3N$-dimensional space
- [ ] `solvers/vmc.py` — variational Monte Carlo with `kfac-jax` natural-gradient optimizer
- [ ] Joint optimization of network weights + nuclear positions

**Molecule progression:**

| Sub-target | Electrons | Why this molecule |
|---|---|---|
| 5a. H₂ | 2 | smallest VMC test; exact energy $-1.1744\,E_h$ known to 8 decimals; validates determinant + Jastrow machinery before scaling |
| 5b. LiH | 4 | first polar diatomic; 4-electron closed-shell; bond length $\approx 3.015\,a_0$ |
| 5c. HF | 10 | diatomic stepping stone to water: same electron count, only a bond length to optimize (no angle) |
| 5d. H₂O | 10 | the project's headline single-molecule result: bond angle $\approx 104.5°$, bond length $\approx 0.958\,$ Å |

**Hardware**: the author's RTX 5070 (12 GB, consumer Blackwell). Phase-5 runs use mixed precision — FP32 for the network forward pass, FP64 only for the local-energy accumulator — to work around the 5070's $\approx 1/32$ FP64 throughput. Expected wall-clock: single-digit hours for H₂/LiH, ~1 day each for HF and H₂O at chemical-accuracy convergence.

**Exit criterion:** H₂O geometry recovered to within $1°$ of $104.5°$ and $0.01\,$ Å of $0.958\,$ Å, at energy within $\sim 5\,$ mHa of the exact nonrelativistic value $-76.44\,E_h$. Beyond-that targets continue in Phase 6.

### Phase 6 — Strong correlation and the water dimer

The end-target for the project: $(\text{H}_2\text{O})_2$, the smallest piece of condensed water and the foundational unit of ice-Ih. Also a natural home for classical strong-correlation benchmarks that demonstrate where neural VMC pays off over mean-field.

- [ ] **N₂ (14 e⁻)** — canonical strong-correlation benchmark; triple-bond dissociation curve where mean-field methods notoriously fail. Published by FermiNet and PauliNet; direct apples-to-apples comparison.
- [ ] **$(\text{H}_2\text{O})_2$ water dimer (20 e⁻)** — hydrogen-bonded system, ~5 kcal/mol binding energy sitting on top of two monomer energies. The first molecule where the project's "planetary ices" motivation actually starts: (H₂O)₂ is what ice-Ih is built from.
- [ ] (stretch) **$(\text{H}_2\text{O})_4$** and larger water clusters as the compute allows
- [ ] (stretch) A methane clathrate fragment CH₄·(H₂O)ₙ — relevant to Titan

**Hardware**: rented cloud A100 80 GB (Lambda, RunPod, Paperspace) at $1-2 / hr on-demand, for production runs. Native FP64 throughput, 80 GB VRAM, and a typical water-dimer run at $\sim 12\text{-}36$ hours translates to roughly $\$20\text{-}75$ per molecule. Development and debugging continue to happen on the local 5070; the cloud GPU is only spun up for the final production training run, per molecule.

**Exit criterion:** water-dimer binding energy within 1 kcal/mol of the reference CCSD(T) value $\approx -5.0\,$ kcal/mol at the equilibrium geometry; equilibrium O–O distance within $0.05\,$ Å of the experimental $2.98\,$ Å.

Past Phase 6 (larger water clusters, periodic ice, condensed methane) is genuine HPC territory (multi-GPU A100/H100 nodes or real cluster allocations), and is scope for a follow-on project.

---

## 6. Dependencies per phase

The package stays pure-Python, pure-JAX throughout. Total additional dependencies across all six phases: **two runtime packages**.

| Phase | Hardware | New runtime deps | Dev-only deps | Why |
|---|---|---|---|---|
| 1 (done) | laptop CPU | `jax`, `optax`, `equinox`, `numpy`, `scipy`, `matplotlib` | `pytest` | base stack |
| 2 (H₂⁺) | laptop CPU | none | none | multi-center potential + autodiff forces only |
| 3 (helium SCF) | laptop CPU (or 5070) | none | none | FFT is in `jax.numpy.fft`; LDA is ~10 lines of math |
| 4 (mean-field molecules) | local 5070 | none | **`pyscf`** (optional) | pyscf is only for cross-validating our numbers against a trusted reference |
| 5 (neural VMC: H₂ → H₂O) | local 5070 (FP32 net, FP64 energy accumulator) | **`kfac-jax`**, optional **`blackjax`** | none | KFAC natural-gradient is near-essential for VMC convergence; blackjax provides HMC if we want it (Metropolis is easy to hand-roll) |
| 6 (N₂, water dimer) | cloud A100 80 GB ($1-2/hr) | same as Phase 5 | none | development on the 5070, production runs burst to cloud where native FP64 + 80 GB VRAM matter |
| all, on any Nvidia GPU | | `jax[cuda12]` (via `--extra gpu`) | — | bundled CUDA 12 + cuDNN |

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

Phase 1 implementation is **complete end-to-end**.

| Metric | Grid | Result | Target |
|---|---|---|---|
| Hydrogen $E_0$ (imag-time, $96^3$, $L=12$, $O(h^2)$) | | $-0.487\,E_h$ | $-0.500$ |
| Hydrogen $E_0$ (imag-time, $64^3$, $L=12$, $O(h^4)$) | | $-0.479\,E_h$ | $-0.500$ |
| Hydrogen $E_0$ (Lanczos + linear $\epsilon\to 0$ fit, $N=64$) | | $-0.504\,E_h$ | $-0.500$ |
| Virial ratio $2\langle T\rangle/|\langle V\rangle|$ (96³) | | $0.984$ | $1.000$ |
| He⁺ $E_0$ (imag-time, $48^3$, $L=6$, $O(h^4)$) | | $-1.900\,E_h$ | $-2.000$ |
| Li²⁺ $E_0$ (imag-time, $48^3$, $L=4$, $O(h^4)$) | | $-4.275\,E_h$ | $-4.500$ |
| 3D HO Lanczos ladder ($N=40$, $L=10$) | | $1.50, 2.50, 3.50, \ldots$ | $1.5, 2.5, 3.5, \ldots$ |

The $\epsilon \to 0$ linear extrapolation closes the hydrogen residual from $2.6\%$ (fixed softening) to $< 1\%$ ($E_0 = -0.504\,E_h$). The $Z^2$ scaling of the hydrogenic ground state is reproduced across $Z \in \{1, 2, 3\}$ at comparable relative accuracy. The Lanczos solver recovers the full 3D harmonic-oscillator ladder on a $40^3$ grid, providing the eigensolver infrastructure that Phase 3 will use inside the SCF loop. All 40 tests pass.

Next up: **Phase 2** — $\text{H}_2^+$ with a multi-center Coulomb potential and autodiff-based geometry optimization, as the first taste of real chemistry. Phases 3 and 4 (helium SCF → mean-field molecules) follow on the local machine; Phase 5 runs on a single consumer GPU (RTX 5070) over a ladder of molecules (H₂ → LiH → HF → H₂O); Phase 6 bursts to a rented cloud A100 for N₂ and the water dimer. See §5 and §6 for the scoped hardware and dependency trajectory.
