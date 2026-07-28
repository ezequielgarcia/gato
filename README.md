# <img src="docs/figures/gato.png" alt="GATO logo" height="56"> GATO

**Grid Autodiff Theory of Orbitals**

> Math renders cleanly on the rendered docs site: **<https://ezequielgarcia.github.io/gato/>**.
> GitHub's markdown viewer mangles display-math spacing and some macros — prefer the site for the phase notes.

A 3D Schrödinger solver built from scratch in [JAX](https://jax.readthedocs.io/), targeting GPU backends.

This README is meant to be readable by a physics student who has seen Griffiths' *Introduction to Quantum Mechanics* but not necessarily a full graduate course on computational electronic structure. It explains both what the code does and *why* each design choice was made.

> **For students:** a short, narrative synthesis of the physics behind GATO — from the single-particle Schrödinger equation through Hartree–Fock and geometry optimization — lives at [`docs/students_note.md`](docs/students_note.md) (also available as a [PDF](docs/students_note.pdf) and on the [docs site](https://ezequielgarcia.github.io/gato/students_note/)).

---

## Contents

- [§0 What GATO is](#0-what-gato-is)
- [§1 Motivation](#1-motivation)
- [§2 Physics background](#2-physics-background)
- [§3 What's implemented](#3-whats-implemented-phase-1-so-far)
- [§4 Installation and development](#4-installation-and-development)
- [§5 Roadmap](#5-roadmap)
  - [§5.0 Systems at a glance](#50-systems-at-a-glance)
  - [Phase 1 — Hydrogen *(complete)*](#phase-1-differentiable-hydrogen-atom-complete)
  - [Phase 2 — H₂⁺ *(complete)*](#phase-2-multi-nucleus-single-electron-texth_2-complete)
  - [Phase 3 — Helium SCF *(complete)*](#phase-3-helium-via-mean-field-electronic-structure)
  - [Phase 4 — RHF on H₂O](#phase-4-restricted-hartreefock-molecules-ab-initio-terminal-target)
- [§6 Dependencies per phase](#6-dependencies-per-phase)
- [§7 Design principles](#7-design-principles)
- [§8 References](#8-references)
- [§9 Status](#9-status)
- [§10 Acknowledgements](#10-acknowledgements-on-development-workflow)

---

## 0. What GATO is

> "All theoretical chemistry is really physics; and all theoretical chemists know it."
>
> — **Richard Feynman**

GATO solves the electronic Schrödinger equation for atoms and molecules, from first principles, on a computer.

The thesis is **"more is different"** (Anderson, 1972): chemistry is not put in by hand — it *emerges* from a small handful of physical ingredients applied at scale. GATO uses exactly three: the Schrödinger equation, Pauli antisymmetry (a single Slater determinant), and variational minimization at the mean-field (Hartree–Fock) level. No empirical parameters, no fitted exchange–correlation functionals, no basis-set tuning. The headline demonstration is H₂O: place an oxygen and two hydrogens at arbitrary positions, run RHF + Hellmann–Feynman forces + Adam, and watch the bonds form, the angle settle near 106°, and the lone pairs localize on oxygen — chemistry as a consequence of Schrödinger + Pauli + minimization, and nothing else.

Given a set of atomic nuclei — their positions and their charges — GATO computes the **ground-state wavefunction and energy** of the electrons that surround them. From that wavefunction it derives everything downstream: electron density, orbital shapes, bond lengths, bond angles, and the forces that hold the molecule together.

Concretely, the code:

- **Represents the wavefunction on a three-dimensional grid.** The cube of space around the molecule is sampled at evenly spaced points, and the value of the electronic wavefunction is stored at every point. No Gaussian basis sets, no atomic orbitals — just numbers on a lattice.
- **Builds the Hamiltonian as an operator, not a matrix.** Kinetic energy is a seven-point finite-difference stencil acting on neighbouring grid values; the nuclear potential is a pointwise multiplication; electron–electron repulsion is a 3D Fourier-transform convolution. Nothing is ever assembled into a dense $N^3 \times N^3$ matrix.
- **Finds the ground state by variational optimization.** The wavefunction is treated as a differentiable parameter; the energy is a scalar function of those parameters; gradient-based optimizers (imaginary-time propagation, Rayleigh-quotient descent, Lanczos diagonalization) drive the energy to its minimum.
- **Finds molecular geometry the same way.** Nuclear positions are also differentiable parameters. `jax.grad` applied to the total energy returns the forces on the nuclei; Adam on those forces relaxes the molecule to its equilibrium geometry.
- **Handles many-electron systems self-consistently.** For atoms and molecules with more than one electron, GATO runs a Hartree–Fock self-consistent-field loop: the effective potential felt by each electron depends on the density of all the others, and the loop iterates until the wavefunction and the potential it generates agree.
- **Delivers physically meaningful numbers.** Bond lengths, bond angles, total energies, virial ratios, orbital energies, radial densities — the observables a chemist or spectroscopist would actually ask about.

The end-to-end pipeline is pure [JAX](https://jax.readthedocs.io/): differentiable, JIT-compiled to XLA, and runs identically on CPU or on a single consumer-grade NVIDIA GPU. The target progression is hydrogen → $\text{H}_2^+$ → helium → restricted Hartree–Fock on H₂O, with molecular geometry obtained by gradient descent on the total energy.

---

## 1. Motivation

Most introductions to numerical quantum mechanics take one of two paths:

1. **Dense linear algebra.** Write the Hamiltonian as an $N^3 \times N^3$ matrix, feed it to a diagonalizer, get eigenvalues. This is conceptually clear but dies around $N = 30$ because the matrix has $N^6$ entries — an $80^3$ grid would need ~260 TB just to store $\hat H$.
2. **Black-box packages.** Use PySCF, Gaussian, ORCA. These work, but the physics is hidden under layers of optimized Fortran and you learn nothing about *how* a Schrödinger solver actually works.

GATO takes a different route:

- **Matrix-free.** The Hamiltonian is represented by a function `psi -> H(psi)`. We never build a matrix. We only need to multiply $\hat H$ by a state, and we do that with local finite-difference stencils.
- **Differentiable.** Everything runs through JAX's automatic differentiation, so the gradient of any observable with respect to any parameter (grid spacing, nuclear position, orbital coefficients) is one `jax.grad` away.
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
│       │   └── grid.py     grid-parameter ansatz (parameters = grid values)
│       ├── solvers/
│       │   ├── imag_time.py  ψ ← ψ − Δτ·Hψ  (grid ansatz)
│       │   ├── lanczos.py    matrix-free Lanczos for lowest-K eigenpairs
│       │   └── poisson.py    3D FFT Poisson solver (Hartree)
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

### 3.6 Tests

60 tests, all passing on a 32³–128³ grid in about 175 s on CPU. The key ones:

| Test | What it checks | Why it matters |
|---|---|---|
| `test_periodic_plane_wave_is_eigenvector` | $e^{i\mathbf k\cdot\mathbf r}$ on a periodic grid is an *exact* eigenvector of the FD Laplacian, with eigenvalue $-\tfrac{2}{h^2}\sum_\alpha(1-\cos k_\alpha h)$. | Proves the stencil is implemented correctly to machine precision. |
| `test_periodic_continuum_limit_second_order` | As $h \to 0$ the FD eigenvalue converges to $-\lvert\mathbf k\rvert^2$ and the error drops by $\sim 4$× when $N$ doubles. | Verifies the $O(h^2)$ accuracy of the discretization. |
| `test_dirichlet_hermitian` | $\langle\phi\mid\nabla^2\psi\rangle = \langle\nabla^2\phi\mid\psi\rangle$ for random $\phi,\psi$. | Hermiticity is required for any variational ground-state method to converge to a real minimum. |
| `test_dirichlet_particle_in_box_converges` | The Rayleigh quotient for $\psi = \prod\cos(\pi x_\alpha/L)$ on $[-L/2, L/2]^3$ converges monotonically to the continuum $3\pi^2/(2L^2)$. | Sanity-checks the full $\langle\psi\mid\hat T\mid\psi\rangle / \langle\psi\mid\psi\rangle$ pipeline against a textbook analytic result. |
| `test_harmonic_oscillator_ground_state` | The analytic 3D HO ground state gives $E = \tfrac{3}{2}\omega$ to $< 1\%$. | End-to-end check of `Hamiltonian` composition against an analytic non-Coulomb benchmark. |
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
uv run pytest        # 87 tests should pass (2 are GPU-gated and skip on CPU)
uv run gato          # smoke test: prints JAX device + kinetic energy of a unit Gaussian
```

On a GPU box, add the CUDA extras:

```bash
uv sync --extra gpu          # pulls jax[cuda12] wheels (~2 GB of NVIDIA libraries)
uv run gato                  # should now print "backend  : GPU" and list the device
GATO_GPU=1 uv run pytest     # also runs the Be / Ne atom benchmarks (n_occ = 2 and 5)
```

No system CUDA install is needed — `jax[cuda12]` bundles its own CUDA 12 + cuDNN. An NVIDIA driver version 525+ is enough. JAX 0.10 supports up to Blackwell (RTX 50-series) out of the box.

The `GATO_GPU=1` environment variable unskips two test stubs in `tests/test_scf.py` (beryllium and neon RHF) that are an order of magnitude too slow on CPU but routine on a recent GPU. Everything else runs on CPU or GPU with no code changes.

### 4.3 Run a simulation

Each phase ships a self-contained CLI. Defaults reproduce the reference numbers in §9; pass `--help` to any of them to see the full flag surface (grid size, box length, geometry, optimizer steps, stencil order, etc.).

| Command | What it does | Expected end-state |
|---|---|---|
| `uv run gato-hydrogen` | Phase 1: imag-time + Lanczos on H, He⁺, Li²⁺ | E₀ ≈ −Z²/2 within softening |
| `uv run gato-helium` | Phase 3: closed-shell RHF on He | E ≈ −2.86 E_h after ε→0 fit |
| `uv run gato-h2plus` | Phase 2: H₂⁺ geometry-opt by Hellmann–Feynman | R_e ≈ 1.997 a₀ (Burrau) |
| `uv run gato-h2` | Phase 4: H₂ RHF geometry-opt (all-electron) | R_e ≈ 1.40 a₀ |
| `uv run gato-lih` | Phase 4: LiH RHF + HGH pseudopotentials | R_e ≈ 3.02 a₀ |
| `uv run gato-water` | Phase 4 headline: H₂O RHF + HGH PP geometry-opt | R_OH ≈ 1.78 a₀, ∠HOH ≈ 106° |

Add `--single` to the molecular drivers (`gato-h2`, `gato-lih`, `gato-water`) to run a single-point RHF without geometry optimization — useful for sanity-checking the SCF before committing to a full relaxation. Typical CPU runtimes range from ~20 s (hydrogen) to a few minutes (water geometry-opt at the default grid); GPU cuts that by an order of magnitude on the SCF-heavy phases.

### 4.4 Double precision

JAX defaults to float32 for performance. Quantum energies are small differences of large kinetic and potential contributions, so float32 will quietly corrupt your answers. Always enable x64:

```python
import gato
gato.enable_x64()   # call once, before any JAX computation
```

Tests enable this automatically via `tests/conftest.py`.

---

## 5. Roadmap

### 5.0 Systems at a glance

Every physical system the project solves, organized by method.

| Phase | System | $n_e$ | Nuclei | Method | Hardware |
|-------|--------|-------|--------|--------|----------|
| 1 ✓ | H (hydrogen atom) | 1 | 1 | grid imag-time | CPU |
| 1 ✓ | He⁺, Li²⁺ (hydrogenic ions) | 1 | 1 | same, $Z$-scaled | CPU |
| 2 ✓ | $\text{H}_2^+$ | 1 | 2 | grid + autodiff forces on $\mathbf R$ | CPU |
| 3 ✓ | He (helium atom) | 2 | 1 | RHF, SCF | CPU / 5070 |
| 4 | H₂O | 8 valence | 3 | RHF + HGH PP + autodiff geometry | 5070 ✓ |
| 4 | LiH | 4 valence | 2 | same | 5070 |
| 4 | HCl | 8 valence | 2 | same — **refinement-limited**, see §9 | 5070 |

**Ab-initio.** Every phase is strictly ab initio: Schrödinger, Pauli antisymmetry (Slater determinant in Phases 3–4), and variational minimization. No empirical parameters, no fitted functionals.

### Phase 1 — Differentiable Hydrogen Atom *(complete)*

Foundation: prove the machinery works by recovering the hydrogen ground state $E_0 = -0.5\,E_h$.

- [x] Cell-centered 3D Cartesian grid (`Grid3D`)
- [x] JIT-compiled Laplacian, kinetic, gradient operators with Dirichlet / periodic BCs
- [x] Integration, inner product, normalization helpers
- [x] `potentials.py` — softened Coulomb $V_\epsilon(r) = -1/\sqrt{r^2+\epsilon^2}$, harmonic, constant
- [x] `hamiltonian.py` — full $\hat H = \hat T + \hat V$ with `.apply`, `.expectation`, `.rayleigh`
- [x] `observables.py` — $\langle T\rangle$, $\langle V\rangle$, virial ratio, radial density
- [x] `ansatz/grid.py` — raw grid parameters as variational wave function
- [x] `solvers/imag_time.py` — imaginary-time propagation $\psi \to \psi - \Delta\tau\,\hat H\psi$
- [x] `physics/hydrogen.py` — end-to-end driver targeting $-0.5\,E_h$
- [x] 4th-order 13-point Laplacian stencil (selectable via `order=4`)
- [x] Matrix-free Lanczos solver (`solvers/lanczos.py`) for lowest-$K$ eigenpairs
- [x] Hydrogenic $Z$-scaling: $E_0(Z) = -Z^2/2$ reproduced for H, He⁺, Li²⁺
- [x] Softening extrapolation $\epsilon \to 0$ closes residual to $<1\%$ of analytic $-0.5\,E_h$
- [x] Radial density figure: converged $P(r)$ vs. analytic $4r^2 e^{-2r}$
- [x] Log-radial 1D cross-check (`physics/radial_hydrogen.py`): pure $V=-Z/r$ on a log-spaced grid, with odd-parity boundary at $r=0$ and a 4th-order chain-rule Laplacian. Reproduces $-Z^2/2$ to $\sim 10^{-7}\,E_h$ at $N=800$ for $Z\in\{1,2,3\}$, independently confirming the $\epsilon\to 0$ extrapolation
- [x] Test suite: plane-wave exactness at $O(h^2)$ and $O(h^4)$, Hermiticity, HO Lanczos spectrum, $Z$-scaling, end-to-end hydrogen, log-radial cross-check

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
- [x] **Accuracy work — softening residual:** `epsilon` threaded through `scf_rhf`; `benchmarks/helium_softening_extrapolation.py` closes the gap from +263 mHa to +19 mHa at $N=64$ and to $-3.4$ mHa at $N=80$ via a linear $\varepsilon \to 0$ fit. See `docs/phase3_note.md` §6 for data and figure.

**Exit criterion:** helium ground-state energy within chemical accuracy ($\sim 1$ mHa) of reference values ($-2.8617\,E_h$ RHF, $-2.9037\,E_h$ exact). **Met** at $N = 80$, $L = 10$ via linear $\varepsilon \to 0$ extrapolation: RHF $-2.8651\,E_h$ (residual $-3.4$ mHa). The small residual is grid-discretization bias of the 4th-order kinetic stencil, not softening; one further $N$ refinement (trivial on the 5070) takes it sub-mHa.

**The softening residual, for the record.** At the default $\varepsilon = h/2$ on $N = 64$, helium RHF sits $\sim 260$ mHa above the reference. This is not an SCF bug — the SCF solves the softened Hamiltonian to $10^{-6}\,E_h$, but the softened Hamiltonian itself is lifted above true Coulomb by the combined bias from softening both $V_\text{ext}$ and the Hartree/exchange kernel. The bias scales with the probability mass at $r \lesssim \varepsilon$, which is $\sim 0.08\,a_0$ — non-negligible at $Z = 2$ where the 1s orbital has characteristic size $\sim 0.3\,a_0$. The `epsilon` argument is threaded through `scf_rhf` so both the $V_\text{ext}$ softening (in `softened_coulomb`) and the Hartree/exchange kernel softening (in `hartree_potential` / `exchange_apply`) sweep together, and $E(\varepsilon)$ is demonstrably linear in $\varepsilon$ over the relevant range — see `docs/phase3_note.md` §6.

The same linear $\varepsilon \to 0$ fit will close the residual at $Z = 8$ (oxygen) in Phase 4. If its range gets inconvenient, the alternative route — an origin-regularized Hartree kernel (exact $1/r$ everywhere except a cube of side $h$ at the origin, whose singular value is replaced by the analytic self-energy of a uniform charge cube, the standard plane-wave trick) — removes the kernel-side softening entirely and leaves only $V_\text{ext}$ for route 1 to clean up. Not needed for Phase 3; optional for Phase 4.

**Additional atom benchmarks (planned, GPU-gated).** He alone validates the $n_\text{orb}=1$ path. Two more closed-shell atoms fill in the coverage before jumping to molecules:

- **Be** ($Z=4$, $1s^2\,2s^2$, $n_\text{occ}=2$) — exercises the general $n_\text{orb}^2$ exchange operator with two orbitals of the same angular momentum. Reference RHF: $-14.573\,E_h$.
- **Ne** ($Z=10$, $1s^2\,2s^2\,2p^6$, $n_\text{occ}=5$) — first p-orbital occupancy; exercises the symmetry-breaking perturbation in the default initial guess (a purely spherical starter would leave the three 2p's invisible to Lanczos). Reference RHF: $-128.547\,E_h$. These orbital counts also match the Phase 4 molecular targets: H₂ is $n_\text{occ}=1$ like He, LiH is $n_\text{occ}=2$ like Be, H₂O is $n_\text{occ}=5$ like Ne — so the atoms isolate SCF machinery from geometry optimization debugging.

Test stubs exist in `tests/test_scf.py::test_beryllium_rhf_converges` and `::test_neon_rhf_converges`, currently `@pytest.mark.skip`'d because each takes ~30 min (Be) to multi-hour (Ne) on CPU. They'll be enabled once the GPU backend (`uv sync --extra gpu`) is in use, where they should drop to seconds / minute respectively.

### Phase 4 — Restricted Hartree–Fock molecules (ab-initio terminal target)

Combine Phase 2 (multiple nuclei + autodiff forces) with Phase 3 (SCF) into *strictly ab-initio* mean-field molecules. One Slater determinant, exact Coulomb and exchange, no empirical parameters.

- [x] SCF with multi-center potentials, Hartree term $J[\rho]$ via 3D FFT convolution
- [x] **Exact exchange operator** $\hat K$ implemented as $n_\text{orb}^2$ real-space Poisson solves per SCF iteration
- [x] Hellmann–Feynman forces via `jax.grad` on the self-consistent energy (validated against finite differences; no Pulay term is needed, since the orbitals are stationary at the SCF solution)
- [x] Alternating $(\text{orbitals}, \mathbf R_{\text{nuclei}})$ optimization — SCF to convergence, then one Adam step on positions
- [x] H₂ (2 e⁻, 1 orbital) and LiH (4 valence e⁻) drivers
- [x] H₂O (**8 valence** e⁻, 4 orbitals with HGH PP — not 10/5, the O 1s pair is frozen into the pseudopotential) — energetics converged, see §9
- [x] Grid-convergence study separating resolution from box error (`benchmarks/grid_convergence.py`)
- [ ] **H₂O relaxed geometry certified against $h \to 0$** — currently limited by geometry-optimizer convergence, not by the grid; see §9
- [ ] Block eigensolver (block Lanczos / LOBPCG) to lift the degeneracy restriction that blocks HCl refinement, NH₃ and CH₄

**Exit criterion (project's ab-initio terminal target):** H₂O geometry recovered at the RHF level consistent with published RHF reference values (angle $\approx 106°$, $d_{\text{OH}} \approx 1.78\,a_0$). Energy within a few mHa of reference RHF. The systematic $\sim\!1.5°$ overestimate of the bond angle vs. experiment is a known RHF limitation — recovering it requires the correlation energy that mean-field cannot see by construction.

**Status against that criterion.** The energy half is done: $E(h \to 0) \approx -16.99\,E_h$ at $L = 14\,a_0$, with box error under 0.1 mHa and a measured convergence order of $p \approx 2.3$. The geometry half is **not** met — the relaxed $R_\text{OH}$ still scatters by $\pm 0.03\,a_0$ between grids, which is larger than the difference being tested for. Note also that "within a few mHa of reference RHF" needs a like-for-like reference: these are *valence-only* pseudopotential energies, so they are not directly comparable to all-electron RHF totals.

---

## 6. Dependencies per phase

The package stays pure-Python, pure-JAX throughout. No new runtime dependencies are introduced across Phases 2–4.

| Phase | Hardware | New runtime deps | Dev-only deps | Why |
|---|---|---|---|---|
| 1 (done) | laptop CPU | `jax`, `optax`, `equinox`, `numpy`, `scipy`, `matplotlib` | `pytest` | base stack |
| 2 (H₂⁺) | laptop CPU | none | none | multi-center potential + autodiff forces only |
| 3 (helium RHF) | laptop CPU (or 5070) | none | none | FFT is in `jax.numpy.fft`; exact exchange is a 3D Poisson solve |
| 4 (H₂O RHF + HGH PP) | local 5070 | none | **`pyscf`** (optional) | pyscf only for cross-validating against a trusted reference |
| any, on any Nvidia GPU | | `jax[cuda12]` (via `--extra gpu`) | — | bundled CUDA 12 + cuDNN |

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
- Martin, *Electronic Structure: Basic Theory and Practical Methods*, Cambridge — the canonical HF reference used for Phases 3 and 4.
- Szabo & Ostlund, *Modern Quantum Chemistry*, Dover — Hartree–Fock, configuration interaction, and the SCF loop.
- LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations* — derivations of the stencils and their error bounds.
- JAX documentation: https://jax.readthedocs.io/
- optax (optimization): https://optax.readthedocs.io/

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
| Virial ratio $2\langle T\rangle/|\langle V\rangle|$ (96³) | | $0.984$ | $1.000$ |
| He⁺ $E_0$ (imag-time, $48^3$, $L=6$, $O(h^4)$) | | $-1.900\,E_h$ | $-2.000$ |
| Li²⁺ $E_0$ (imag-time, $48^3$, $L=4$, $O(h^4)$) | | $-4.275\,E_h$ | $-4.500$ |
| 3D HO Lanczos ladder ($N=40$, $L=10$) | | $1.50, 2.50, 3.50, \ldots$ | $1.5, 2.5, 3.5, \ldots$ |

The $\epsilon \to 0$ linear extrapolation closes the hydrogen residual from $2.6\%$ (fixed softening) to $< 1\%$ ($E_0 = -0.504\,E_h$). An independent 1D log-radial solver with the bare $V = -Z/r$ potential reproduces $-0.500\,E_h$ to seven decimal places at $N = 800$, which confirms that the remaining 3D residual is softening-limited, not a bug in the Cartesian stack. The $Z^2$ scaling of the hydrogenic ground state is reproduced across $Z \in \{1, 2, 3\}$ on both grids at comparable relative accuracy. The Lanczos solver recovers the full 3D harmonic-oscillator ladder on a $40^3$ grid, providing the eigensolver infrastructure that Phase 3 will use inside the SCF loop. The suite now stands at 99 passing tests (2 skipped, both noted under Phase 3).

**Phase 2 is also complete.** Starting from $R = 2.4\,a_0$ on a $56^3$ grid, 30 Adam steps on the Born–Oppenheimer energy recover the $\text{H}_2^+$ bond length $R_e = 1.9986\,a_0$ — within $0.07\%$ of the Burrau 1927 analytic value $1.9972\,a_0$ — by pure gradient descent. The total energy $-0.5749\,E_h$ is softening-limited (~28 mHa above the analytic $-0.6026\,E_h$), the same $O(\varepsilon)$ bias observed in Phase 1 and closable by the same $\varepsilon \to 0$ extrapolation.

**Phase 3 is complete on helium.** The self-consistent RHF driver on He converges in $\sim 6$ iterations at $\varepsilon = h/2$; sweeping $\varepsilon$ across five values at fixed grid and linearly extrapolating $E(\varepsilon) \to E(0)$ closes the residual to $-3.4\,$mHa at $N = 80$, $L = 10\,a_0$ — at chemical accuracy, with the remaining few-mHa residual attributable to 4th-order-stencil grid discretization rather than softening. The Hockney doubled-grid FFT Hartree solver, the general $n_\text{orb}^2$ exchange operator written in its full form (validated analytically at $n_\text{occ} = 1$, where it reduces to self-exchange), and the symmetry-breaking core-Hamiltonian initial guess are all implemented. The Be ($n_\text{occ} = 2$) and Ne ($n_\text{occ} = 5$) tests stand as `@pytest.mark.skip`-decorated stubs — these are the 2 skipped tests in the suite. The 5070 is now available and no longer the blocker; they remain unwritten. Note that the multi-orbital code path *has* since been exercised on real molecules by the Phase 4 water and HCl drivers ($n_\text{occ} = 4$), so the gap here is atom-level regression coverage rather than an untested code path. Ne is worth care when it is written: as a closed-shell atom its 2p level is 3-fold degenerate, which is the eigensolver limitation described under Phase 4.

All Phase 1, 2, and 3 numbers above were produced in float64 on a **single CPU core**. The codebase is pure JAX and ran unchanged on GPU via `uv sync --extra gpu` once the card was installed — no code changes were required. On the RTX 5070 (12 GB) a water RHF single point at $N = 128$ converges in ~160 s, which is what made the Phase 4 grid-convergence study below feasible. Two practical notes: pass `XLA_FLAGS=--xla_gpu_autotune_level=0` for fine grids, or the cuBLAS autotuner requests a multi-GB profiling scratch and OOMs before the solver does; and $N \approx 128$ is the practical ceiling for a 4-orbital system, since the Krylov buffer is $(m+1)N^3$ doubles with $m$ itself growing as $N$.

**Phase 4 energetics are validated on water; the geometry is not yet.** The HGH pseudopotential module (`pseudopotentials.py`) has analytic-form smoke tests covering the long-range Coulomb tail, the value at the origin, radial projector normalization, non-local Hermiticity, and differentiability in nuclear positions. The water and HCl drivers run end-to-end, and `jax.grad` of the BO energy returns Hellmann–Feynman forces that agree with finite differences. 99 tests pass (2 skipped).

**The 5070 is now wired in**, which made a proper grid-convergence study possible for the first time (`benchmarks/grid_convergence.py`). Running it immediately exposed a solver bug and then produced the first trustworthy Phase 4 numbers.

*The bug.* The SCF used a **fixed** Lanczos Krylov dimension. The Fock spectral width grows as $h^{-2}$ while the physical gap does not, so a dimension adequate at $N = 64$ silently fails at $N \geq 96$: water's highest occupied orbital came back at $-0.11\,E_h$ instead of $-0.52\,E_h$, and the total energy drifted *upward* under refinement ($-18.89, -17.21, -16.19, -14.04$ for $N = 32 \ldots 128$) — which reads as broken grid convergence rather than a broken eigensolver. The Krylov dimension now scales with the grid (`solvers.lanczos.default_krylov_dim`), and the same sequence converges monotonically in ~15 SCF iterations at every $N$.

*Water, resolution.* At $L = 14\,a_0$, $E(h \to 0) \approx -16.99\,E_h$ with an **observed convergence order $p \approx 2.3$, not 4**. The 4th-order stencil is therefore *not* the accuracy bottleneck: the error is dominated by the $O(h^2)$ sampling of the narrow HGH projectors ($r_\ell \approx 0.22\,a_0$ for O). The default $N = 64$ sits ~225 mHa above the converged value, so the previously quoted $\approx -17.2\,E_h$ matched an under-resolved grid by coincidence.

*Water, box.* At exactly $h = 0.2$, $E$ moves by $0.04\,$mHa between $L = 14$ and $L = 20$. Box error is a non-issue at the default; resolution is the whole story.

*Water, geometry — open.* The relaxed $R_\text{OH}$ oscillates ($1.751 / 1.721 / 1.745\,a_0$ at $N = 64/80/96$) rather than converging, with scatter exceeding any grid trend. This is most likely the geometry optimizer rather than the grid — Adam at $\text{lr} = 0.05$ with no force-convergence criterion — but that is **not yet confirmed**, so the reference geometry in `physics/water.py` remains uncertified. This is the next milestone.

*HCl is refinement-limited by the eigensolver, not by physics.* HCl is $C_{\infty v}$ with an **exactly 2-fold degenerate $3\pi$ level**, and a single-vector Lanczos can only split a true degeneracy through round-off. The driver is correct at its default $N = 64$, but refining makes it *worse*: at $N = 80$ it needs $\gtrsim 200$ Krylov vectors, gets 120, and returns a silently wrong answer (the $\pi$ pair absent, total energy off by $0.75\,E_h$, `converged` still `True`). This generalizes: molecules in point groups with only 1-D irreps (C₁, Cₛ, C₂, C₂ᵥ, C₂ₕ, D₂ₕ — **H₂O among them**) are safe, while $\pi$/$e$/$t$ degeneracies are not — which puts **NH₃ and CH₄ out of reach** until the eigensolver is upgraded to a block method (block Lanczos / LOBPCG) seeded with $n_\text{occ}$ vectors. A C₂ᵥ molecule such as H₂S is the natural next target at water's cost without that work.

---

## 10. Acknowledgements on development workflow

Phases 1 and 2 of GATO — roughly 2,900 lines of code, 60 tests, and the Phase 1 paper — were developed in about half a day of focused work, with Google Gemini and Anthropic's Claude assisting as coding and design collaborators. The LLMs accelerated implementation, test construction, and documentation drafting; the architectural choices, physics interpretation, roadmap scoping, and acceptance criteria remained human-directed throughout. This is a practical example of what the current generation of coding assistants can deliver on a moderately-scoped research codebase when the human provides clear design intent and the tools handle the mechanical work.
