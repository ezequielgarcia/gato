---
title: "Computational Quantum Mechanics: From First Principles to Molecular Reality"
subtitle: "A student's guide to the GATO (Grid Autodiff Theory of Orbitals) framework"
author: "Ezequiel Garcia"
date: "April 2026"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
linkcolor: MidnightBlue
urlcolor: MidnightBlue
header-includes:
  - \usepackage{amsmath,amssymb,bm}
---

> **Read this on <https://ezequielgarcia.github.io/gato/students_note/>** — GitHub's markdown viewer does not render display math reliably.

This note traces the physical foundations and mathematical architectures implemented in GATO, following the evolution from an isolated particle in a vacuum to the relativistic, N-body systems required to simulate heavy elements and modern chemical dynamics. It is aimed at a physics or chemistry student who has taken a first course in quantum mechanics (Griffiths level) but not a graduate course on electronic structure.

Throughout we use **Hartree atomic units**: $\hbar = m_e = e = 4\pi\varepsilon_0 = 1$. In these units the hydrogen ground state has energy $-\tfrac{1}{2}\,E_h$ and Bohr radius $1\,a_0$. Dimensional constants like $\hbar$, $m_e$, $4\pi\varepsilon_0$ are kept explicit only where they aid the physical reading of a formula (notably in §4, where factors of $c$ matter).

---

## Contents

- [§1 The single-particle foundation](#1-the-single-particle-foundation)
- [§2 The many-body challenge and Hartree–Fock](#2-the-many-body-challenge-and-hartreefock)
- [§3 The Density Functional Theory alternative](#3-the-density-functional-theory-alternative)
- [§4 Scaling to heavy elements: the relativistic imperative](#4-scaling-to-heavy-elements-the-relativistic-imperative)
- [§5 Geometry optimization and forces](#5-geometry-optimization-and-forces)

---

## 1. The Single-Particle Foundation

The journey begins with the **time-independent Schrödinger equation** (TISE). For a single electron moving in a static external potential — hydrogen, or the one-electron molecular ion $\text{H}_2^+$ — the goal is to find the eigenstates of the Hamiltonian,

$$
\hat H\,\psi(\mathbf r) = E\,\psi(\mathbf r),
\qquad
\hat H = -\tfrac{1}{2}\nabla^2 + V(\mathbf r).
$$

The first term is the kinetic-energy operator (in atomic units, $\hat T = -\tfrac{1}{2}\nabla^2$); the second is the electrostatic attraction of the electron to the fixed nuclei. For a single point-nucleus of charge $Z$ at position $\mathbf R$, the textbook Coulomb potential is

$$
V_{\text{Coulomb}}(\mathbf r) = -\frac{Z}{\lvert \mathbf r - \mathbf R \rvert}.
$$

### 1.1 Why the bare Coulomb potential is a problem on a grid

GATO stores $\psi$ as an array of values on a cubic lattice of points. At the grid point nearest the nucleus, $\lvert \mathbf r - \mathbf R \rvert$ can be arbitrarily small — in the continuum it goes to zero and $V \to -\infty$. Any finite-precision computer evaluating a $-\infty$ produces garbage (`NaN`s, overflow) and the simulation crashes before it begins.

GATO resolves this with a **softened Coulomb potential**, introducing a small regularization length $\epsilon$:

$$
V_\epsilon(\mathbf r) = -\frac{Z}{\sqrt{\lvert \mathbf r - \mathbf R \rvert^2 + \epsilon^2}}.
$$

Two properties are worth noticing:

- **At long range** ($\lvert \mathbf r - \mathbf R \rvert \gg \epsilon$), the $\epsilon^2$ inside the square root is negligible and $V_\epsilon \to -Z/\lvert \mathbf r - \mathbf R\rvert$ — we recover the physical Coulomb tail that actually binds the electron.
- **At the nucleus** ($\mathbf r = \mathbf R$), $V_\epsilon = -Z/\epsilon$, a finite number. The singularity is capped.

In practice $\epsilon$ is set comparable to the grid spacing $h$ (e.g. $\epsilon = h/2$), so the softening affects only the sub-grid region that the mesh could not resolve anyway. Extrapolating $\epsilon \to 0$ is the standard convergence check.

---

## 2. The Many-Body Challenge and Hartree–Fock

When multiple electrons are present, the problem becomes exponentially harder because every electron continuously repels every other. The Hamiltonian picks up a pairwise electron–electron term,

$$
\hat H = \sum_{i=1}^{N_e}\Bigl[-\tfrac{1}{2}\nabla_i^2 + V_\text{ext}(\mathbf r_i)\Bigr]
\;+\; \sum_{i<j}\frac{1}{\lvert \mathbf r_i - \mathbf r_j\rvert},
$$

and the wavefunction now depends on all $N_e$ electron coordinates simultaneously, $\Psi(\mathbf r_1, \mathbf r_2, \dots, \mathbf r_{N_e})$. Naïvely storing this on a grid of $N$ points per axis costs $N^{3 N_e}$ numbers — ten electrons on a $64^3$ grid would require more storage than atoms in the observable universe. This is the **curse of dimensionality**.

The physical "volume" and structural integrity of matter are not dictated by classical collisions, but by the **Pauli exclusion principle** (PEP): no two electrons can occupy the same quantum state, and the many-electron wavefunction must be antisymmetric under the exchange of any two electron labels,

$$
\Psi(\dots,\mathbf r_i,\dots,\mathbf r_j,\dots) = -\Psi(\dots,\mathbf r_j,\dots,\mathbf r_i,\dots).
$$

### 2.1 The Slater-determinant ansatz

To make the problem tractable, **Hartree–Fock** (HF) assumes the many-electron wavefunction takes the form of a **Slater determinant** of one-electron orbitals $\phi_1,\dots,\phi_{N_e}$:

$$
\Psi(\mathbf r_1,\dots,\mathbf r_{N_e}) = \frac{1}{\sqrt{N_e!}}\,
\det\!\begin{pmatrix}
\phi_1(\mathbf r_1) & \phi_2(\mathbf r_1) & \cdots & \phi_{N_e}(\mathbf r_1) \\
\phi_1(\mathbf r_2) & \phi_2(\mathbf r_2) & \cdots & \phi_{N_e}(\mathbf r_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(\mathbf r_{N_e}) & \phi_2(\mathbf r_{N_e}) & \cdots & \phi_{N_e}(\mathbf r_{N_e})
\end{pmatrix}.
$$

For two electrons this collapses to the familiar $\tfrac{1}{\sqrt{2}}[\phi_1(\mathbf r_1)\phi_2(\mathbf r_2) - \phi_2(\mathbf r_1)\phi_1(\mathbf r_2)]$. The determinant structure automatically enforces antisymmetry — swapping two rows flips the sign — and if two columns coincide ($\phi_i = \phi_j$) the determinant vanishes, which *is* the Pauli exclusion principle.

### 2.2 The Fock equation

Minimising the expectation value $\langle\Psi\lvert \hat H\rvert\Psi\rangle$ with respect to the orbitals, subject to their staying orthonormal, yields the **Fock equation** — a one-electron eigenvalue problem of exactly the same shape as TISE, but with an effective potential that depends on the occupied orbitals themselves:

$$
\hat f\,\phi_i(\mathbf r) = \epsilon_i\,\phi_i(\mathbf r),
\qquad
\hat f = -\tfrac{1}{2}\nabla^2 + V_\text{ext}(\mathbf r) + \hat J - \hat K.
$$

The two new operators encode the average electron–electron interaction:

- **Hartree (Coulomb) operator** $\hat J$. The classical electrostatic repulsion from the mean charge density of all other electrons,
$$
\hat J\,\phi_i(\mathbf r) = \left[\,\sum_{j}\int\frac{\lvert \phi_j(\mathbf r')\rvert^2}{\lvert \mathbf r - \mathbf r'\rvert}\,d\mathbf r'\,\right]\phi_i(\mathbf r).
$$
This is a *local* multiplicative potential: at each point in space, $\phi_i$ is multiplied by a number.

- **Exchange operator** $\hat K$. A purely quantum effect with no classical analogue, arising from antisymmetry,
$$
\hat K\,\phi_i(\mathbf r) = \sum_{j}\phi_j(\mathbf r)\int\frac{\phi_j^{*}(\mathbf r')\,\phi_i(\mathbf r')}{\lvert \mathbf r - \mathbf r'\rvert}\,d\mathbf r'.
$$
This is *non-local*: the value of $\hat K\phi_i$ at $\mathbf r$ depends on $\phi_i$ everywhere. Physically, exchange keeps same-spin electrons apart and is why matter has volume.

Because $\hat f$ depends on the orbitals $\phi_j$ it acts on, the equations are solved iteratively: guess orbitals $\to$ build $\hat f$ $\to$ diagonalise $\to$ update orbitals $\to$ repeat. This is the **self-consistent field** (SCF) loop.

---

## 3. The Density Functional Theory Alternative

Tracking $N_e$ individual orbitals is still expensive, and the non-local exchange operator $\hat K$ is particularly unfriendly on a grid. **Density Functional Theory** (DFT) sidesteps the orbitals by shifting focus entirely to the electron density

$$
\rho(\mathbf r) = \sum_{i} \lvert \phi_i(\mathbf r)\rvert^2,
$$

a three-dimensional scalar field regardless of how many electrons there are.

The theoretical foundation is the **Hohenberg–Kohn theorem**: for a non-degenerate ground state, the density $\rho(\mathbf r)$ uniquely determines the external potential (up to an additive constant), and therefore the full Hamiltonian, the wavefunction, and every observable. In principle, all ground-state physics is a functional of $\rho$ alone.

### 3.1 The Kohn–Sham mapping

Hohenberg–Kohn is a non-constructive existence proof; the **Kohn–Sham** (KS) scheme turns it into a working algorithm by mapping the interacting problem onto a fictitious system of *non-interacting* electrons that share the same density. These non-interacting orbitals satisfy a TISE-like equation,

$$
\left[-\tfrac{1}{2}\nabla^2 + v_\text{KS}(\mathbf r)\right]\phi_i(\mathbf r) = \epsilon_i\,\phi_i(\mathbf r),
$$

with an effective potential

$$
v_\text{KS}(\mathbf r) = V_\text{ext}(\mathbf r) \;+\; \int\frac{\rho(\mathbf r')}{\lvert \mathbf r - \mathbf r'\rvert}\,d\mathbf r' \;+\; v_\text{xc}[\rho](\mathbf r).
$$

The first two pieces are the external (nuclear) potential and the classical Hartree repulsion — exactly as in HF. The third, the **exchange–correlation potential** $v_\text{xc}$, absorbs everything quantum-mechanical that we do not know how to compute exactly, and must be approximated:

- **LDA (Local Density Approximation).** At each point $\mathbf r$, pretend the electron gas has the density $\rho(\mathbf r)$ *everywhere* and use the exchange-correlation energy per particle of that uniform electron gas. Surprisingly good for solids and decent for many molecules; tends to over-bind.
- **PBE (a Generalized Gradient Approximation).** Lets $v_\text{xc}$ depend on both $\rho$ and $\lvert\nabla\rho\rvert$, capturing the inhomogeneity of real molecules. Standard workhorse for chemistry and materials.

Like HF, the KS equations are solved self-consistently. The payoff is that the non-local exchange operator is replaced by a local multiplicative potential, which is much cheaper to evaluate on a grid.

---

## 4. Scaling to Heavy Elements: the Relativistic Imperative

For light atoms, standard non-relativistic physics works fine. But for heavy atoms like gold ($Z = 79$), electrons near the nucleus move so fast that Einstein's special relativity starts to matter.

### 4.1 Deriving the 137 heuristic for electron velocity

We can estimate the innermost electron's speed by balancing classical forces against Bohr's quantization rule. The centripetal force on a circular orbit must equal the Coulomb attraction to the nucleus:

$$
\frac{m_e\,v^2}{r} = \frac{Z\,e^2}{4\pi\varepsilon_0\,r^2}.
$$

Bohr further required the angular momentum of the innermost ($n=1$) shell to be quantized:

$$
m_e\,v\,r = \hbar
\quad\Longrightarrow\quad
r = \frac{\hbar}{m_e\,v}.
$$

Substituting this radius back into the force balance, one power of $m_e$ and one of $v$ cancel on both sides, leaving

$$
v = \frac{Z\,e^2}{4\pi\varepsilon_0\,\hbar}.
$$

The combination of constants on the right is, up to a factor of $c$, the **fine-structure constant**:

$$
\alpha = \frac{e^2}{4\pi\varepsilon_0\,\hbar\,c} \approx \frac{1}{137.036}.
$$

Putting $\alpha c$ back into the velocity expression gives the useful heuristic

$$
v = Z\,\alpha\,c \approx \frac{Z}{137}\,c.
$$

- For hydrogen ($Z=1$): $v \approx 0.007\,c$. Relativistic effects are negligible.
- For gold ($Z=79$): $v \approx 79/137\,c \approx 0.58\,c$. The inner electron moves at 58% the speed of light.

At these speeds the relativistic mass $m = \gamma\,m_e$ increases appreciably. Because quantum orbital radii are inversely proportional to mass ($a_0 \propto 1/m$), the $1s$ orbital **contracts** severely toward the nucleus, and quantum orthogonality then forces higher $s$ and $p$ orbitals to contract alongside it.

### 4.2 Why the Schrödinger equation fails

The standard Schrödinger kinetic-energy operator,

$$
\hat T = \frac{\hat p^2}{2m},
$$

is simply wrong at these speeds. The exact relativistic treatment is the **Dirac equation**, a $4\times 4$ matrix equation that accounts for both matter and antimatter components. If we shift the energy scale by $-mc^2$ so that it lines up with non-relativistic conventions, the time-independent Dirac equation reads

$$
\begin{pmatrix} V & c\,\vec{\sigma}\cdot\mathbf p \\ c\,\vec{\sigma}\cdot\mathbf p & V - 2mc^2 \end{pmatrix}
\begin{pmatrix} \psi_L \\ \psi_S \end{pmatrix} = E
\begin{pmatrix} \psi_L \\ \psi_S \end{pmatrix}.
$$

Here $\psi_L$ is the "large" component (the ordinary electron we care about), $\psi_S$ is the "small" component (antimatter/positronic contributions), and $\vec{\sigma}$ are the Pauli spin matrices. This is two coupled equations. Solving the second row for $\psi_S$:

$$
\psi_S = \frac{c}{E - V + 2mc^2}\,\vec{\sigma}\cdot\mathbf p\,\psi_L.
$$

Plugging this back into the first row gives an exact equation for the ordinary electron alone:

$$
\left[\,\vec{\sigma}\cdot\mathbf p\;\frac{c^2}{E - V + 2mc^2}\;\vec{\sigma}\cdot\mathbf p + V\,\right]\psi_L = E\,\psi_L.
$$

This is still computationally brutal, because the unknown $E$ appears on both sides — and inside a denominator.

### 4.3 Deriving the ZORA approximation

GATO simplifies this using the **Zeroth-Order Regular Approximation** (ZORA). In chemistry the binding energy $E$ is tiny compared to the rest-mass energy $2mc^2$, so we can drop $E$ from the denominator:

$$
\frac{c^2}{E - V + 2mc^2} \;\approx\; \frac{c^2}{2mc^2 - V(\mathbf r)}.
$$

Dropping spin-orbit coupling to keep the operator scalar, this collapses to the ZORA Hamiltonian:

$$
\hat H_\text{ZORA} = \mathbf p \cdot \frac{c^2}{2mc^2 - V(\mathbf r)}\,\mathbf p + V(\mathbf r).
$$

**Why this makes physical sense.**

- *Far from the nucleus.* $V(\mathbf r)$ is small, the denominator reduces to $2mc^2$, and the kinetic term recovers the standard $\hat p^2/(2m)$ of the Schrödinger equation.
- *Near the nucleus.* $V(\mathbf r)$ is a large negative number, so $2mc^2 - V$ grows large and the kinetic prefactor shrinks. The electron effectively becomes "heavier" inside the deep potential well, automatically contracting the orbital without ever invoking antimatter.

### 4.4 Recovering the non-relativistic limit

A reassuring sanity check: the ZORA Hamiltonian should reduce to the ordinary Schrödinger kinetic operator in the regime where first-course quantum mechanics is valid. "Slow speeds" or "non-relativistic limit" here means the energies involved (in particular the potential $V(\mathbf r)$) are much smaller than the electron's rest-mass energy $E_\text{rest} = mc^2$.

**Step 1 — isolate the kinetic prefactor.** Focus on the bracketed operator sitting between the two $\mathbf p$ factors in $\hat H_\text{ZORA}$:

$$
\text{Operator} = \frac{c^2}{2mc^2 - V(\mathbf r)}.
$$

**Step 2 — algebraic manipulation.** To compare this with the familiar $1/(2m)$ of the Schrödinger equation, pull $2mc^2$ out of the denominator:

$$
\frac{c^2}{2mc^2\left(1 - \dfrac{V(\mathbf r)}{2mc^2}\right)}.
$$

The $c^2$ in numerator and denominator cancel, leaving

$$
\frac{1}{2m}\cdot\frac{1}{1 - \dfrac{V(\mathbf r)}{2mc^2}}.
$$

**Step 3 — Taylor expansion.** In the non-relativistic regime, $V(\mathbf r)/(2mc^2)$ is tiny: $mc^2 \approx 511{,}000$ eV, while typical chemical potentials are a few eV. Using the geometric series $1/(1-x) \approx 1 + x + x^2 + \dots$ with $x = V(\mathbf r)/(2mc^2)$:

$$
\frac{1}{2m}\left(1 + \frac{V(\mathbf r)}{2mc^2} + \left(\frac{V(\mathbf r)}{2mc^2}\right)^2 + \cdots\right).
$$

**Step 4 — the zeroth-order term.** Keeping only the leading term (the *zeroth* order, hence the name *Zeroth-Order Regular Approximation*) drops everything proportional to $V/2mc^2$, giving

$$
\text{Operator} \;\approx\; \frac{1}{2m}.
$$

Plugging back into the Hamiltonian recovers the familiar Schrödinger form:

$$
\hat H \;\approx\; \mathbf p\cdot\frac{1}{2m}\,\mathbf p + V(\mathbf r) = \frac{\hat p^2}{2m} + V(\mathbf r).
$$

**Why "regular"?** The virtue of ZORA isn't just that it reduces to Schrödinger at low speeds — it stays well-behaved when $V$ is large. In other relativistic expansions (e.g. the Pauli Hamiltonian), the expansion variable is $V/(2mc^2)$ itself, and the series diverges near a nucleus where $V \to -\infty$, making the kinetic energy blow up. ZORA keeps $V(\mathbf r)$ *inside* the denominator: as an electron dives into a deep well and $V$ becomes a huge negative number, $2mc^2 - V$ only grows, damping the kinetic prefactor and preventing pathological behaviour near the nucleus.

In short: ZORA is a clean middle ground — it gives standard Schrödinger physics for valence electrons while automatically switching on the relativistic heavy lifting for core electrons.

### 4.5 Real-world consequences

- **The color of gold.** Relativistic contraction pulls the $6s$ orbital closer to the nucleus and lowers its energy, while the $5d$ orbitals are pushed up by additional shielding. The resulting $5d\to 6s$ absorption transition shifts from the ultraviolet (where silver absorbs) down into the blue visible range. Because gold absorbs blue, it reflects the opposite — yellow.
- **Liquid mercury.** The same contraction stabilizes mercury's $6s^2$ valence electrons. Held tightly to the nucleus, they refuse to share into strong lattice bonds with neighboring atoms, leaving mercury a liquid at room temperature.

---

## 5. Geometry Optimization and Forces

The ultimate application of these theoretical models is predicting real-world structural chemistry: *where do the nuclei actually sit*?

Within the **Born–Oppenheimer approximation**, nuclei move on the potential-energy surface traced out by the electronic ground-state energy $E_\text{elec}(\mathbf R_1,\dots,\mathbf R_{N_n})$ as the nuclei are held fixed. Equilibrium geometries are the minima of this surface; forces on the nuclei are minus its gradient.

### 5.1 The Hellmann–Feynman theorem

Computing $\partial E/\partial \mathbf R_A$ looks expensive — in principle the orbitals themselves depend on the nuclear positions. The **Hellmann–Feynman theorem** shows that, for an exact ground state, the wavefunction derivatives cancel and the force on nucleus $A$ reduces to the expectation value of $\partial \hat H/\partial \mathbf R_A$:

$$
\mathbf F_A = -\frac{\partial E}{\partial \mathbf R_A}
= -\left\langle\Psi\left\lvert\frac{\partial \hat H}{\partial \mathbf R_A}\right\rvert\Psi\right\rangle.
$$

Only the external nuclear–electron attraction and the classical nucleus–nucleus repulsion depend explicitly on $\mathbf R_A$, so the force becomes a classical-looking electrostatic integral of the converged electron density $\rho(\mathbf r)$ against the nuclear charges — nothing more exotic. Physically: once the electron cloud is equilibrated, it pulls on the positive nuclei like a diffuse charged fluid would.

### 5.2 Descending the energy surface

In GATO, nuclear positions are simply additional differentiable parameters in the JAX computation graph. A single call to `jax.grad(total_energy)` returns the forces on every nucleus by automatic differentiation — equivalent to the Hellmann–Feynman result when the electronic problem is fully converged. Feeding those forces to a gradient-based optimiser (Adam, L-BFGS) iteratively "pushes" the nuclei downhill,

$$
\mathbf R_A^{(k+1)} = \mathbf R_A^{(k)} - \eta\,\frac{\partial E}{\partial \mathbf R_A^{(k)}},
$$

until the total energy is stationary and the forces fall below a tolerance. This is how the ab-initio framework transitions from abstract wave mechanics to the observable, predictable geometric reality of molecules like $\text{H}_2\text{O}$ — its $0.96\,\text{\AA}$ O–H bond and $104.5^{\circ}$ bond angle emerge from minimising an electron-cloud energy, not from any empirical input.
