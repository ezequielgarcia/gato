"""Generic alternating geometry optimization for Born–Oppenheimer surfaces.

Both Phase 2 (H₂⁺ on imaginary-time) and Phase 4 (H₂O on RHF + HGH PPs)
relax nuclei the same way: solve the electronic problem at fixed geometry,
take an Adam step on positions using `jax.grad` of the BO energy at fixed
electrons (Hellmann–Feynman), recenter to remove translational drift, and
repeat. This module owns that skeleton so both phases share one
implementation.

The skeleton is generic over the "point" type the caller's electronic
solver returns. The only requirement is that the point exposes a
`.nuclei: Nuclei` attribute — everything else (wavefunction, energy
decomposition, orbital count) is opaque to this layer.
"""
from __future__ import annotations

from typing import Callable, TypeVar

import jax
import optax

from .geometry import Nuclei, recenter


Point = TypeVar("Point")


def optimize_geometry_alternating(
    initial_nuclei: Nuclei,
    *,
    solve_at_geometry: Callable[[Nuclei, "Point | None"], Point],
    bo_grad: Callable[[Point], jax.Array],
    geom_steps: int,
    learning_rate: float = 0.05,
    on_step: Callable[[int, Point], None] | None = None,
) -> tuple[Point, list[Point]]:
    """Alternating SCF + nuclear gradient descent with Adam.

    Parameters
    ----------
    initial_nuclei : starting geometry. Will be recentered at the
        charge-weighted COM before the first solve.
    solve_at_geometry : callable ``(nuclei, prev_point) -> point``. The
        previous point is passed in so callers can warm-start the
        electronic solve from the prior orbitals; pass ``None`` for the
        very first call.
    bo_grad : callable ``point -> ∂E/∂R`` (shape (K, 3)). Typically
        ``jax.grad(bo_energy, argnums=0)`` evaluated at the current
        point's wavefunction.
    geom_steps : number of nuclear gradient steps to take.
    learning_rate : Adam step size in Bohr per gradient unit.
    on_step : optional observer called after each completed nuclear
        update with ``(step_index, new_point)`` — handy for trajectory
        bookkeeping and progress printing.

    Returns
    -------
    final_point, trajectory : the last point and the list of every
    intermediate point including the initial solve.
    """
    nuclei = recenter(initial_nuclei)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nuclei.positions)

    current = solve_at_geometry(nuclei, None)
    trajectory: list[Point] = [current]

    for step in range(geom_steps):
        grad_R = bo_grad(current)
        updates, opt_state = optimizer.update(grad_R, opt_state)
        new_positions = optax.apply_updates(current.nuclei.positions, updates)
        nuclei = recenter(Nuclei(new_positions, current.nuclei.charges))
        current = solve_at_geometry(nuclei, current)
        if on_step is not None:
            on_step(step, current)
        trajectory.append(current)

    return current, trajectory
