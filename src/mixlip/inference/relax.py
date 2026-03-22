"""Structure relaxation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ase
    from mixlip.calculators.base import MixLIPCalculator


def relax_structure(
    atoms: ase.Atoms,
    calculator: MixLIPCalculator,
    fmax: float = 0.05,
    max_steps: int = 500,
    optimizer: str = "FIRE",
    relax_cell: bool = True,
    pressure_gpa: float = 0.0,
    trajectory_path: str | None = None,
) -> tuple[ase.Atoms, dict]:
    """Relax an atomic structure using an MLIP calculator.

    Parameters
    ----------
    atoms:
        Input structure (modified in place; a copy is made internally).
    calculator:
        Any MixLIPCalculator (used as ASE calculator).
    fmax:
        Force convergence threshold in eV/Å.
    max_steps:
        Maximum number of optimizer steps.
    optimizer:
        ASE optimizer name: "FIRE", "BFGS", "LBFGS".
    relax_cell:
        If True, also relax the unit cell (requires stress support).
    pressure_gpa:
        External pressure in GPa (applied when relax_cell=True).
    trajectory_path:
        If given, write trajectory to this extxyz file.

    Returns
    -------
    relaxed_atoms, info_dict
    """
    import ase
    import ase.optimize
    from ase.filters import ExpCellFilter, FrechetCellFilter
    from ase.constraints import StrainFilter

    atoms = atoms.copy()
    atoms.calc = calculator.ase_calculator

    optimizer_cls = {
        "FIRE": ase.optimize.FIRE,
        "BFGS": ase.optimize.BFGS,
        "LBFGS": ase.optimize.LBFGS,
    }.get(optimizer, ase.optimize.FIRE)

    if relax_cell:
        try:
            filtered = FrechetCellFilter(atoms, scalar_pressure=pressure_gpa * 1e-1)
        except ImportError:
            filtered = ExpCellFilter(atoms, scalar_pressure=pressure_gpa * 1e-1)
    else:
        filtered = atoms

    opt_kwargs: dict = {}
    if trajectory_path:
        opt_kwargs["trajectory"] = trajectory_path

    opt = optimizer_cls(filtered, **opt_kwargs)
    converged = opt.run(fmax=fmax, steps=max_steps)

    info = {
        "converged": converged,
        "n_steps": opt.get_number_of_steps(),
        "final_energy_eV": float(atoms.get_potential_energy()),
        "final_fmax_eV_per_A": float(atoms.get_forces().max()),
    }
    return atoms, info
