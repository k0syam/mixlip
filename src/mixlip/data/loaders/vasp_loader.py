"""VASP output loaders (OUTCAR / vasprun.xml) via pymatgen."""

from __future__ import annotations

from pathlib import Path

from mixlip.data.schema import AtomicSample


def load_outcar(path: Path) -> list[AtomicSample]:
    """Parse a VASP OUTCAR file and return one sample per ionic step."""
    from pymatgen.io.vasp.outputs import Outcar, Vasprun
    from pymatgen.io.ase import AseAtomsAdaptor

    # Prefer vasprun.xml in the same directory if available
    vasprun_path = path.parent / "vasprun.xml"
    if vasprun_path.exists():
        return load_vasprun(vasprun_path)

    outcar = Outcar(str(path))
    samples = []

    # Outcar ionic step data
    for i, step in enumerate(outcar.ionic_steps):
        structure = step.get("structure")
        if structure is None:
            continue
        forces = step.get("forces")
        stress = step.get("stress")  # kbar → eV/Å³
        energy = step.get("e_fr_energy") or step.get("e_wo_entrp")

        if stress is not None:
            import numpy as np

            stress_arr = np.array(stress)
            # VASP stress in kbar, convert to eV/Å³
            # 1 kbar = 0.00062415091 eV/Å³
            if stress_arr.shape == (3, 3):
                stress_arr = stress_arr * 0.00062415091
            # Convert to Voigt
            stress_voigt = np.array(
                [
                    stress_arr[0, 0],
                    stress_arr[1, 1],
                    stress_arr[2, 2],
                    stress_arr[1, 2],
                    stress_arr[0, 2],
                    stress_arr[0, 1],
                ]
            )
        else:
            stress_voigt = None

        samples.append(
            AtomicSample(
                structure=structure,
                energy=float(energy) if energy is not None else None,
                forces=forces,
                stress=stress_voigt,
                source="vasp_outcar",
                metadata={"ionic_step": i, "file": str(path)},
            )
        )
    return samples


def load_vasprun(path: Path) -> list[AtomicSample]:
    """Parse a VASP vasprun.xml file and return one sample per ionic step."""
    import numpy as np
    from pymatgen.io.vasp.outputs import Vasprun

    vasprun = Vasprun(str(path), parse_dos=False, parse_eigen=False)
    samples = []

    for i, (struct, forces, stress, energy) in enumerate(
        zip(
            vasprun.ionic_steps_structures,
            vasprun.ionic_steps_forces,
            vasprun.ionic_steps_stress,
            vasprun.ionic_steps_e_fr_energy,
        )
    ):
        # stress in kbar, convert to eV/Å³ Voigt
        stress_arr = np.array(stress) * 0.00062415091
        stress_voigt = np.array(
            [
                stress_arr[0, 0],
                stress_arr[1, 1],
                stress_arr[2, 2],
                stress_arr[1, 2],
                stress_arr[0, 2],
                stress_arr[0, 1],
            ]
        )
        samples.append(
            AtomicSample(
                structure=struct,
                energy=float(energy),
                forces=np.array(forces),
                stress=stress_voigt,
                source="vasp_vasprun",
                metadata={"ionic_step": i, "file": str(path)},
            )
        )
    return samples
