from ase import Atom, Atoms
from ase.calculators.espresso import Espresso
from ase.constraints import FixAtoms
from ase.io.cif import read_cif
from ase.io.espresso import read_espresso_out, write_espresso_in
from ase.io.vasp import read_vasp
from ase.visualize import view
from ase.visualize.plot import plot_atoms

from copy import deepcopy
from matplotlib import pyplot as plt
import math
import numpy as np
import argparse
from datetime import datetime

from qe_utils import (velocity,
    import_vasp,
    output_to_atoms,
    relax,
    pin_bottom_layers,
    get_D_position,
    preview,
    md,
    sanitize)

# Numeric constants
# 1 picosecond = n Rydberg a.u.
PS_TO_AU = 1e-12 / (4.8378 * 1e-17)

# 1 femtosecond
FS_TO_AU = 1e-15 / (4.8378 * 1e-17)

RUN_RELAXATION = False
VACUUM = 2.0
DEUTERIUM_MASS_AMU = 2.014
INITIAL_DISTANCE_A = 1.6
AXIS = 'x'
N_STEPS = 20
DT = 0.2 * round(FS_TO_AU) # 0.2fs

def setup(incident_angle_deg, polar_angle_deg):
    # Run relaxation, if needed
    if RUN_RELAXATION:
        slab = import_vasp('input/HfNbTaZr_8.vasp', truncate=False)
        slab.center(vacuum=1)
        relax_output_filename = relax(slab)

    # Create our slab
    slab = output_to_atoms(relax_output_filename if RUN_RELAXATION else 'relax_data/relax_Hf5Nb2Ta10Zr5.out') # This slab is the result of relaxing a 22-atom crystal
    atoms = deepcopy(slab)
    atoms.center(vacuum=VACUUM, axis=2)
    atoms = pin_bottom_layers(atoms, nlayers=1, axis=AXIS)

    # Place the D atom in the center of the slab, `INITIAL_DISTANCE_A` Angstroms away
    DEUTERIUM_XYZ = get_D_position(atoms, initial_distance_a=INITIAL_DISTANCE_A, axis=AXIS, normal_angle_deg=incident_angle_deg, polar_angle_deg=polar_angle_deg)
    deuterium = Atom('H', mass=DEUTERIUM_MASS_AMU, position=DEUTERIUM_XYZ)
    atoms.append(deuterium)

    # Expand unit cell so that the D atom fits
    existing_cell = atoms.get_cell()
    atoms.set_cell(np.array([ # TODO this code is dogshit, clean it up
        existing_cell[0][0] + (2 * INITIAL_DISTANCE_A if AXIS == 'x' else 0),
        existing_cell[1][1] + (2 * INITIAL_DISTANCE_A if AXIS == 'y' else 0),
        existing_cell[2][2] + (2 * INITIAL_DISTANCE_A if AXIS == 'z' else 0)]
    ))

    return atoms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncores')
    parser.add_argument('--evs', help="Comma-delimited string of initial D energies in eV", type=str)
    parser.add_argument('--velocitymul', help='Multiplier (alat?) for velocity', type=float, default=1.0)
    args = parser.parse_args()

    ncores = int(args.ncores)
    eVs = [int(eV) for eV in args.evs.split(',')]
    velocity_multiplier = float(args.velocitymul)

    print(f'Using {args.ncores} cores, velocity multiplier {velocity_multiplier}')
    counter = 0

    for INCIDENT_ANGLE_DEG in [45]:
        for POLAR_ANGLE_DEG in range(0, 181, 30):
            if POLAR_ANGLE_DEG in {0, 180}:
                continue
            for INITIAL_EV in eVs: #[50, 100]:
                atoms = setup(INCIDENT_ANGLE_DEG, POLAR_ANGLE_DEG)
                print(f'{counter+1}: Running MD for incident angle={INCIDENT_ANGLE_DEG}deg, polar angle={POLAR_ANGLE_DEG}deg, D eV={INITIAL_EV}eV. {N_STEPS} steps, {DT} integration timestep, starting {INITIAL_DISTANCE_A}angstrom away')
                output_filename = md(
                    atoms,
                    nsteps=N_STEPS,
                    dt=DT,
                    initial_eV=INITIAL_EV,
                    incident_angle_deg=INCIDENT_ANGLE_DEG,
                    polar_angle_deg=POLAR_ANGLE_DEG,
                    is_cluster=False,
                    ncores=ncores,
                    velocity_multiplier=velocity_multiplier
                )
                sanitize(output_filename)
                print('-------------------------------------------------------')
