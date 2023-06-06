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
import numpy as np
from datetime import datetime

from .qe_utils import (velocity,
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

current_datetime = datetime.now()
# Format the datetime object as a string
formatted_datetime = current_datetime.strftime("%m-%d_%H:%M")
# Print the formatted string
print("STARTING JOB AT:", formatted_datetime)

#########################################################################
# RUN_RELAXATION
#    - True: run a relaxation step (can take several hours)
#    - False: don't run a relaxation step, use an existing relaxed crystal
# An existing relaxed crystal exists in relax_data/relax_Hf5Nb2Ta10Zr5.out
#########################################################################
RUN_RELAXATION = False

#########################################################################
# VACUUM
#    - Amount of vacuum, in Angstroms, to place on either side of the system
#########################################################################
VACUUM = 2.0

#########################################################################
# DEUTERIUM_MASS_AMU
#    - Deuterium mass in AMU
#########################################################################
DEUTERIUM_MASS_AMU = 2.014

#########################################################################
# INITIAL_DISTANCE_A
#    - Initial distance between D atom and surface of slab, in Angstroms
#########################################################################
INITIAL_DISTANCE_A = 1.5

#########################################################################
# AXIS
#    - One of { 'x', 'y', 'z' }: axis along which the D atom should travel
#########################################################################
AXIS = 'x'

#########################################################################
# N_STEPS
#    - Number of steps to run MD for
N_STEPS = 15

#########################################################################
# INITIAL_EV
#    - Initial eV to impart on D atom
#########################################################################
INITIAL_EV = 1000

#########################################################################
# DT
#    - dt, in AU
#########################################################################
DT = 0.2 * round(FS_TO_AU) # 0.2fs

# Run relaxation, if needed
if RUN_RELAXATION:
    slab = import_vasp('input/HfNbTaZr_8.vasp', truncate=False)
    slab.center(vacuum=1)
    relax_output_filename = relax(slab)
    
# Create our slab
slab = output_to_atoms(relax_output_filename if RUN_RELAXATION else 'relax_data/relax_Hf5Nb2Ta10Zr5.out') # This slab is the result of relaxing a 22-atom crystal
atoms = deepcopy(slab)
atoms.center(vacuum=VACUUM, axis=2)
atoms = pin_bottom_layers(atoms, nlayers=2, axis=AXIS)

# Place the D atom in the center of the slab, `INITIAL_DISTANCE_A` Angstroms away
DEUTERIUM_XYZ = get_D_position(atoms, INITIAL_DISTANCE_A=INITIAL_DISTANCE_A, axis=AXIS)
deuterium = Atom('H', mass=DEUTERIUM_MASS_AMU, position=DEUTERIUM_XYZ)
atoms.append(deuterium)

# Expand unit cell so that the D atom fits
existing_cell = atoms.get_cell()
atoms.set_cell(np.array([ # TODO this code is dogshit, clean it up
    existing_cell[0][0] + (2 * INITIAL_DISTANCE_A if AXIS == 'x' else 0),
    existing_cell[1][1] + (2 * INITIAL_DISTANCE_A if AXIS == 'y' else 0),
    existing_cell[2][2] + (2 * INITIAL_DISTANCE_A if AXIS == 'z' else 0)]
))

# preview(atoms)
output_filename = md(atoms, nsteps=N_STEPS, dt=DT, AXIS=AXIS, initial_eV=INITIAL_EV, suffix='relaxed_slab_{}'.format(current_datetime))

# We have to do this bit to get `ase` to be able to read the file in again
# TODO: file bugfix PR in ase
sanitize(output_filename)

end_time = datetime.now()

# Calculate the difference between the start and end time
time_difference = end_time - current_datetime

# Extract the hours, minutes, and seconds from the time difference
hours = time_difference.seconds // 3600
minutes = (time_difference.seconds % 3600) // 60
seconds = time_difference.seconds % 60

# Output the elapsed time
print("Elapsed time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))