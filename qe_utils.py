from ase import Atom, Atoms
from ase.calculators.espresso import Espresso
from ase.constraints import FixAtoms
from ase.io.cif import read_cif
from ase.io.espresso import read_espresso_out, write_espresso_in
from ase.io.vasp import read_vasp
from ase.visualize import view
from ase.visualize.plot import plot_atoms

from copy import deepcopy
from enum import Enum

from matplotlib import pyplot as plt
import math
import numpy as np

import os
import subprocess
import re
import shutil

# File/directory constants
PSEUDO_DIR = 'q-e/pseudo/'
PSEUDOPOTENTIALS = {
    'Ta': 'ta_pbe_v1.uspp.F.UPF',
    'Nb': 'nb_pbe_v1.uspp.F.UPF',
    'Hf': 'hf_pbe_v1.uspp.F.UPF',
    'Zr': 'zr_pbe_v1.uspp.F.UPF',
    'H': 'h_pbe_v1.4.uspp.F.UPF'
}
OUTDIR = 'outdir'

class Calculation(Enum):
    RELAX = 'relax'
    VC_RELAX = 'vc-relax'
    MD = 'md'

class FileType(Enum):
    INPUT = 'in'
    OUTPUT = 'out'


def get_qe_filename(atoms, calc, filetype, suffix=''):
    """
    Returns a filename for a QE input/output file given an ase.Atoms object
    e.g. relax_Hf16Ta12Nb20Zr13H.in
    """
    assert isinstance(calc, Calculation)
    assert isinstance(filetype, FileType)
    _suffix = '_' + suffix if suffix else ''
    return f'{calc.value}_{atoms.symbols}{_suffix}.{filetype.value}'

def output_to_atoms(output_filename):
    """
    Reads QE atomic positions output into an ase.Atoms object
    """
    with open(output_filename, 'r') as f:
        generator = read_espresso_out(f, index=slice(None))
        atoms = [g for g in generator]
        atoms = atoms[-1]
    return atoms

def run(input_filename, output_filename, ncores):
    """
    Runs a QE operation. Assumes input file already exists
    """
    with open(output_filename, 'w') as f:
        p = subprocess.call(['mpirun', '/burg/opt/QE/7.2/bin/pw.x', '-inp', input_filename, '--use-hwthread-cpus'], stdout=f)
        p.wait()

def velocity(atomic_mass, energy, normal_angle_deg=0, polar_angle_deg=0, axis='x'):
    """
    Computes the velocity of an atom in Hartree atomic units, given the atomic mass in atomic units and the energy in electronvolts.

    Args:
    atomic_mass: The atomic mass of the atom in atomic units.
    energy: The energy of the atom in electronvolts.

    Returns:
    The velocity of the atom in Hartree atomic units.
    """
    if axis != 'x':
        raise NotImplementedError

    # Convert the atomic mass to kilograms.
    atomic_mass_kg = atomic_mass * 1.660539040e-27

    # Convert the energy to joules.
    energy_joules = energy * 1.60217662e-19

    # Calculate the velocity.
    velocity_ms = math.sqrt((2 * energy_joules) / atomic_mass_kg)

    # Convert the velocity to Hartree atomic units.
    velocity_ha = velocity_ms / 2.187691e6
    
    normal_angle_rad, polar_angle_rad = np.radians(normal_angle_deg), np.radians(polar_angle_deg)
    planar_magnitude = velocity_ha * np.sin(normal_angle_rad)
    vx = velocity_ha * np.cos(normal_angle_rad)
    vy = planar_magnitude * np.cos(polar_angle_rad)
    vz = planar_magnitude * np.sin(polar_angle_rad)
    
    return (-vx, vy, vz) # negate vx so that particle approaches the slab

assert velocity(1., 100, 0, 0) == velocity(1., 100, 0, 360)
assert np.all(np.isclose(
    velocity(1., 100, 45, 0),
    velocity(1., 100, 45, 360)
))


def preview(atoms):
    """
    Previews an ase.Atoms object.
    
        y
        |
        |_____ z
       / 
      /
     x

    """
    RADIUS = 0.1
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), nrows=1, ncols=2)
    plot_atoms(atoms, ax1, radii=RADIUS, rotation=('0x,0y,0z'))
    plot_atoms(atoms, ax2, radii=RADIUS, rotation=('0x,90y,0z'))

def sanitize(filename):
    """
    Hack around a bug in ase.io: 
    
        File ~/git/md/venv/lib/python3.11/site-packages/ase/io/espresso.py:264, in <listcomp>(.0)
        261 for magmoms_index in indexes[_PW_MAGMOM]:
        262     if image_index < magmoms_index < next_index:
        263         magmoms = [
    --> 264             float(mag_line.split()[5]) for mag_line
        265             in pwo_lines[magmoms_index + 1:
        266                          magmoms_index + 1 + len(structure)]]
        268 # Fermi level / highest occupied level
        269 efermi = None

    ValueError: could not convert string to float: 'magn='
    """
    # Copy and rename the old file
    backup_filename = f'{filename}.old'
    shutil.copyfile(filename, backup_filename)
    print(f'Preserving existing job output in {backup_filename}') 
    with open(filename, 'r') as f:
        content = f.read()
    
    content = re.sub('magn=', '', content)
    with open(filename, 'w') as f:
        f.write(content)

def import_vasp(filename: str, truncate=False):
    """
    Helper to import a VASP/POSCAR file into an ase.Atoms object
    
    Args:
        filename: .vasp/.poscar(?) filename
        truncate: If True, truncate the input atomic positions into a smaller slab (for faster calculations later)
    """
    with open(filename, 'r') as f:
        generator = read_vasp(f)
        atoms = [a for a in generator]
    
    if truncate:
        Y_MAX, X_MAX = 5.0, 8.0
        atoms = [a for a in atoms if a.position[2] < X_MAX] # Fixed threshold
        atoms = [a for a in atoms if a.position[1] < Y_MAX]
    slab = Atoms(atoms)
    slab.set_positions(slab.positions + np.min(slab.positions) + 0.2)

    # Leave some room in the simulation box to avoid this error:
    # https://www.researchgate.net/post/what_should_I_do_with_this_error_in_quantum_espressoError_in_routine_check_atoms_1_atoms_1_and_2_differ_by_lattice_vector-1_0_1
    BUFFER = 0.5
    CELL_DIMS = np.max(slab.positions, axis=0) + np.array([BUFFER, BUFFER, BUFFER])
    
    slab.set_cell(CELL_DIMS.tolist())
    slab.set_pbc(False)
    return slab

def relax(atoms, vc=False):
    """
    Runs a QE relax operation. Can take a few hours.
    
    Args:
        atoms: ase.Atoms object
        vc: Run variable cell relax if True; regular relax otherwise
    """
    op = Calculation.VC_RELAX if vc else Calculation.RELAX
    input_filename = get_qe_filename(atoms, op, FileType.INPUT)
    output_filename = get_qe_filename(atoms, op, FileType.OUTPUT)
    
    input_data = {
        'control': {
            'calculation': op.value,
            'pseudo_dir': PSEUDO_DIR,
            'etot_conv_thr': 1e-3, # 1e-5,
            'forc_conv_thr': 1e-2, # 1e-4,
            'outdir': OUTDIR
        },
        'electrons': {
            'conv_thr': 1e-6, # 1e-8
        },
        'k_points': 'gamma',
        'system': {
            # Try a higher ecut to avoid varying scf accuracy: https://lists.quantum-espresso.org/pipermail/users/2016-January/034157.html
            'ecutwfc': 120,
            'ecutrho': 960,
            'occupations': 'smearing', # Need smearing params to avoid `Error in routine electrons (1): charge is wrong: smearing is needed`
            'smearing': 'gaussian',
            'degauss': 0.022,
            #'ibrav': 3, 
            #'alat': 3.34
        }
    }

    # NOTE: kpts = 3 causes failure to converge after 100 iterations; kpts in {1, 4} seems to work?
    k = 1
    with open(input_filename, 'w') as f:
      write_espresso_in(
          f,
          atoms,
          input_data=input_data,
          pseudopotentials=PSEUDOPOTENTIALS,
          tstress=False,
          tprnfor=False,
          kpts=None,
        )
    
    run(input_filename, output_filename)
    return output_filename

def get_D_position(atoms, initial_distance_a=5.0, axis='z', normal_angle_deg=0, polar_angle_deg=0):
    """
    Get a position at which to place a single D atom.
    
    Args:
        atoms: ase.Atoms object
        initial_distance_a: initial distance between top of slab & D atom, in Angstroms
        normal_angle_deg: Angle from slab surface normal, in degrees
        polar_angle_deg: Angle on face of slab. y-axis (up) is 0deg
    """
    if axis != 'x':
        raise NotImplementedError

    normal_angle_rad, polar_angle_rad = np.radians(normal_angle_deg), np.radians(polar_angle_deg)
    radius = initial_distance_a * np.sin(normal_angle_rad)
    dx = initial_distance_a * np.cos(normal_angle_rad)
    dy = radius * np.cos(polar_angle_rad)
    dz = radius * np.sin(polar_angle_rad)
    
    mean_xyz = np.mean(atoms.positions, axis=0)
    top_x = np.max(atoms.positions[:, 0]) # largest cartesian x coordinate (top of slab)
    return np.array([top_x + dx, mean_xyz[1] + dy, mean_xyz[2] + dz])

def consecutive(data, stepsize=1.0):
    return np.split(data, np.where(np.diff(data) >= stepsize)[0]+1)

def pin_bottom_layers(atoms, nlayers, axis='z'):
    """
    "Pin"s some inner layers, i.e. make the positions of atoms in that layer fixed during MD.
    Otherwise, when we send the particles to the slab with some momentum, the slab would move.
    
    Args:
        atoms: ase.Atoms object
        nlayers: The number of layers to fix/pin, starting from the bottom of the slab
    """
    if axis == 'y':
        ys = np.sort(np.unique(atoms.positions[:, 1]))
        max_y = consecutive(ys)[nlayers - 1][0]
        pinned_atoms = deepcopy(atoms)
        mask = [atom for atom in atoms if atom.position[1] <= max_y]
    if axis == 'x':
        xs = np.sort(np.unique(atoms.positions[:, 0]))
        max_x = consecutive(xs)[nlayers - 1][0]
        pinned_atoms = deepcopy(atoms)
        mask = [atom for atom in atoms if atom.position[0] <= max_x]  
    pinned_atoms.set_constraint(FixAtoms(mask=mask))
    return pinned_atoms

def md(atoms, nsteps, dt, AXIS="x", initial_eV=None, incident_angle_deg=0, polar_angle_deg=0, suffix=None, ncores=12):
    suffix = f'{nsteps}steps_{initial_eV}eV_incident{incident_angle_deg}_polar{polar_angle_deg}' + (f'_{suffix}' if suffix else '')
    input_filename = get_qe_filename(atoms, Calculation.MD, FileType.INPUT, suffix=suffix)
    output_filename = get_qe_filename(atoms, Calculation.MD, FileType.OUTPUT, suffix=suffix)
    
    # TODO: this is so hacky, but ase QE doesn't support ATOMIC_VELOCITIES(?) and the ATOMIC_VELOCITIES input itself isn't well-documented
    if initial_eV:
        atomic_velocities_str = 'ATOMIC_VELOCITIES\n'
        for el in atoms.get_chemical_symbols():
            if el != 'H':
                atomic_velocities_str += f'{el} 0.0 0.0 0.0\n'

        DEUTERIUM_MASS_AMU = 2.014
        vx_au, vy_au, vz_au = velocity(DEUTERIUM_MASS_AMU, initial_eV, normal_angle_deg=incident_angle_deg, polar_angle_deg=polar_angle_deg)
        if AXIS == 'x': # TODO this code is dogshit, clean it up
            format_str = 'H {:.5f} {:.5f} {:.5f}'
        else:
            raise NotImplementedError
        atomic_velocities_str += format_str.format(vx_au, vy_au, vz_au)
    
    input_data = {
        'control': {
            'calculation': Calculation.MD.value,
            'dt': dt,
            'nstep': nsteps,
            'pseudo_dir': PSEUDO_DIR,
            'etot_conv_thr': 1e-2, # 1e-4
            'forc_conv_thr': 1e-2, # 1e-5
            'outdir': OUTDIR
        },
        'electrons': {
            'conv_thr': 1e-3, #1e-6, # 1e-8
            'mixing_mode': 'TF',
            'mixing_beta': 0.7
        },
        'system': {
            'ecutwfc': 120,
            'occupations': 'smearing', # Need smearing params to avoid `Error in routine electrons (1): charge is wrong: smearing is needed`
            'smearing': 'gaussian',
            'degauss': 0.022,
            'nspin': 2
        },
        'ions': {
            'ion_temperature': 'initial',
            'tempw': 300,
            'ion_velocities': 'from_input'
        }
    }

    with open(input_filename, 'w') as f:
        write_espresso_in(
            f,
            atoms,
            input_data=input_data,
            pseudopotentials=PSEUDOPOTENTIALS,
            tstress=False,
            tprnfor=False,
            kpts=None, # gamma k-points
        )
        if initial_eV:
            print(f'Writing D initial velocity {initial_eV}eV (vx={vx_au}, vy={vy_au}, vz={vz_au} Hartree au)')
            f.write(atomic_velocities_str)

    run(input_filename, output_filename, ncores)
    return output_filename