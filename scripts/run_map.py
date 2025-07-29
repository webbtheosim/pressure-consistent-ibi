import numpy as np
from ibisimulation.prep_cg import CGmapper, generate_topology
import ibisimulation.tools_lammps as tl_lmp
import os
# import pdb

# n_cpus = 16
n_cpus = int(os.getenv("SLURM_CPUS_ON_NODE", default=1))
RDF_cutoff = 15
RDF_delta_r = 0.1
AA_file = './equil_retype.data'
AA_dump = './dump.xyz'
mapping_file = 'cg_mapping.json'
freq = 1

lmp = tl_lmp.read_lammps_full(AA_file)
natom_per_mol = np.sum(lmp.atom_info[:,1]==1)

mapping_list = [
    {'cg_type': 1, 'aa_type': np.arange(1,natom_per_mol+1), 'num_H': 0, 'charge': 0.0},
    ]

cg_bond_list = [
    ]

cg_angle_list = [
    ]

# generate a CG mapping file for the system
generate_topology(AA_file, mapping_list, cg_bond_list, cg_angle_list, mapping_file)

cg_map = CGmapper(AA_file, AA_dump, freq=freq, mapping_file=mapping_file, n_cpus=n_cpus)
