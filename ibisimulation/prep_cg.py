import numpy as np
import ibisimulation.tools_lammps as tl_lmp
import ibisimulation.tools_CG_polymer as tcp
import json
import pandas as pd
from joblib import Parallel, delayed
import scipy.constants as scc
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import pdb


def generate_topology(AA_file, mapping_list, cg_bond_list, cg_angle_list, mapping_file):
    """
    Generate a CG mapping file for the system in json format.
    Args:
        AA_file (str): The all-atom data file.
        mapping_list (list): A list of dictionaries, each dictionary contains the following keys:
            - cg_type: the CG bead type
            - aa_type: the all-atom bead type
            - charge: the CG bead charge
            - num_H: number of H atoms
        cg_bond_list (list): A list of dictionaries, each dictionary contains the following keys:
            - id: the bond ID
            - type: the bond type
            - beads: a list of bead IDs that are bonded
        cg_angle_list (list): A list of dictionaries, each dictionary contains the following keys:
            - id: the angle ID
            - type: the angle type
            - beads: a list of bead IDs that are in the angle
        mapping_file (str): The output CG mapping file in json format.
    """

    beads, bonds, angles = [], [], []

    # Combine into a single dictionary
    cg_data = {
        "beads": beads,
        "bonds": bonds,
        "angles": angles
    }

    lmp_aa = tl_lmp.read_lammps_full(AA_file)
    lmp_aa.atom_info = lmp_aa.atom_info[np.argsort(lmp_aa.atom_info[:, 0])]

    mol_id = lmp_aa.atom_info[:, 1]
    mol_id_unique = np.unique(mol_id)

    bead_id = 0
    for mol in mol_id_unique:
        atom_id = lmp_aa.atom_info[:, 0][mol_id == mol]
        atom_type = lmp_aa.atom_info[:, 2][mol_id == mol]
        for mapping in mapping_list:
            bead_id += 1
            cg_type = mapping['cg_type']
            aa_id = []
            aa_mass = []
            for iatom in range(len(atom_id)):
                if atom_type[iatom] in mapping['aa_type']:
                    aa_id.append(int(atom_id[iatom]))
            # cg_mass = mapping['mass']
            cg_charge = mapping['charge']
            num_H = mapping['num_H']
            # aa_id = np.array(aa_id).astype(int)
            bead = {'id': bead_id,
                    'type': cg_type,
                    'charge': cg_charge,
                    'aa_ids': aa_id,
                    'mol_id': mol,
                    'num_H': num_H}
            cg_data["beads"].append(bead)

    # bead 1-2, 2-3, ... 6-7 are bonded
    for i in range(0, len(cg_data["beads"]), len(mapping_list)):
        for j in range(len(cg_bond_list)):
            bead0 = cg_bond_list[j]['bead'][0]
            bead1 = cg_bond_list[j]['bead'][1]
            bond = {'id': i//len(mapping_list) + j + 1,
                    'type': cg_bond_list[j]['type'],
                    'beads': [i+bead0, i+bead1]}
            cg_data["bonds"].append(bond)

    # angle 1-2-3, 2-3-4, ... 5-6-7
    for i in range(0, len(cg_data["beads"]), len(mapping_list)):
        for j in range(len(cg_angle_list)):
            bead0 = cg_angle_list[j]['bead'][0]
            bead1 = cg_angle_list[j]['bead'][1]
            bead2 = cg_angle_list[j]['bead'][2]
            angle = {'id': i//len(mapping_list) + j + 1,
                     'type': cg_angle_list[j]['type'],
                     'beads': [i+bead0, i+bead1, i+bead2]}
            cg_data["angles"].append(angle)

    # Write to a JSON file
    with open(mapping_file, "w") as f:
        json.dump(cg_data, f, indent=4)


class CGmapper:
    """
    The class reads the all-atom data and dump files, applies a mapping to convert them into CG format, and writes the CG data and dump files.
    """
    def __init__(self, AA_file: str, AA_dump:str, mapping_file: str,
                 freq: int = 1, n_cpus: int = 1):
        """
        Args:
            AA_file (str): The all-atom data file.
            AA_dump (str): The all-atom dump file.
            mapping_file (str): The CG mapping file in json format.
            freq (int): The frequency of the dump file.
            n_cpus (int): The number of CPUs to use for parallel processing.

        returns:
            write the CG data file and dump file
        """
        self.AA_file = AA_file
        self.AA_dump = AA_dump
        self.mapping_file = mapping_file
        self.freq = freq
        self.n_cpus = n_cpus

        self._load_aa_data()
        self._cg_mapping()
        self._cg_mapping_dump()

    def _load_aa_data(self):
        # read the lammps data file
        self.lmp_aa = tl_lmp.read_lammps_full(self.AA_file)
        self.lmp_aa.atom_info = self.lmp_aa.atom_info[np.argsort(self.lmp_aa.atom_info[:, 0])]
        self.box_size = self.lmp_aa.x[1] - self.lmp_aa.x[0]
        self.mass_aa = self.lmp_aa.mass[self.lmp_aa.atom_info[:, 2].astype(int)-1, 1].astype(float)
        self.total_mass = np.sum(self.mass_aa)

    def _cg_mapping(self):
        # read the json mapping file
        with open(self.mapping_file, 'r') as f:
            self.cg_mapping = json.load(f)
        
        # prepare basic of lmp_cg
        self.lmp_cg = tl_lmp.lammps()
        self.lmp_cg.natoms = len(self.cg_mapping['beads'])
        self.lmp_cg.natom_types = len(np.unique([bead['type'] for bead in self.cg_mapping['beads']]))
        self.lmp_cg.nbonds = len(self.cg_mapping['bonds'])
        self.lmp_cg.nbond_types = len(np.unique([bond['type'] for bond in self.cg_mapping['bonds']]))
        if self.cg_mapping['angles'] is not None:
            self.lmp_cg.nangles = len(self.cg_mapping['angles'])
            self.lmp_cg.nangle_types = len(np.unique([angle['type'] for angle in self.cg_mapping['angles']]))
        # mass_bead = 1*15.999 + 2 * 12.011 + 5*1.008
        self.lmp_cg.mass = []
        for i in range(self.lmp_cg.natom_types):
            idx = np.where(np.array([bead['type'] for bead in self.cg_mapping['beads']]) == i+1)[0][0]
            self.lmp_cg.mass.append([i+1, self.mass_aa[self.cg_mapping['beads'][int(idx)]['aa_ids']].sum() + 1.008*self.cg_mapping['beads'][int(idx)]['num_H']])
        self.lmp_cg.mass = np.array(self.lmp_cg.mass)

        self.lmp_cg.x = [0, self.box_size]
        self.lmp_cg.y = [0, self.box_size]
        self.lmp_cg.z = [0, self.box_size]

        # prepare CG atom info, particularly coord
        atom_info_cg = np.zeros((self.lmp_cg.natoms, 7))
        for i, bead in enumerate(self.cg_mapping['beads']):
            atom_info_cg[i, 0] = bead['id']
            atom_info_cg[i, 1] = bead['mol_id']
            atom_info_cg[i, 2] = bead['type']
            atom_info_cg[i, 3] = bead['charge']
            aa_id = bead['aa_ids']
            aa_coord = self.lmp_aa.atom_info[:, 4:7][np.isin(self.lmp_aa.atom_info[:, 0], bead['aa_ids'])]
            aa_mass = self.lmp_aa.mass[self.lmp_aa.atom_info[np.array(bead['aa_ids'])-1, 2].astype(int)-1, 1].astype(float)
            cg_coord = tcp.map_CG_one(aa_coord, self.box_size, aa_mass)
            cg_coord = cg_coord - np.array([self.lmp_aa.x[0], self.lmp_aa.y[0], self.lmp_aa.z[0]])
            # move them into 0, box_size based on PBC
            cg_coord = cg_coord - np.floor(cg_coord / self.box_size) * self.box_size
            atom_info_cg[i, 4:7] = cg_coord 

        # prepare bond_info_cg
        bond_info_cg = np.zeros((self.lmp_cg.nbonds, 4))
        for i, bond in enumerate(self.cg_mapping['bonds']):
            bond_info_cg[i, 0] = bond['id']
            bond_info_cg[i, 1] = bond['type']
            bond_info_cg[i, 2] = bond['beads'][0]
            bond_info_cg[i, 3] = bond['beads'][1]

        # prepare angle_info_cg
        if self.cg_mapping['angles'] is not None:
            angle_info_cg = np.zeros((self.lmp_cg.nangles, 5))
            for i, angle in enumerate(self.cg_mapping['angles']):
                angle_info_cg[i, 0] = angle['id']
                angle_info_cg[i, 1] = angle['type']
                angle_info_cg[i, 2] = angle['beads'][0]
                angle_info_cg[i, 3] = angle['beads'][1]
                angle_info_cg[i, 4] = angle['beads'][2]

        self.lmp_cg.atom_info = atom_info_cg
        self.lmp_cg.bond_info = bond_info_cg
        self.lmp_cg.angle_info = angle_info_cg
        self.lmp_cg.write('cg.data')

    def _cg_mapping_dump(self):

        self.CG_mapping_idx = [np.array(self.cg_mapping['beads'][i]['aa_ids']) for i in range(len(self.cg_mapping['beads']))]
        self.CG_type = [self.cg_mapping['beads'][i]['type'] for i in range(len(self.cg_mapping['beads']))]
        self.CG_molid = [self.cg_mapping['beads'][i]['mol_id'] for i in range(len(self.cg_mapping['beads']))]

        # convert all-atom dump into CG dump
        frames, t_list, L_list = tl_lmp.read_lammps_dump_custom(self.AA_dump, self.freq)

        args = [(frames[i], L_list[i]) for i in np.arange(0, len(frames))]
        results = Parallel(n_jobs=self.n_cpus)(delayed(self._process_frame)(*arg) for arg in args)

        density_list, CG_coord_list, box_size_list = [], [], []
        CGframe_list = []
        for result in results:
            if result is None:
                continue
            density, CG_coord, box_size, CGframe = result
            density_list.append(density)
            CG_coord_list.append(CG_coord)
            box_size_list.append(box_size)
            CGframe_list.append(CGframe)

        L_CG_list = []
        for L in box_size_list:
            L_CG = np.array([[0, L], [0, L], [0, L]])
            L_CG_list.append(L_CG)

        print(len(CG_coord_list), flush=True)
        density = np.mean(density_list)
        print('density:', density, flush=True)

        # write the CG dump file 
        tl_lmp.write_lammps_dump_custom('dump_CG.xyz', CGframe_list, t_list, L_CG_list)

    def _process_frame(self, frame, L):
        box_size = L[0][1] - L[0][0]
        coord_AA = frame.loc[:,'x':'z'].to_numpy() - np.array([L[0][0], L[1][0], L[2][0]])
        CG_coord = tcp.map_CG_coord(coord_AA, self.CG_mapping_idx, box_size, self.mass_aa)

        # write a new dataframe for the CG system, including atom id, type, mol id, x y z
        CGframe_list = []
        for i in np.arange(len(CG_coord)):
            CGframe_list.append([i+1, self.CG_type[i], self.CG_molid[i], CG_coord[i][0], CG_coord[i][1], CG_coord[i][2]])
        CGframe = pd.DataFrame(CGframe_list, columns=['id','type','mol','x','y','z'])

        # density 
        V = box_size**3
        density = self.total_mass/V*1e24/scc.Avogadro

        return density, CG_coord, box_size, CGframe


class LammpsAnalyzer:
    """
    A class to analyze LAMMPS trajectory files and compute structural properties.
    """
    def __init__(self, AA_file: str, lmp_mode: str, dump_file: str, freq: int = 10,
                 RDF_cutoff: float = 20, RDF_delta_r: float = 0.01,
                 bl_range: Optional[np.ndarray] = None, angle_range: Optional[np.ndarray] = None):
        """
        Initialize the LammpsAnalyzer with the given parameters.
        Modification: non-bonded PDF to intermolecular PDF
        Args:
            AA_file (str): Path to the LAMMPS data file.
            lmp_mode (str): Mode of LAMMPS data ('full' or 'atomic').
            dump_file (str): Path to the LAMMPS dump file.
            freq (int): Frequency of frames to analyze.
            RDF_cutoff (float): Cutoff distance for RDF calculation.
            RDF_delta_r (float): Bin size for RDF calculation.
            bl_range (Optional[np.ndarray]): Range for bond length histogram.
            angle_range (Optional[np.ndarray]): Range for angle histogram.
        """
        self.AA_file = AA_file
        self.lmp_mode = lmp_mode
        self.dump_file = dump_file
        self.freq = freq
        self.RDF_cutoff = RDF_cutoff
        self.RDF_delta_r = RDF_delta_r
        self.bl_range = bl_range if bl_range is not None else np.arange(0, 10, 0.1)
        self.angle_range = angle_range if angle_range is not None else np.arange(0, 181, 1)

        self._load_topology()
        self._load_trajectory()

    def _load_topology(self):
        self.lmp = tl_lmp.read_lammps_full(self.AA_file)
        self.lmp.atom_info = self.lmp.atom_info[np.argsort(self.lmp.atom_info[:, 0])]
        if self.lmp_mode == 'full':
            self.type_atom = self.lmp.atom_info[:, 2]
        elif self.lmp_mode == 'atomic':
            self.type_atom = self.lmp.atom_info[:, 1]
        if self.lmp.nbonds > 0:
            self.bond_atom_idx = self.lmp.bond_info[:, 2:4].astype(int) - 1
        if self.lmp.nangles > 0:
            self.angle_atoms = np.array(self.lmp.angle_info[:, 2:5] - 1, dtype=int)
        self.mass_atom = np.array([self.lmp.mass[int(self.type_atom[iatom - 1]) - 1, 1] for iatom in np.arange(1, len(self.type_atom) + 1)], dtype=float).reshape(-1, 1)
        self.total_mass = np.sum(self.mass_atom)
 
        self.bond_atom_idx_type = {}
        # we could do this based on bonds, but also based on whether they are from the same molecule
        # if self.lmp.nbonds > 0:
        #     for i in range(self.lmp.natom_types):
        #         for j in range(i, self.lmp.natom_types):
        #             self.bond_atom_idx_type[(i+1, j+1)] = []
        #             idx_type1 = np.where(self.type_atom == i+1)[0]
        #             idx_type2 = np.where(self.type_atom == j+1)[0]
        #
        #             for k, l in self.bond_atom_idx:
        #                 if self.type_atom[k] == i+1 and self.type_atom[l] == j+1:
        #                     self.bond_atom_idx_type[(i+1, j+1)].append([np.where(idx_type1==k)[0], np.where(idx_type2==l)[0]])
        #                 if self.type_atom[l] == i+1 and self.type_atom[k] == j+1:
        #                     self.bond_atom_idx_type[(i+1, j+1)].append([np.where(idx_type1==l)[0], np.where(idx_type2==k)[0]])
        self.compute_RDF_matrix = {}
        for i in range(self.lmp.natom_types):
            for j in range(self.lmp.natom_types):
                ntype1 = np.sum(self.type_atom == i+1)
                idx_type1 = np.where(self.type_atom == i+1)[0]
                ntype2 = np.sum(self.type_atom == j+1)
                idx_type2 = np.where(self.type_atom == j+1)[0]
                if self.lmp_mode == 'full':
                    mol_type1 = self.lmp.atom_info[idx_type1, 1]
                    mol_type2 = self.lmp.atom_info[idx_type2, 1]
                elif self.lmp_mode == 'atomic':
                    mol_type1 = self.lmp.atom_info[idx_type1, 0]
                    mol_type2 = self.lmp.atom_info[idx_type2, 0]
                else:
                    raise ValueError("Invalid lmp_mode. Choose 'full' or 'atomic'.")
                self.compute_RDF_matrix[(i+1, j+1)] = (mol_type1[:, None] != mol_type2[None, :]).astype(int)
        # else:
        #     for i in range(self.lmp.natom_types):
        #         for j in range(i, self.lmp.natom_types):
        #             self.bond_atom_idx_type[(i+1, j+1)] = []


    def _load_trajectory(self):
        self.frames, _, self.L_list = tl_lmp.read_lammps_dump_custom(self.dump_file, interval=self.freq)
        print(len(self.frames), flush=True)

    # def _run_with_timeout(func, *arg, timeout):
    #     with ThreadPoolExecutor(max_workers=1) as executor:
    #         future = executor.submit(func, *arg)
    #         try:
    #             return future.result(timeout=timeout)
    #         except TimeoutError:
    #             return "Timed out"

    def _process_frame(self, frame, L, id):
        try:
            begin = time.time()
            property = {}

            box_size = L[0][1] - L[0][0]
            property['box_size'] = box_size

            box = np.array([[box_size, 0, 0],
                            [0, box_size, 0],
                            [0, 0, box_size]])
            coord_AA = frame.loc[:, 'x':'z'].to_numpy()
            coord = {}
            for i in range(self.lmp.natom_types):
                coord[i + 1] = coord_AA[self.type_atom == i + 1, :]

            # density
            V = box_size ** 3
            density = self.total_mass / V * 1e24 / scc.Avogadro
            property['density'] = density

            # RDF
            property['RDF'] = {}
            property['Sq'] = {}
            for i in range(self.lmp.natom_types):
                for j in range(i, self.lmp.natom_types):
                    R, g, Q, S = tcp.pdf_sq_cross_mask(box, coord[i + 1], coord[j + 1], self.compute_RDF_matrix[(i+1, j+1)], r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r)
                    property['RDF'][(i + 1, j + 1)] = g
                    property['Sq'][(i + 1, j + 1)] = S

            # bond length
            if self.lmp.nbonds > 0:
                property['bl'] = {}
                for i in range(self.lmp.nbond_types):
                    bond_length = tcp.compute_bond_length(coord_AA, self.bond_atom_idx[self.lmp.bond_info[:, 1] == i + 1], box_size)
                    bl_hist, _ = np.histogram(bond_length, bins=self.bl_range, density=True)
                    property['bl'][i + 1] = bl_hist

            # angle
            if self.lmp.nangles > 0:
                property['angle'] = {}
                for i in range(self.lmp.nangle_types):
                    angle = tcp.compute_angle(coord_AA, self.angle_atoms[self.lmp.angle_info[:, 1] == i + 1], box_size)
                    angle_hist, _ = np.histogram(angle, bins=self.angle_range, density=True)
                    property['angle'][i + 1] = angle_hist
            end = time.time()
            print(f'Frame{id} Time:', end - begin, flush=True)
            return property

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def _batch_process_frame(self, frames, Ls, ids):
        try:
            begin = time.time()
            property_list = []
            for i, frame in enumerate(frames):
                property = {}
                L = Ls[i]
                id = ids[i]

                box_size = L[0][1] - L[0][0]
                property['box_size'] = box_size

                box = np.array([[box_size, 0, 0],
                                [0, box_size, 0],
                                [0, 0, box_size]])
                coord_AA = frame.loc[:, 'x':'z'].to_numpy()
                coord = {}
                for i in range(self.lmp.natom_types):
                    coord[i + 1] = coord_AA[self.type_atom == i + 1, :]

                # density
                V = box_size ** 3
                density = self.total_mass / V * 1e24 / scc.Avogadro
                property['density'] = density

                # RDF
                property['RDF'] = {}
                property['Sq'] = {}
                for i in range(self.lmp.natom_types):
                    for j in range(i, self.lmp.natom_types):
                        R, g, Q, S = tcp.pdf_sq_cross_mask(box, coord[i + 1], coord[j + 1], self.compute_RDF_matrix[(i+1, j+1)], r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r)
                        property['RDF'][(i + 1, j + 1)] = g
                        property['Sq'][(i + 1, j + 1)] = S
                property['Sq']['Q'] = Q

                # bond length
                if self.lmp.nbonds > 0:
                    property['bl'] = {}
                    for i in range(self.lmp.nbond_types):
                        bond_length = tcp.compute_bond_length(coord_AA, self.bond_atom_idx[self.lmp.bond_info[:, 1] == i + 1], box_size)
                        bl_hist, _ = np.histogram(bond_length, bins=self.bl_range, density=True)
                        property['bl'][i + 1] = bl_hist

                # angle
                if self.lmp.nangles > 0:
                    property['angle'] = {}
                    for i in range(self.lmp.nangle_types):
                        angle = tcp.compute_angle(coord_AA, self.angle_atoms[self.lmp.angle_info[:, 1] == i + 1], box_size)
                        angle_hist, _ = np.histogram(angle, bins=self.angle_range, density=True)
                        property['angle'][i + 1] = angle_hist
                end = time.time()
                print(f'Frame{id} Time:', end - begin, flush=True)
                property_list.append(property)
            return property_list

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def analyze(self, n_jobs: int = 8, batch_size: int = 10) -> dict:
        """
        Analyze the trajectory and compute structural properties.
        Args:
            n_jobs (int): Number of parallel jobs to run.
        Returns:
            dict: A dictionary containing computed properties.
        """
        # args = [(self.frames[i], self.L_list[i], i) for i in np.arange(0, len(self.frames), 1)]
        print("Number of CPU cores: ", n_jobs, flush=True)
        # results = Parallel(n_jobs=n_jobs)(delayed(self._process_frame)(*arg) for arg in args) 
        # print("Number of processed frames: ", len(results), flush=True)
        # self.property_avg = self._average_properties(results)
        batch_args = [(self.frames[i:i + batch_size], self.L_list[i:i + batch_size], np.arange(i, i + batch_size)) for i in range(0, len(self.frames), batch_size)]
        results = Parallel(n_jobs=n_jobs)(delayed(self._batch_process_frame)(*arg) for arg in batch_args)
        # Flatten the list of lists
        results = [item for sublist in results for item in sublist]
        print("Number of processed frames: ", len(results), flush=True)
        # Filter out None results
        results = [result for result in results if result is not None]
        self.property_avg = self._average_properties(results)
        return self.property_avg

    def _average_properties(self, results: list) -> dict:
        property_avg = {}
        for result in results:
            for key1 in result.keys():
                if isinstance(result[key1], dict):
                    if key1 not in property_avg.keys():
                        property_avg[key1] = {}
                    for key2 in result[key1].keys():
                        if key2 not in property_avg[key1]:
                            property_avg[key1][key2] = []
                        property_avg[key1][key2].append(result[key1][key2])
                else:
                    property_avg[key1] = []
                    property_avg[key1].append(result[key1])
        # Compute the average
        for key1 in property_avg.keys():
            if isinstance(property_avg[key1], dict):
                for key2 in property_avg[key1].keys():
                    property_avg[key1][key2] = np.array(property_avg[key1][key2])
                    property_avg[key1][key2] = np.mean(property_avg[key1][key2], axis=0)
            else:
                property_avg[key1] = np.mean(property_avg[key1], axis=0)

        return property_avg

    def plot_and_save(self):
        self.bl_r = (self.bl_range[:-1] + self.bl_range[1:]) / 2 
        self.angle_r = (self.angle_range[:-1] + self.angle_range[1:]) / 2
        self.r_range = np.arange(0, self.RDF_cutoff, self.RDF_delta_r)
        self.R = (self.r_range[:-1] + self.r_range[1:]) / 2
        self.Q = self.property_avg['Sq']['Q']

        # plot
        n_subplots = 4
        fig, ax = plt.subplots(1, n_subplots, dpi=300, figsize=(2*n_subplots, 3))
        for i, key in enumerate(self.property_avg['RDF'].keys()):
            g = self.property_avg['RDF'][key]
            ax[0].plot(self.R, g+i*0.5, label=key)
        ax[0].set_xlabel('r (a)')
        ax[0].set_ylabel('g(r)')
        ax[0].legend(frameon=False)
        if 'bl' in self.property_avg.keys():
            r_min = self.bl_r[-1]
            r_max = self.bl_r[0]
            for key in self.property_avg['bl'].keys():
                bl_hist = self.property_avg['bl'][key]
                ax[1].plot(self.bl_r, bl_hist, label=key)
                if self.bl_r[bl_hist >0][0] < r_min:
                    r_min = self.bl_r[bl_hist > 0][0]
                if self.bl_r[bl_hist >0][-1] > r_max:
                    r_max = self.bl_r[bl_hist > 0][-1]
            ax[1].set_xlim(r_min, r_max)
        ax[1].set_xlabel('bond length (a)')
        ax[1].set_ylabel('p(r)')
        ax[1].legend(frameon=False)
        if 'angle' in self.property_avg.keys():
            for key in self.property_avg['angle'].keys():
                angle_hist = self.property_avg['angle'][key]
                ax[2].plot(self.angle_r, angle_hist, label=key)
        ax[2].set_xlabel('angle (degree)')
        ax[2].set_ylabel('p(theta)')
        ax[2].legend(frameon=False)
        for i, key in enumerate(self.property_avg['Sq'].keys()):
            S = self.property_avg['Sq'][key]
            ax[3].plot(self.Q, S, label=key)
        ax[3].set_xlabel('q')
        ax[3].set_ylabel('S(q)')
        ax[3].legend(frameon=False)
        plt.tight_layout()
        plt.savefig('str.png')

        if 'RDF' in self.property_avg.keys():
            g_list = []
            for key in self.property_avg['RDF'].keys():
                g = self.property_avg['RDF'][key]
                g_list.append(g)
            g_list = np.array(g_list)
            np.savetxt('pdf.txt', np.column_stack([self.R, g_list.T]))

        if 'Sq' in self.property_avg.keys():
            S_list = []
            for key in self.property_avg['Sq'].keys():
                S = self.property_avg['Sq'][key]
                S_list.append(S)
            S_list = np.array(S_list)
            np.savetxt('sq.txt', np.column_stack([self.Q, S_list.T]))

        if 'bl' in self.property_avg.keys():
            bl_list = []
            for key in self.property_avg['bl'].keys():
                bl_hist = self.property_avg['bl'][key]
                bl_list.append(bl_hist)
            bl_list = np.array(bl_list)
            np.savetxt('bl.txt', np.column_stack([self.bl_r, bl_list.T]))

        if 'angle' in self.property_avg.keys():
            angle_list = []
            for key in self.property_avg['angle'].keys():
                angle_hist = self.property_avg['angle'][key]
                angle_list.append(angle_hist)
            angle_list = np.array(angle_list)
            np.savetxt('angle.txt', np.column_stack([self.angle_r, angle_list.T]))

        np.savetxt('density.txt', [self.property_avg['density']])

    def write_pot_table(self, T: float = 300):
        """
        Write the potential table, 'pot_AA.table', which is useful as reference for CG force field.
        Args:
            T (float): Temperature in Kelvin.
        """

        # prepare Boltzmann Inversion FF
        zip_FF = []
        for key in self.property_avg['RDF'].keys():
            g = self.property_avg['RDF'][key]
            PMF = -np.log(g + 1e-8) * scc.Boltzmann * scc.Avogadro / 4184 * T
            PMF = savgol_filter(PMF, 3, 1)
            FF = -np.gradient(PMF, self.R)
            zip_FF.append((self.R, PMF, FF, 'pair_' + str(key[0]) + str(key[1])))

        if 'bl' in self.property_avg.keys():
            for key in self.property_avg['bl'].keys():
                bl_hist = self.property_avg['bl'][key]
                PMF_bond = -np.log(bl_hist + 1e-8) * scc.Boltzmann * scc.Avogadro / 4184 * T
                PMF_bond = savgol_filter(PMF_bond, 3, 1)
                FF_bond = -np.gradient(PMF_bond, self.bl_r)
                zip_FF.append((self.bl_r, PMF_bond, FF_bond, 'bond_' + str(key)))

        if 'angle' in self.property_avg.keys():
            for key in self.property_avg['angle'].keys():
                angle_hist = self.property_avg['angle'][key]
                PMF_angle = -np.log(angle_hist + 1e-8) * scc.Boltzmann * scc.Avogadro / 4184 * T
                PMF_angle = savgol_filter(PMF_angle, 3, 1)
                FF_angle = -np.gradient(PMF_angle, self.angle_r)
                zip_FF.append((self.angle_r, PMF_angle, FF_angle, 'angle_' + str(key)))

        tcp.write_pot_table('pot_AA.table', zip_FF)
