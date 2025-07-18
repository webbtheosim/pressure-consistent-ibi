import os
import numpy as np
from joblib import Parallel, delayed
from scipy.constants import Avogadro as NA
from . import tools_lammps as tl_lmp
from . import tools_structure as tcp

class ResultProcessor:
    def __init__(self, directory, config, initializer, alpha):
        self.directory = directory
        self.config = config
        self.init = initializer
        self.alpha = alpha
        self._alias_initializer_attributes()

    def _alias_initializer_attributes(self):
        for attr in [
            "pdf_ref", "r_RDF_ref", "RDF_cutoff", "RDF_delta_r", "bond_types",
            "r_bond", "r_bond_dist", "bond_length_ref", "angle_types",
            "r_angle", "r_angle_dist", "angle_dist_ref"
        ]:
            if hasattr(self.init, attr):
                setattr(self, attr, getattr(self.init, attr))

    def run(self):
        dump_file = os.path.join(self.directory, "dump.xyz")
        output_file = os.path.join(self.directory, "out.dat")
        log_file = os.path.join(self.directory, "log.lammps")

        frame, _, L_list = tl_lmp.read_lammps_dump_custom(dump_file)
        nframes = len(frame)

        self.lmp = tl_lmp.read_lammps_full(output_file)
        self.lmp.atom_info = self.lmp.atom_info[np.argsort(self.lmp.atom_info[:, 0])]
        self.mass_AA = self.lmp.mass[self.lmp.atom_info[:, 2].astype(int)-1, 1].astype(float)
        self.total_mass = np.sum(self.mass_AA)
        self.type_atom = self.lmp.atom_info[:, 2]

        self._compute_rdf_matrix()

        args = [(self.lmp, frame[i], L_list[i]) for i in np.arange(int(nframes / 2), nframes)]
        results = Parallel(n_jobs=self.config["n_cpus"])(delayed(self._process_frame)(*arg) for arg in args)

        property_avg = self._average_results(results)
        log = tl_lmp.read_log_lammps(log_file)

        if self.config["target"] == "density":
            target_val = np.array(log[-1]["Density"])
        elif self.config["target"] == "pressure":
            target_val = np.array(log[-1]["Press"])
        else:
            raise ValueError("Target not supported")

        property_avg["target"] = np.mean(target_val)
        property_avg["target_all"] = target_val
        self.box_size = np.mean(log[-1]["Lx"])

        self._write_output(property_avg)
        return property_avg

    def _compute_rdf_matrix(self):
        self.compute_RDF_matrix = {}
        for i in range(self.lmp.natom_types):
            for j in range(i, self.lmp.natom_types):
                idx_type1 = np.where(self.type_atom == i+1)[0]
                idx_type2 = np.where(self.type_atom == j+1)[0]
                mol_type1 = self.lmp.atom_info[idx_type1, 1]
                mol_type2 = self.lmp.atom_info[idx_type2, 1]
                self.compute_RDF_matrix[(i+1, j+1)] = (mol_type1[:, None] != mol_type2[None, :]).astype(int)

    def _process_frame(self, lmp, frame, L):
        properties = {}
        box_size = L[0][1] - L[0][0]
        box = np.array([[box_size, 0, 0], [0, box_size, 0], [0, 0, box_size]])
        atom_types = lmp.atom_info[:, 2].astype(int)
        natom_types = lmp.natom_types

        coors = np.hstack((
                frame["x"].to_numpy().reshape(-1, 1),
                frame["y"].to_numpy().reshape(-1, 1),
                frame["z"].to_numpy().reshape(-1, 1),
                ))

        V = box_size**3
        rho = self.total_mass / V * 1e24 / NA
        properties["density"] = rho

        coors_type = {i: coors[atom_types == i+1] for i in range(natom_types)}
        RDFs = {}
        for i in range(natom_types):
            for j in range(i, natom_types):
                keyname = f"pair_{i+1}{j+1}"
                if keyname not in self.pdf_ref:
                    continue
                if self.config["RDF_type"] == 1:
                    bond_atom_idx = lmp.bond_info[:, 2:4] - 1 if lmp.nbonds > 0 else None
                    bond_atom_idx_type = None
                    if bond_atom_idx is not None:
                        idx_i = np.where(atom_types == i+1)[0]
                        idx_j = np.where(atom_types == j+1)[0]
                        bond_atom_idx_type = [
                            [np.where(idx_i == k)[0], np.where(idx_j == l)[0]]
                            for k, l in bond_atom_idx
                            if (atom_types[k], atom_types[l]) in [(i+1, j+1), (j+1, i+1)]
                        ]
                    _, g, _, _ = tcp.pdf_sq_cross(
                        box, coors_type[i], coors_type[j], bond_atom_idx_type,
                        r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r
                    )
                elif self.config["RDF_type"] == 2:
                    _, g, _, _ = tcp.pdf_sq_cross_mask(
                        box, coors_type[i], coors_type[j], self.compute_RDF_matrix[(i+1, j+1)],
                        r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r
                    )
                else:
                    continue
                RDFs[keyname] = g
        properties["RDF"] = RDFs

        if self.bond_types:
            bl_dict = {}
            bond_atom_idx = lmp.bond_info[:, 2:4] - 1
            for btype in self.bond_types:
                key = f"bond_{btype}"
                if key not in self.bond_length_ref:
                    continue
                idx = lmp.bond_info[:, 1] == int(btype)
                bond_atoms = bond_atom_idx[idx]
                bond_lengths = tcp.compute_bond_length(coors, bond_atoms, box_size)
                hist, _ = np.histogram(bond_lengths, bins=self.r_bond_dist, density=True)
                bl_dict[key] = hist
            properties["bl"] = bl_dict

        if self.angle_types:
            angle_dict = {}
            angle_atoms = lmp.angle_info[:, 2:5] - 1
            for atype in self.angle_types:
                key = f"angle_{atype}"
                if key not in self.angle_dist_ref:
                    continue
                idx = lmp.angle_info[:, 1] == int(atype)
                angle_atoms_type = angle_atoms[idx]
                angles = tcp.compute_angle(coors, angle_atoms_type, box_size)
                hist, _ = np.histogram(angles, bins=self.r_angle_dist, density=True)
                angle_dict[key] = hist
            properties["angle"] = angle_dict

        return properties

    def _average_results(self, results):
        property_avg = {}
        for result in results:
            for key1 in result:
                if isinstance(result[key1], dict):
                    if key1 not in property_avg:
                        property_avg[key1] = {}
                    for key2 in result[key1]:
                        property_avg[key1].setdefault(key2, []).append(result[key1][key2])
                else:
                    property_avg.setdefault(key1, []).append(result[key1])

        for key1 in property_avg:
            if isinstance(property_avg[key1], dict):
                for key2 in property_avg[key1]:
                    property_avg[key1][key2] = np.mean(property_avg[key1][key2], axis=0)
            else:
                property_avg[key1] = np.mean(property_avg[key1], axis=0)

        return property_avg

    def _write_output(self, property_avg):
        if "RDF" in property_avg:
            g_list = [property_avg["RDF"][key] for key in property_avg["RDF"]]
            np.savetxt(os.path.join(self.directory, "pdf.txt"), np.column_stack((self.r_RDF_ref, np.array(g_list).T)))

        if "bl" in property_avg:
            bl_list = [property_avg["bl"][key] for key in property_avg["bl"]]
            np.savetxt(os.path.join(self.directory, "bond_length_dist.txt"), np.column_stack((self.r_bond, np.array(bl_list).T)))

        if "angle" in property_avg:
            angle_list = [property_avg["angle"][key] for key in property_avg["angle"]]
            np.savetxt(os.path.join(self.directory, "angle_dist.txt"), np.column_stack((self.r_angle, np.array(angle_list).T)))

