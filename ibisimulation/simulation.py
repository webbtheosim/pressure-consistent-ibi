import time, os, copy, subprocess
import numpy as np
from .utils import write_pot_table
from joblib import Parallel, delayed
from . import tools_lammps as tl_lmp
from scipy.constants import Avogadro as NA
from . import tools_structure as tcp
from .result_processor import ResultProcessor
from .plot import Plotter
from .update import PotentialUpdater
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

class Simulator:
    def __init__(self, initializer):
        self.init = initializer
        self.config = initializer.config
        self.alpha = initializer.alpha

        for attr in [
            "r_pot", "e_pot", "f_pot", "pair_types", "bond_types", "angle_types",
            "r_bond", "e_bond", "f_bond", "r_angle", "e_angle", "f_angle",
            "r_bond_dist", "r_angle_dist", "pdf_ref", "bond_length_ref", "angle_dist_ref",
            "RDF_cutoff", "RDF_delta_r", "n_iter", "n_cpus", "target"
        ]:
            if hasattr(initializer, attr):
                setattr(self, attr, getattr(initializer, attr))

    def run(self, start=1):
        print("Simulation started with config:", self.config)
        for i in range(start, self.n_iter + start):
            start_time = time.time()
            self.i_iter = i
            self.run_iteration()
            print(f"Iteration {i}, Time taken: {time.time() - start_time:.2f} seconds")

            if self.config.get("is_lr_decay") and i % self.config["decay_freq"] == 0:
                if abs(self.alpha) > abs(self.config["min_alpha"]):
                    self.alpha *= self.config["decay_rate"]
                    print(f"Learning rate decayed to {self.alpha}")

    def run_iteration(self):
        self._prepare_directory()
        self._write_potentials()
        self._run_lammps()
        self._process_results()

    def _prepare_directory(self):
        self.directory = f"CG{self.i_iter}"
        os.makedirs(self.directory, exist_ok=True)
        os.system(f"cp {self.config['init_str']} {self.directory}/tmp.dat")
        os.system(f"cp {self.config['lmp_input']} {self.directory}/")
    
    def _write_potentials(self):
        r_pot_write = copy.copy(self.r_pot)
        r_pot_write[0] = 1e-8
        pot_args = [(r_pot_write, self.e_pot[f"pair_{k}"], self.f_pot[f"pair_{k}"], f"pair_{k}") for k in self.pair_types]

        if self.bond_types:
            r_bond_write = copy.copy(self.r_bond)
            r_bond_write[0], r_bond_write[-1] = 1e-8, 100
            pot_args += [
                (r_bond_write, self.e_bond[f"bond_{k}"], self.f_bond[f"bond_{k}"], f"bond_{k}")
                for k in self.bond_types
            ]

        if self.angle_types:
            pot_args += [
                (self.r_angle, self.e_angle[f"angle_{k}"], self.f_angle[f"angle_{k}"], f"angle_{k}")
                for k in self.angle_types
            ]

        write_pot_table(f"{self.directory}/pot.table", pot_args)

    def _run_lammps(self):
        subprocess.run(
            f"{self.init.run_lammps} -in {self.config['lmp_input']} > log",
            shell=True,
            cwd=self.directory
        )

    def _process_results(self):
        processor = ResultProcessor(
            directory=self.directory,
            config=self.config,
            initializer=self.init,
            alpha=self.alpha
        )
        self.property = processor.run()
        self.target_traj = self.property["target_all"]
        self.box_size = processor.box_size

        Plotter(self).plot_results()
        Plotter(self).plot_potentials()
        PotentialUpdater(self).update()

    def _smooth_potential(self, r, e, num_points=10000, sigma=3):
        # interpolate and then smooth the data
        r_new = np.linspace(r[0], r[-1], num_points)
        interp_func = interp1d(r, e, kind='linear')
        e_interp = interp_func(r_new)
        # Gaussian smoothing
        e_smooth = gaussian_filter1d(e_interp, sigma=sigma)

        # # cubic spline smoothing
        # # only apply to the range beyond self.effective_rmin
        # length_tmp = self.effective_rmin - 0.5
        # r_new1 = r_new[r_new > length_tmp]
        # e_interp1 = e_interp[r_new > length_tmp]
        # e_smooth1 = UnivariateSpline(r_new1, e_interp1, s=sigma, ext=3)(r_new1)
        # e_smooth = np.zeros_like(e_interp)
        # e_smooth[r_new > length_tmp] = e_smooth1
        # e_interp0 = e_interp[r_new <= length_tmp]
        # e_interp0 = e_interp0 + e_smooth1[0] - e_interp1[0]
        # e_smooth[r_new <= length_tmp] = e_interp0

        interp_back = interp1d(r_new, e_smooth, kind='linear')
        e_smoothed_on_old_r = interp_back(r)
        f_smooth = -np.gradient(e_smoothed_on_old_r, r)
        return e_smoothed_on_old_r, f_smooth
    
    # def process_results(self):
    #     dump_file = os.path.join(self.directory, "dump.xyz")
    #     output_file = os.path.join(self.directory, "out.dat")
    #     log_file = os.path.join(self.directory, "log.lammps")

    #     # Read the LAMMPS dump and output files
    #     frame, _, L_list = tl_lmp.read_lammps_dump_custom(dump_file)
    #     nframes = len(frame)
    #     self.lmp = tl_lmp.read_lammps_full(output_file)
    #     self.lmp.atom_info = self.lmp.atom_info[np.argsort(self.lmp.atom_info[:, 0]), :]
    #     # natom_types = lmp.natom_types
    #     mass_AA = self.lmp.mass[self.lmp.atom_info[:, 2].astype(int)-1, 1].astype(float)
    #     self.total_mass = np.sum(mass_AA)
    #     self.type_atom = self.lmp.atom_info[:, 2]

    #     self.compute_RDF_matrix = {}
    #     # if self.lmp.nbonds > 0:
    #     for i in range(self.lmp.natom_types):
    #         for j in range(i, self.lmp.natom_types):
    #             idx_type1 = np.where(self.type_atom == i+1)[0]
    #             idx_type2 = np.where(self.type_atom == j+1)[0]
    #             mol_type1 = self.lmp.atom_info[idx_type1, 1]
    #             mol_type2 = self.lmp.atom_info[idx_type2, 1]
    #             self.compute_RDF_matrix[(i+1, j+1)] = (mol_type1[:, None] != mol_type2[None, :]).astype(int)

    #     # Map the process_frame function across all frames
    #     args = [
    #         (self.lmp, frame[i], L_list[i])
    #         for i in np.arange(int(nframes / 2), nframes)
    #     ]
    #     results = Parallel(n_jobs=self.n_cpus)(delayed(self.process_frame)(*arg) for arg in args)

    #     # Collect the results, directly for average
    #     property_avg = {}
    #     for result in results:
    #         for key1 in result.keys():
    #             if type(result[key1]) == dict:
    #                 if key1 not in property_avg.keys():
    #                     property_avg[key1] = {}
    #                 for key2 in result[key1].keys():
    #                     if key2 not in property_avg[key1]:
    #                         property_avg[key1][key2] = []
    #                     property_avg[key1][key2].append(result[key1][key2])
    #             else:
    #                 property_avg[key1] = []
    #                 property_avg[key1].append(result[key1])
    #     # Compute the average
    #     for key1 in property_avg.keys():
    #         if type(property_avg[key1]) == dict:
    #             for key2 in property_avg[key1].keys():
    #                 property_avg[key1][key2] = np.array(property_avg[key1][key2])
    #                 property_avg[key1][key2] = np.mean(property_avg[key1][key2], axis=0)
    #         else:
    #             property_avg[key1] = np.mean(property_avg[key1], axis=0)

    #     log = tl_lmp.read_log_lammps(log_file)
    #     if self.config["target"] == "density":
    #         self.target = np.array(log[-1]["Density"])
    #     elif self.config["target"] == "pressure":
    #         self.target = np.array(log[-1]["Press"])
    #     else:
    #         raise ValueError("Target not supported")

    #     self.box_size = np.mean(log[-1]["Lx"])

    #     property_avg["target"] = np.mean(self.target)

    #     # Save pdfs
    #     if 'RDF' in property_avg.keys():
    #         g_list = []
    #         for key in property_avg["RDF"].keys():
    #             g_list.append(property_avg["RDF"][key])
    #         g_list = np.array(g_list)
    #         np.savetxt(os.path.join(self.directory, "pdf.txt"), np.column_stack((self.init.r_RDF_ref, g_list.T)))

    #     if 'bl' in property_avg.keys():
    #         bl_list = []
    #         for key in property_avg["bl"].keys():
    #             bl_list.append(property_avg["bl"][key])
    #         bl_list = np.array(bl_list)
    #         np.savetxt(os.path.join(self.directory, "bond_length_dist.txt"), np.column_stack((self.r_bond, bl_list.T)),)

    #     if 'angle' in property_avg.keys():
    #         angle_list = []
    #         for key in property_avg["angle"].keys():
    #             angle_list.append(property_avg["angle"][key])
    #         angle_list = np.array(angle_list)
    #         np.savetxt(os.path.join(self.directory, "angle_dist.txt"), np.column_stack((self.r_angle, angle_list.T)))
        
    #     self.property = property_avg
    #     self.plot_results()
    #     self.update_potentials()

    # def process_frame(self, lmp, frame, L):
    #     # property to compute: density, all RDFs, all bond length distributions, all angle distributions
    #     property = {}
    #     box_size = L[0][1] - L[0][0]
    #     box = np.array([[box_size, 0, 0], [0, box_size, 0], [0, 0, box_size]])
    #     atom_types = lmp.atom_info[:, 2].astype(int)
    #     natom_types = lmp.natom_types
    #     bond_atom_idx = []
    #     if lmp.nbonds > 0:
    #         bond_atom_idx = lmp.bond_info[:, 2:4] - 1
    #         bond_atom_idx = bond_atom_idx.astype(int)

    #     coors = np.hstack((
    #             frame["x"].to_numpy().reshape(-1, 1),
    #             frame["y"].to_numpy().reshape(-1, 1),
    #             frame["z"].to_numpy().reshape(-1, 1),
    #             ))

    #     # Compute the density
    #     V = box_size**3
    #     rho = self.total_mass / V * 1e24 / NA
    #     property["density"] = rho

    #     # Compute the RDF
    #     # n_RDFs = len(self.pair_types)
    #     # assert n_RDFs == natom_types * (natom_types + 1) / 2, "RDFs not equal to atom types"
    #     coors_type = {}
    #     RDFs = {}
    #     for i in range(natom_types):
    #         coors_type[i] = coors[atom_types == i+1, :]
    #     for i in range(natom_types):
    #         for j in range(i, natom_types):
    #             keyname = f"pair_{i+1}{j+1}"
    #             if self.config["RDF_type"] == 1: # non-bonded RDF
    #                 if lmp.nbonds > 0:
    #                     bond_atom_idx_type = []
    #                     idx_typei = np.where(atom_types == i + 1)[0]
    #                     idx_typej = np.where(atom_types == j + 1)[0]
    #                     for k, l in bond_atom_idx:
    #                         if atom_types[k] == i+1 and atom_types[l] == j+1:
    #                             bond_atom_idx_type.append([np.where(idx_typei==k)[0], np.where(idx_typej==l)[0]])
    #                         if atom_types[k] == j+1 and atom_types[l] == i+1:
    #                             bond_atom_idx_type.append([np.where(idx_typei==l)[0], np.where(idx_typej==k)[0]])
    #                 else:
    #                     bond_atom_idx_type = None
    #                 if keyname in self.pdf_ref.keys():
    #                     _, g, _, _ = tcp.pdf_sq_cross(box, coors_type[i], coors_type[j], bond_atom_idx_type, r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r)
    #                     RDFs[keyname] = g
    #             elif self.config["RDF_type"] == 2: # different molecules
    #                 if keyname in self.pdf_ref.keys(): 
    #                     _, g, _, _ = tcp.pdf_sq_cross_mask(box, coors_type[i], coors_type[j], self.compute_RDF_matrix[(i+1, j+1)], r_cutoff=self.RDF_cutoff, delta_r=self.RDF_delta_r)
    #                     RDFs[keyname] = g
    #     property["RDF"] = RDFs

    #     # Compute the bond length distribution
    #     if len(self.bond_types) > 0:
    #         bl_dict = {}
    #         # nbond_types = len(self.bond_types)
    #         for i in self.bond_types:
    #             keyname = f"bond_{i}"
    #             if keyname not in self.bond_length_ref.keys():
    #                 continue
    #             bond_atom_idx_btype = bond_atom_idx[lmp.bond_info[:, 1] == int(i)]
    #             bond_length = tcp.compute_bond_length(coors, bond_atom_idx_btype, box_size)
    #             a, _ = np.histogram(bond_length, bins=self.r_bond_dist, density=True)
    #             bl_dict[keyname] = a
    #         property["bl"] = bl_dict

    #     # Compute angle distribution
    #     if len(self.angle_types) > 0:
    #         angle_dict = {}
    #         # nangle_types = len(self.angle_types)
    #         angle_atoms = np.array(lmp.angle_info[:, 2:5] - 1, dtype=int)
    #         for i in self.angle_types:
    #             keyname = f"angle_{i}"
    #             if keyname not in self.angle_dist_ref.keys():
    #                 continue
    #             angle_atoms_atype = angle_atoms[lmp.angle_info[:, 1] == int(i)].astype(int)
    #             angle = tcp.compute_angle(coors, angle_atoms_atype, box_size)
    #             aa, _ = np.histogram(angle, bins=self.r_angle_dist, density=True)
    #             angle_dict[keyname] = aa
    #         property["angle"] = angle_dict

    #     return property
