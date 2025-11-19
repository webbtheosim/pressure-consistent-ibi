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