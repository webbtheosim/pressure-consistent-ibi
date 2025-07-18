import yaml
import numpy as np
import os
from .potential import lj, morse, harmonic, linear_attra
from .utils import read_table_pot 

class Initializer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.n_cpus = self.config.get("n_cpus", int(os.getenv("SLURM_CPUS_ON_NODE", 1)))
        self.initialize_parameters()

    def initialize_parameters(self):
         
        self.run_lammps = self.config["run_lammps_command"]
        self.alpha = self.config["alpha"]
        self.n_iter = self.config["n_iter"]
        self.temp = self.config["temp"]
        self.density_ref = self.config["density_ref"]

        self.turn_on_LC = False # default off
        self.target = []

        # RDF and pair potenital
        self.RDF_cutoff, self.RDF_delta_r = self.config["RDF_cutoff"], self.config["RDF_delta_r"]
        r_tmp = np.arange(0, self.config["RDF_cutoff"], self.config["RDF_delta_r"])
        self.r_pot = (r_tmp[1:] + r_tmp[:-1]) / 2
        self.pair_types = self.config["pair_types"]
        # load RDF reference
        rdf_data = np.loadtxt(self.config["RDF_ref"])
        # cut the RDF reference to the same length as r_pot
        rdf_data = rdf_data[:len(self.r_pot), :]
        self.r_RDF_ref = rdf_data[:,0]
        
        self.pdf_ref = {}
        self.e_pot = {}
        self.f_pot = {}
        self.effective_rmin = 10
        for i, keyname in enumerate(self.pair_types):
            self.pdf_ref[f"pair_{keyname}"] = rdf_data[:,i+1]
            if self.r_RDF_ref[np.argwhere(rdf_data[:, i+1] > 1e-6)][0] < self.effective_rmin:
                self.effective_rmin = self.r_RDF_ref[np.argwhere(rdf_data[:, i+1] > 1e-6)][0][0]
            if self.config["use_existing_FF"]:
                self.e_pot[f"pair_{keyname}"] = read_table_pot(self.config["use_existing_FF"], f"pair_{keyname}")[1]
                self.f_pot[f"pair_{keyname}"] = read_table_pot(self.config["use_existing_FF"], f"pair_{keyname}")[2]
                print(f"Read existing FF for pair_{keyname}", flush=True)
                if "add_LC" in self.config:
                    self.e_pot[f"pair_{keyname}"] = linear_attra(self.r_pot, self.e_pot[f"pair_{keyname}"], self.config["add_LC"], self.r_pot[-1]/5*4)
                    print("Add linear correction", flush=True)
            else:
                if 'lj_params' in self.config:
                    self.e_pot[f"pair_{keyname}"] = lj(self.r_pot, *self.config["lj_params"][i])
                if 'morse_params' in self.config:
                    self.e_pot[f"pair_{keyname}"] = morse(self.r_pot, *self.config["morse_params"][i])
                self.f_pot[f"pair_{keyname}"] = -np.gradient(self.e_pot[f"pair_{keyname}"], self.r_pot)
        # print(f"Effective rmin: {self.effective_rmin}", flush=True) 

        # bond
        self.bond_types = self.config["bond_types"] if 'bond_types' in self.config else []
        if len(self.bond_types) > 0:
            self.alpha_bond = self.config["alpha_bond"]
            bond_data = np.loadtxt(self.config["bond_ref"])
            self.r_bond = bond_data[:, 0]
            self.bond_length_ref = {}
            self.e_bond = {}
            self.f_bond = {}
            for i,keyname in enumerate(self.bond_types):
                self.bond_length_ref[f"bond_{keyname}"] = bond_data[:, i + 1]

            r_bond_dist = np.zeros(len(self.r_bond) + 1)
            r_bond_dist[0] = self.r_bond[0] - 0.5 * (self.r_bond[1] - self.r_bond[0])
            r_bond_dist[-1] = self.r_bond[-1] + 0.5 * (self.r_bond[-1] - self.r_bond[-2])
            r_bond_dist[1:-1] = 0.5 * (self.r_bond[1:] + self.r_bond[:-1])
            self.r_bond_dist = r_bond_dist
            
            for i,keyname in enumerate(self.bond_types):
                if self.config["use_existing_bondFF"]:
                    table = read_table_pot(self.config["use_existing_bondFF"], f"bond_{keyname}")
                    self.e_bond[f"bond_{keyname}"] = table[1]
                    self.f_bond[f"bond_{keyname}"] = table[2]
                else:
                    self.e_bond[f"bond_{keyname}"] = harmonic(self.r_bond, *self.config["bond_params"][i])
                    self.f_bond[f"bond_{keyname}"] = -np.gradient(self.e_bond[f"bond_{keyname}"], self.r_bond)
        
        # angle
        self.angle_types = self.config["angle_types"] if 'angle_types' in self.config else []
        if len(self.angle_types) > 0:
            self.alpha_angle = self.config["alpha_angle"]
            angle_data = np.loadtxt(self.config["angle_ref"])
            self.r_angle = angle_data[:, 0]
            self.angle_dist_ref = {}
            self.e_angle = {}
            self.f_angle = {}
            for i, keyname in enumerate(self.angle_types):
                self.angle_dist_ref[f"angle_{keyname}"] = angle_data[:, i + 1]

            r_angle_dist = np.zeros(len(self.r_angle) + 1)
            r_angle_dist[0] = self.r_angle[0] - 0.5 * (self.r_angle[1] - self.r_angle[0])
            r_angle_dist[-1] = self.r_angle[-1] + 0.5 * (self.r_angle[-1] - self.r_angle[-2])
            r_angle_dist[1:-1] = 0.5 * (self.r_angle[1:] + self.r_angle[:-1])
            self.r_angle_dist = r_angle_dist
            
            for i, keyname in enumerate(self.angle_types):
                if self.config["use_existing_bondFF"]:
                    table = read_table_pot(self.config["use_existing_bondFF"], f"angle_{keyname}")
                    self.e_angle[f"angle_{keyname}"] = table[1]
                    self.f_angle[f"angle_{keyname}"] = table[2]
                else:
                    self.e_angle[f"angle_{keyname}"] = harmonic(self.r_angle, *self.config["angle_params"][i])
                    # self.angle = np.arange(0, 180.5, 1)
                    self.f_angle[f"angle_{keyname}"] = -np.gradient(self.e_angle[f"angle_{keyname}"], self.r_angle)

        self.error_list = {}
        self.density_perror_list = {}
        # density corretion
        if self.config.get('density_correction', False):
            self.gamma = self.config.get("gamma", None) if self.config['PM'] in ['rscale', 'hybrid'] else None
            self.LC_A = self.config.get("A", None) if self.config['PM'] == 'linear' else None
        # smooth
        self.smooth_sigma = self.config.get("smooth_sigma", 3)
        # print(f"Smooth sigma: {self.smooth_sigma}", flush=True)