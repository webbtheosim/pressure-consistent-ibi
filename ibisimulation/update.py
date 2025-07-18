import numpy as np
import pandas as pd
import os
from .potential import rscale_table_pot, linear_attra, shift_table_pot
from .utils import rweight, rweight_back

class PotentialUpdater:
    def __init__(self, simulator):
        self.sim = simulator
        self.log_file = "log.txt"

    def update(self):
        sim = self.sim
        target = sim.property["target"]
        sim.target.append(target)

        new_weight = self._get_weight()

        error_PDF, perror_PDF = {}, {}
        update_pdf = not (sim.config['target'] == 'density' and target < 0.2)

        for key in sim.property["RDF"].keys():
            if key not in sim.pdf_ref:
                raise ValueError(f"Key {key} not in reference RDF")

            err = np.log((sim.property["RDF"][key] + 1e-8)/(sim.pdf_ref[key] + 1e-8)) * new_weight
            error_PDF[key] = err
            perror_PDF[key] = np.mean(np.abs(sim.property["RDF"][key] - sim.pdf_ref[key]))

            if update_pdf:
                sim.e_pot[key] -= err * sim.alpha * sim.config["temp"]
            sim.e_pot[key] -= sim.e_pot[key][-1]

            if sim.config.get("density_correction") and sim.i_iter % sim.config["density_correction_freq"] == 0:
                self._apply_pressure_matching(key, target)

            if sim.i_iter % sim.config["smooth_freq"] == 0:
                sim.e_pot[key] = sim._smooth_potential(sim.r_pot, sim.e_pot[key], sigma=sim.init.smooth_sigma)[0]
            sim.f_pot[key] = -np.gradient(sim.e_pot[key], sim.r_pot)

        self._update_bonds()
        self._update_angles()
        self._log_errors(perror_PDF)

    def _get_weight(self):
        sim = self.sim
        r_pot = sim.r_pot
        config = sim.config

        if config.get("use_weight"):
            if config["use_weight"] == "r-1":
                return np.ones_like(r_pot) / r_pot * 5
            elif config["use_weight"] == "r-2":
                return np.ones_like(r_pot) / r_pot**2 * 25
            elif config["use_weight"] == "r-3":
                return np.ones_like(r_pot) / r_pot**3 * 125
            elif config["use_weight"] == "grad":
                r_grad = min(config["grad_r1"], (config["grad_r1"] - config["grad_r0"]) * sim.i_iter / config["grad_steps"] + config["grad_r0"])
                return rweight(r_pot, r_grad, 0.5)
            elif config["use_weight"] == "LD":
                min_r = np.min(r_pot[sim.pdf_ref["pair_11"] > 0])
                w = np.ones_like(r_pot)
                w[r_pot > min_r] = 1 - (r_pot[r_pot > min_r] - min_r) / (r_pot[-1] - min_r)
                return w
            elif config["use_weight"] == "LI":
                min_r = np.min(r_pot[sim.pdf_ref["pair_11"] > 0])
                w = np.zeros_like(r_pot)
                w[r_pot > min_r] = (r_pot[r_pot > min_r] - min_r) / (r_pot[-1] - min_r)
                return w
            elif config["use_weight"] == "grad_back":
                r_grad = min(config["grad_r1"], (config["grad_r1"] - config["grad_r0"]) * sim.i_iter / config["grad_steps"] + config["grad_r0"])
                return rweight_back(r_pot, r_grad, 0.5)
        elif config.get("use_SR"):
            w = np.ones_like(r_pot)
            w[r_pot > config["use_SR"]] = 0
            return w
        return np.ones_like(r_pot)

    def _apply_pressure_matching(self, key, rho):
        sim = self.sim
        config = sim.config
        scale = 1.0
        sim.rho = sim.target

        if config["PM"] in ["rscale", "hybrid"]:
            if len(sim.rho) > 2:
                tmp = (sim.rho[-2] - sim.init.density_ref) / (sim.rho[-1] - sim.init.density_ref)
                if tmp > 0 and tmp < 1:
                    sim.init.gamma *= 1.01
                elif tmp < 0:
                    sim.init.gamma /= 1.01
            scale = (rho / sim.init.density_ref) ** float(sim.init.gamma if config["PM"] == "rscale" else config["gamma"])
            scale = np.clip(scale, 0.99, 1.01)
            sim.e_pot[key] = rscale_table_pot(sim.r_pot, sim.e_pot[key], scale)[1]

        elif config["PM"] in ["linear", "postlinear"]:
            if config["PM"] == "linear" and len(sim.rho) > 2:
                tmp = (sim.rho[-2] - sim.init.density_ref) / (sim.rho[-1] - sim.init.density_ref)
                if tmp > 0 and tmp < 1:
                    sim.LC_A *= 1.5
                elif tmp < 0:
                    sim.LC_A /= 1.5
            scale = np.log(rho / sim.init.density_ref) * (sim.LC_A if config["PM"] == "linear" else config["A"])
            if config["target"] == "pressure":
                scale = -scale
            scale = np.clip(scale, -0.001 if config["PM"] == "linear" else -0.01, 0.001 if config["PM"] == "linear" else 0.01)
            sim.e_pot[key] = linear_attra(sim.r_pot, sim.e_pot[key], scale, config["LC_rcut"])

        elif config["PM"] == "shift":
            delta = (rho - sim.init.density_ref) * config["delta"]
            delta = np.clip(delta, -0.1, 0.1)
            sim.e_pot[key] = shift_table_pot(sim.r_pot, sim.e_pot[key], delta)[1]
        else:
            raise ValueError("PM method not supported")

    def _update_bonds(self):
        sim = self.sim
        if "bl" not in sim.property:
            return
        for key in sim.property["bl"].keys():
            if key not in sim.bond_length_ref:
                continue
            vec = np.vstack((sim.property["bl"][key], sim.bond_length_ref[key]))
            weight = np.max(vec, axis=0) / np.max(vec)
            err = np.log((sim.property["bl"][key] + 1e-8) / (sim.bond_length_ref[key] + 1e-8)) * weight
            sim.e_bond[key] -= err * sim.alpha_bond * sim.config["temp"]
            if sim.config.get("smooth_bond") and sim.i_iter % sim.config["smooth_freq"] == 0:
                sim.e_bond[key] = sim._smooth_potential(sim.r_bond, sim.e_bond[key])[0]
            sim.f_bond[key] = -np.gradient(sim.e_bond[key], sim.r_bond)

    def _update_angles(self):
        sim = self.sim
        if "angle" not in sim.property:
            return
        for key in sim.property["angle"].keys():
            if key not in sim.angle_dist_ref:
                continue
            vec = np.vstack((sim.property["angle"][key], sim.angle_dist_ref[key]))
            weight = np.max(vec, axis=0) / np.max(vec)
            err = np.log((sim.property["angle"][key] + 1e-8) / (sim.angle_dist_ref[key] + 1e-8)) * weight
            sim.e_angle[key] -= err * sim.alpha_angle * sim.config["temp"]
            if sim.config.get("smooth_angle") and sim.i_iter % sim.config["smooth_freq"] == 0:
                sim.e_angle[key] = sim._smooth_potential(sim.r_angle, sim.e_angle[key])[0]
            sim.f_angle[key] = -np.gradient(sim.e_angle[key], sim.r_angle)

    def _log_errors(self, perror_PDF):
        sim = self.sim
        log_file = self.log_file.replace(".txt", ".csv")
        
        # Prepare a dictionary of the row to log
        row = {
            "iteration": sim.i_iter,
            "target": float(sim.property["target"]),
            "alpha": sim.alpha
        }
        
        # Add RDF errors
        for key, val in perror_PDF.items():
            row[f"PDF_error_{key}"] = val

        # Add bond length errors
        if "bl" in sim.property:
            for key in sim.property["bl"]:
                val = np.mean(np.abs(sim.property["bl"][key] - sim.bond_length_ref[key]))
                row[f"Bond_error_{key}"] = val

        # Add angle distribution errors
        if "angle" in sim.property:
            for key in sim.property["angle"]:
                val = np.mean(np.abs(sim.property["angle"][key] - sim.angle_dist_ref[key]))
                row[f"Angle_error_{key}"] = val

        # Write to CSV
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(log_file, index=False)

