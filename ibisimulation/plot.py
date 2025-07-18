import os
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, simulator):
        self.sim = simulator

    def plot_results(self):
        # Determine how many subplots
        n_subplots = 2
        if len(self.sim.bond_types) > 0:
            n_subplots += 1
        if len(self.sim.angle_types) > 0:
            n_subplots += 1
        fig, ax = plt.subplots(1, n_subplots, figsize=[2 * n_subplots, 3], dpi=300)
        ax = np.atleast_1d(ax)

        # Plot density or pressure target
        i_plot = 0
        ax[i_plot].plot(np.arange(len(self.sim.target_traj)), self.sim.target_traj, label="Target")
        ax[i_plot].plot([0, len(self.sim.target_traj)], [self.sim.init.density_ref, self.sim.init.density_ref], "k--", lw=1)
        ax[i_plot].set_xlabel("Iteration")
        ax[i_plot].set_ylabel("Target")
        i_plot += 1

        # Plot RDFs
        for i, key in enumerate(self.sim.property["RDF"].keys()):
            g = self.sim.property["RDF"][key]
            ax[i_plot].plot(self.sim.init.r_RDF_ref, g + 0.5 * i, label=key)
            ax[i_plot].plot(self.sim.init.r_RDF_ref, self.sim.pdf_ref[key] + 0.5 * i, "k--", lw=1)
        ax[i_plot].set_xlabel("r")
        ax[i_plot].set_ylabel("g(r)")
        i_plot += 1

        # Plot bond distributions
        if "bl" in self.sim.property:
            r_min, r_max = 10, 0
            for i, key in enumerate(self.sim.property["bl"].keys()):
                bl = self.sim.property["bl"][key]
                ax[i_plot].plot(self.sim.r_bond, bl, label=key)
                ax[i_plot].plot(self.sim.r_bond, self.sim.bond_length_ref[key], "k--", lw=1)
                if self.sim.r_bond[self.sim.bond_length_ref[key] > 0][-1] > r_max:
                    r_max = self.sim.r_bond[self.sim.bond_length_ref[key] > 0][-1]
                if self.sim.r_bond[self.sim.bond_length_ref[key] > 0][0] < r_min:
                    r_min = self.sim.r_bond[self.sim.bond_length_ref[key] > 0][0]
            ax[i_plot].set_xlabel("l")
            ax[i_plot].set_ylabel("P(l)")
            ax[i_plot].set_xlim(r_min, r_max)
            i_plot += 1

        # Plot angle distributions
        if "angle" in self.sim.property:
            for i, key in enumerate(self.sim.property["angle"].keys()):
                angle = self.sim.property["angle"][key]
                ax[i_plot].plot(self.sim.r_angle, angle, label=key)
                ax[i_plot].plot(self.sim.r_angle, self.sim.angle_dist_ref[key], "k--", lw=1)
            ax[i_plot].set_xlabel("theta")
            ax[i_plot].set_ylabel("P(theta)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.sim.directory, "results.jpg"))
        plt.close(fig)

    def plot_potentials(self):
        n_subplots = 1
        if len(self.sim.bond_types) > 0:
            n_subplots += 1
        if len(self.sim.angle_types) > 0:
            n_subplots += 1

        fig, ax = plt.subplots(1, n_subplots, figsize=[2 * n_subplots, 3], dpi=300)
        ax = np.atleast_1d(ax)

        i_plot = 0
        y_min = 0
        for key in self.sim.e_pot:
            ax[i_plot].plot(self.sim.r_pot, self.sim.e_pot[key], label=key)
            y_min = min(y_min, self.sim.e_pot[key].min())
        ax[i_plot].set_ylim(y_min, 2)
        ax[i_plot].legend(fontsize=5)
        ax[i_plot].set_xlabel("r")
        ax[i_plot].set_ylabel("E(r)")
        i_plot += 1

        if len(self.sim.bond_types) > 0:
            for key in self.sim.e_bond:
                ax[i_plot].plot(self.sim.r_bond, self.sim.e_bond[key], label=key)
            ax[i_plot].legend(fontsize=5)
            ax[i_plot].set_xlabel("l")
            ax[i_plot].set_ylabel("E(l)")
            i_plot += 1

        if len(self.sim.angle_types) > 0:
            for key in self.sim.e_angle:
                ax[i_plot].plot(self.sim.r_angle, self.sim.e_angle[key], label=key)
            ax[i_plot].legend(fontsize=5)
            ax[i_plot].set_xlabel("theta")
            ax[i_plot].set_ylabel("E(theta)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.sim.directory, "potentials.jpg"))
        plt.close(fig)

