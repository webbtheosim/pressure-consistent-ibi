# Pressure-matching IBI
tools for performing pressure-matching iterative Boltzmann inversion (IBI) for coarse-grained molecular dynamics simulation

## Introduction

Bottom-up coarse-grained (CG) methods enable efficient simulations of complex molecular systems at mesoscopic scales. Among them, iterative Boltzmann inversion (IBI) is one of the simplest yet widely used approaches, optimizing pairwise potentials to reproduce radial distribution functions (RDF). While IBI effectively captures structural correlations, it often suffers from thermodynamic inconsistencies, particularly in reproducing the correct pressure of the all-atom reference system.

In this repository, we present simple yet effective extensions to IBI—iLS (iterative length-scale rescaling) and iLC (iterative linear correction)—which incorporate pressure corrections directly into the IBI framework.  This repository contains the core code for implementing and testing these methods.

## Installation and Dependencies

Clone the repository and set up the required environment:
```
git clone https://github.com/zyumse/IBI_PM.git
cd gECG_thiophene/

# create virtual environment 
conda create --name your_env_name python=3.11
conda activate your_env_name

# Install Python package
pip install -e . 
```
After installation, create a new terminal and activate the environment. `ibisim` will be available to use

## Usage
### CG mapping of all-atom simulation
1. Perform MD simulation at the all-atom level

    Here, we employe [Lammps](https://github.com/lammps/lammps) as the MD engine, but others work too 

2. Perform CG mapping

   One could use the ['scripts/run_map.py'](scripts/run_map.py) for assistance, which will covert the all-atom lammps data file and trajectory to the corresponding CG ones based on a user defined mapping.

3. Extract structure properties and density

   One could use ['scripts/get_str.py'](scripts/get_str.py) for assistance, which will compute the pdfs, and several other structural properties. These serve as targets in the IBI.

### Pressure-matching IBI

Create a separate folder for performing IBI. In the folder, one needs 
- A YAML config file for specifying IBI settings. See ['test/toluene/config.yaml'](test/toluene/config.yaml) for an example. 
- A lammps input file
- An initial CG configuration lammps data file

Pressure matching features are controlled by parameters in the config file. Turn on `density_correction: True` and set reasonable `density_correction_freq`. For 
1. IBI-iRT
   - `PM: rscale`
   - `gamma: 0.1`

2. IBI-iLC
   - `PM: linear`
   - `A: 0.01`

3. other variants of IBI
   - regular IBI: `density_correction: False`
   - NVT or NPT is controlled in in.lmp
   - tail correction: after performing IBI-NVT and then do `PM: linear` with `alpha: 0`

## Example (test)

An example of 1-site toluene model is provided in [test/toluene](test/toluene/). In the folder, one can use `ibisim` to test the package. Note that the simulation time has been reduced for testing efficiency; one should use a longer time for converged performance. 

