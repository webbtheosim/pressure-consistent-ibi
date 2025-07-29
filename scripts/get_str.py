import numpy as np
import pdb
from ibisimulation.prep_cg import LammpsAnalyzer
import os


try: 
    n_cpus = int(os.getenv("SLURM_CPUS_ON_NODE", default=8))
    bl_range = np.arange(0, 10, 0.01)
    angle_range = np.arange(-0.5, 181, 1)
    RDF_cutoff = 15
    RDF_delta_r = 0.01
    r_range = np.arange(0, RDF_cutoff, RDF_delta_r)
    R = (r_range[:-1] + r_range[1:]) / 2
    T = 300

    analyzer = LammpsAnalyzer(AA_file='./cg.data',
                              lmp_mode='full',
                              dump_file='./dump_CG.xyz',
                              freq=1,
                              RDF_cutoff=RDF_cutoff,
                              RDF_delta_r=RDF_delta_r,
                              bl_range=bl_range,
                              angle_range=angle_range)
    results = analyzer.analyze(n_jobs=n_cpus)
    analyzer.plot_and_save()
    analyzer.write_pot_table(T=T)


except Exception as e:
    pdb.post_mortem()

