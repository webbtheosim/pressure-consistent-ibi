# reweighting functions
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import datetime

def read_table_pot(file_name, key_words):
    """
    input: file_name, key_words
    output: r, e_pot, f_pot
    """
    r = []
    e_pot = []
    f_pot = []
    with open(file_name, 'r') as f:
        while True:
            line = f.readline()
            if key_words in line:
                meta_info = f.readline()
                f.readline()
                for _ in range(int(meta_info.split()[1])):
                    line = f.readline()
                    data = line.split()
                    r.append(float(data[1]))
                    e_pot.append(float(data[2]))
                    f_pot.append(float(data[3]))
                break
    r = np.array(r)
    e_pot = np.array(e_pot)
    f_pot = np.array(f_pot)
    return r, e_pot, f_pot

def write_pot_table(file_name, zip_FF):
    """
    write the two-body potential to a table file for lammps simulations
    inputs: file_name, zip_FF (zip of r, E, F, key)
    """
    t = datetime.date.today().strftime('%m/%d/%Y')
    f = open(file_name, 'w')
    f.write('# DATE: {}  UNITS: real \n'.format(t))
    f.write('# potential\n\n')
    # zip_FF contains r, E, F, 'keys' of each interaction
    # unzip the zip_FF, check how many types of interactions
    num_types = len(zip_FF)
    for i in range(num_types):
        r, E, F, key = zip_FF[i]

        f.write(f'{key}\n')
        f.write('N {0:}\n'.format(len(r)))
        f.write('\n')
        for i in range(len(r)):
            f.write('   {0:5d}      {1:12.8f} {2:21f} {3:21f}\n'.format(
                i+1, r[i], E[i], F[i]))
        f.write('\n')
    f.close()

def rweight(r, r0, delta=0.5):
    weight = 1 / 2 * (1 + np.tanh((r0 - r) / delta))
    return weight

def rweight_back(r, r0, delta=0.5):
    weight = 1 / 2 * (1 - np.tanh((r0 - r) / delta))
    return weight

def lowpass(U, r, cutoff):
    U_k = fft(U)
    freqs = fftfreq(len(U), r[1] - r[0])
    U_k[np.abs(freqs) > cutoff] = 0
    return np.real(ifft(U_k))
