import numpy as np
import matplotlib.pyplot as plt

with open('log.txt') as f:
    data = f.readlines()
    error11, error12, error22 = [], [], []
    # error2 = []
    # error3 = []
    density = []
    alpha = []
    step = []
    for line in data:
        step.append(int(line.split('=')[1].split(' ')[0]))
        density.append(float(line.split('=')[2].split(' ')[0]))
        error11.append(float(line.split('=')[3].split(' ')[0]))

# error = np.mean((error11, error12, error22), axis=0)
error = np.array(error11)
np.savetxt('plot_density.txt', np.array([step, error, density]).T, header='step error density ')

# density_ref = np.loadtxt('density_ref.txt')[0]
density_ref = 0.8635989985280287939
min_error = 1000
for i in range(len(density)):
    if abs(density[i] - density_ref) < 0.005*density_ref:
        if error[i] < min_error:
            min_error = error[i]
            step_min = step[i]
print('minimum error at iteration:', step_min)
# print('minimum error1 at iteration:', step[np.argmin(error)])

# # standard IBI in nvt
# density_IBI = np.loadtxt('../CG500_nvt/density.dat')

# # standard IBI in npt
# density_IBI_npt = np.loadtxt('../CG500_nvt_npt2/density.dat')

fig, ax = plt.subplots(2, 1, dpi=300, figsize=(4, 3), sharex=True)
ax[0].plot(range(len(density)), density, 'o-', ms=2, label='IBI-density')
# plt.plot(range(len(density_IBI)),density_IBI, 's-',ms=2,label='Standard IBI nvt')
# plt.plot(range(len(density_IBI_npt)),density_IBI_npt, '^-',ms=2,label='Standard IBI npt')
ax[0].plot([0, len(density)], [density_ref, density_ref], 'k--', label='AA-density')
ax[1].set_xlabel('IBI iterations')
ax[0].set_ylabel('density')
ax[0].legend()
ax[1].plot(range(len(density)), error11, 'o-', ms=2, label='error1')
# ax[1].plot(range(len(density)), error12, 's-', ms=2, label='error2')
# ax[1].plot(range(len(density)), error22, '^-', ms=2, label='error3')
ax[1].set_ylabel('RDF error')
ax[1].set_yscale('log')
plt.tight_layout()
plt.savefig('density.png')
