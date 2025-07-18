import numpy as np
from . import tools_lammps as tl
import datetime


def map_CG_one(coors_AA, box_size, mass_AA):
    """
    map the all-atom coordinates to CG coordinate
    inputs:
        coors_AA: all-atom coordinates
        box_size: box size of the system (cubic box)
        mass_AA: mass of all-atom particles
    return coord_CG: CG coordinates
    """
    rcoors_AA = coors_AA - coors_AA[0]
    rcoors_AA = rcoors_AA - np.rint(rcoors_AA/box_size)*box_size
    coord_CG = coors_AA[0] + np.sum(rcoors_AA*mass_AA.reshape(-1, 1), axis=0)/np.sum(mass_AA)
    return coord_CG


def find_neighbor(atom_index, bond_info):
    C1 = bond_info[bond_info[:, 2] == atom_index, 3]
    C2 = bond_info[bond_info[:, 3] == atom_index, 2]
    neighbors = np.concatenate([C1, C2]).astype(int)
    return neighbors

def compute_bond_length(coors, bonded_atoms, box_size):
    vector = coors[bonded_atoms[:, 0]]-coors[bonded_atoms[:, 1]]
    vector = vector - np.rint(vector/box_size)*box_size
    bond_length = np.linalg.norm(vector, axis=1)
    return bond_length


def compute_angle(coors, angle_atoms, box_size):
    vector1 = coors[angle_atoms[:, 0]]-coors[angle_atoms[:, 1]]
    vector2 = coors[angle_atoms[:, 2]]-coors[angle_atoms[:, 1]]
    vector1 = vector1 - np.rint(vector1/box_size)*box_size
    vector2 = vector2 - np.rint(vector2/box_size)*box_size
    angle = np.arccos(np.sum(vector1*vector2, axis=1) /
                      (np.linalg.norm(vector1, axis=1)*np.linalg.norm(vector2, axis=1)))
    return angle/np.pi*180


def pdf_sq_1type(box_size, natom, coors, r_cutoff=10, delta_r=0.01):
    """
    only one type of particles
    inputs: box,natom,type_atom,coors,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    box = np.array([[box_size, 0, 0],
                    [0, box_size, 0],
                    [0, 0, box_size]])
    n1 = natom
    rcoors = np.dot(coors, np.linalg.inv(box))
    rdis = np.zeros([natom, natom, 3])
    for i in range(natom):
        tmp = rcoors[i]
        rdis[i, :, :] = tmp - rcoors
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1 / V
    c = np.array([rho1 * rho1]) * V
    g1 = np.histogram(dis[:n1, :n1], bins=r)[0] / (4 * np.pi *
                                                   (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross_mask(box, coors1, coors2,  mask_matrix, r_cutoff:float=10, delta_r:float=0.01):
    """
    compute pdf and sq for two types of particles (can be same type)
    inputs: box,coors1,coors2,bond_atom_idx,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    n1 = len(coors1)
    n2 = len(coors2)
    natom = n1 + n2
    rcoors1 = np.dot(coors1, np.linalg.inv(box))
    rcoors2 = np.dot(coors2, np.linalg.inv(box))
    rdis = np.zeros([n1, n2, 3])
    for i in range(n1):
        tmp = rcoors1[i]
        rdis[i, :, :] = tmp - rcoors2
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))

    dis = dis * mask_matrix

    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1/V
    rho2 = n2/V
    c = np.array([rho1 * rho2]) * V
    g1 = np.histogram(dis, bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1


def pdf_sq_cross(box, coors1, coors2,  bond_atom_idx, r_cutoff:float=10, delta_r:float=0.01):
    """
    compute pdf and sq for two types of particles (can be same type)
    inputs: box,coors1,coors2,bond_atom_idx,r_cutoff=10,delta_r = 0.01
    outputs: R,g1,Q,S1
    """
    # check if coors1 and coors2 are exactly the same
    if np.array_equal(coors1, coors2):
        is_same = True
    else:
        is_same = False

    # type_atom = np.array(type_atom)
    n1 = len(coors1)
    n2 = len(coors2)
    natom = n1 + n2
    rcoors1 = np.dot(coors1, np.linalg.inv(box))
    rcoors2 = np.dot(coors2, np.linalg.inv(box))
    rdis = np.zeros([n1, n2, 3])
    for i in range(n1):
        tmp = rcoors1[i]
        rdis[i, :, :] = tmp - rcoors2
    rdis[rdis < -0.5] = rdis[rdis < -0.5] + 1
    rdis[rdis > 0.5] = rdis[rdis > 0.5] - 1
    a = np.dot(rdis[:, :, :], box)
    dis = np.sqrt((np.square(a[:, :, 0]) + np.square(a[:, :, 1]) + np.square(a[:, :, 2])))
    # Exclude bonded atoms by setting the distances to NaN
    if bond_atom_idx is not None:
        for bond_pair in bond_atom_idx:
            i1 = int(bond_pair[0])
            i2 = int(bond_pair[1])
            dis[i1, i2] = np.nan
            if is_same:
                dis[i2, i1] = np.nan

    r_max = r_cutoff
    r = np.linspace(delta_r, r_max, int(r_max / delta_r))
    V = np.dot(np.cross(box[1, :], box[2, :]), box[0, :])
    rho1 = n1/V
    rho2 = n2/V
    c = np.array([rho1 * rho2]) * V
    g1 = np.histogram(dis, bins=r)[0] / (4 * np.pi * (r[1:] - delta_r / 2) ** 2 * delta_r * c[0])
    R = r[1:] - delta_r / 2

    dq = 0.01
    qrange = [np.pi / 2 / r_max, 25]
    Q = np.arange(np.floor(qrange[0] / dq), np.floor(qrange[1] / dq), 1) * dq
    S1 = np.zeros([len(Q)])
    rho = natom / np.dot(np.cross(box[1, :], box[2, :]), box[0, :]) #/ 10 ** 3
    # use a window function for fourier transform
    for i in np.arange(len(Q)):
        S1[i] = 1 + 4 * np.pi * rho / Q[i] * np.trapz(
            (g1 - 1) * np.sin(Q[i] * R) * R * np.sin(np.pi * R / r_max) / (np.pi * R / r_max), R)

    return R, g1, Q, S1
