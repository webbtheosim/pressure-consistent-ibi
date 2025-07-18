import numpy as np

def lj(r, epsilon1, epsilon2, sigma, inner_cut=2):
    e_pot = 4 * (epsilon1 * (sigma / r) ** 12 - epsilon2 * (sigma / r) ** 6)
    # linear extrapolation using the slope at r=4
    idx = np.argmin(np.abs(r - inner_cut))
    slope = (e_pot[idx + 1] - e_pot[idx]) / (r[idx + 1] - r[idx])
    e_pot[:idx] = e_pot[idx] + slope * (r[:idx] - r[idx])
    return e_pot

def harmonic(r, k, r0):
    return 0.5 * k * (r - r0) ** 2

def morse(r, D, alpha, r0):
    """
    Morse potential
    :param r: distance
    :param D: depth of the potential well
    :param alpha: width of the potential well
    :param r0: equilibrium distance
    :return: potential energy
    """
    return D * (1 - np.exp(-alpha * (r - r0))) ** 2

def rscale_table_pot(r, e, scale):
    # r_scaled = r*scale # scale from the head
    scale = 1/scale
    # interpolate r and e to a very dense grid
    r_dense = np.linspace(r[0], r[-1], 100000)
    e_dense = np.interp(r_dense, r, e)
    r_scaled_dense = (r_dense - r_dense[-1]) * scale + r_dense[-1]  # scale from the tail
    # corresponding e to the original r in the scaled r
    e_scaled_dense = np.interp(r_dense, r_scaled_dense, e_dense)
    # interpolate back to the original r
    e_scaled = np.interp(r, r_dense, e_scaled_dense)
    e_scaled -= e_scaled[-1]
    return r, e_scaled

def shift_table_pot(r, e, delta):
    r_cutoff_right = r[-1]
    r_cutoff_left = r[0]
    # we first think the shift delta is positive
    if delta > 0:
        # shifted r values
        r_shifted = r + delta
        # cut the right end
        e_shifted = e[r_shifted <= r_cutoff_right]
        # linear extrapolation on the left end
        p1 = np.polyfit(r_shifted[:5], e_shifted[:5], 1)
        r_extra = r[r < r_shifted[0]]
        offset = e_shifted[0] - r_shifted[0] * p1[0]
        e_extra = p1[0] * r_extra + offset
        # combine the extrapolated part and the shifted part
        e_new = np.concatenate([e_extra, e_shifted])
    else:
        # shifted r values
        r_shifted = r + delta
        # cut the left end
        e_shifted = e[r_shifted >= r_cutoff_left]
        # linear extrapolation on the right end
        p1 = np.polyfit(r_shifted[-5:], e_shifted[-5:], 1)
        r_extra = r[r > r_shifted[-1]]
        offset = e_shifted[-1] - r_shifted[-1] * p1[0]
        e_extra = p1[0] * r_extra + offset
        # combine the extrapolated part and the shifted part
        e_new = e_new = np.concatenate([e_shifted, e_extra])
    e_new -= e_new[-1]
    # apply a window function to make the gradient of e_new to zero at the tail
    G = np.exp(-(0.5**2) / (r - r[-1] + 1e-8) ** 2)
    e_new = e_new * G
    return r, e_new

def apply_tail_window(r, e, window_width=5):
    """
    Apply a window function to the tail of the potential to make the gradient of the potential to zero at the tail
        :param r: r values
        :param e: e values
        :param window_width: the width of the window function
        :return: r, e
    """
    G = np.exp(-(window_width**2) / (r - r[-1] + 1e-8) ** 2)
    e = e * G
    return r, e

def linear_attra(r, e_pot, A, rcut):
    e_pot[r>rcut] = e_pot[r>rcut] + A * (r[-1] - r[r>rcut])
    e_pot[r<=rcut] = e_pot[r<=rcut] + A * (r[-1] - r[r>rcut][0])
    return e_pot
