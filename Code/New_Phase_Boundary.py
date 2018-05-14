# Trying to investigate the phase boundary with non-zero magnetic field

import numpy as np
import scipy.special as special


global rho, J
rho = 0.4
J = 0.4


def Tc_stochastic(b):
    """
    Defining the parametric T in terms of difficult expressions
    Calculated using a stochastic magnetic field
    """

    z_squared = z2_stochastic(b)

    t = (1 / (2 * np.pi * np.sqrt(rho * J))) *\
        np.exp((1 - 1 / z_squared) / (rho * J)) *\
        np.exp(np.real(- 4j * np.pi * special.loggamma(0.5 + b / (4j * np.pi)) / b))

    return t

def Tc(b):
    """
    Defining the parametric T in terms of b and other quantities
    Not using a stochastic magnetic field
    """

    t = (1 / (2 * np.pi * np.sqrt(rho * J))) *\
        np.exp((1 - 1 / z2_saturated(b)) / (rho * J)) *\
        np.abs(np.exp(- special.digamma(0.5 + b / (2j * np.pi))))

    return t


def z2_stochastic(b):
    """
    Defining the KR operator for the case of a stochastic B field
    This essentially makes the function constant
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))
    K = 2 * K

    return K * (2 - K) + 0 * b


def z2_sat(b):
    """
    Defining the KR operator in terms of a B-field parameter, but treating the high-B case more carefully
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))
    K = 2 * K

    im_psi = np.imag(special.digamma(0.5 + b / (2j * np.pi)))
    diff_squared = np.square(0.5 - K / 4) - np.square(im_psi / np.pi)

    p2_up = (1 / 2 - K / 4) + im_psi / np.pi
    p2_down = (1 / 2 - K / 4) - im_psi / np.pi

    z_squared = 0
    if (p2_up <= 0 or p2_down <= 0):
        z_squared = (1 - K / 2) / (1 - K / 4)
    else:
        z_squared = (K / 4) * np.square(np.sqrt(p2_up) + np.sqrt(p2_down)) / ((K / 4 + p2_up) * (K / 4 + p2_down))

    return z_squared


z2_saturated = np.vectorize(z2_sat)