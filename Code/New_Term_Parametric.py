# Trying to recreate previous plots without resorting to the Newton-Raphsen method

import numpy as np
import scipy.special as special
import scipy.optimize as optimize


global rho, J
rho = 0.4
J = 0.4


def MF_lambda_SC(Temp):
    """
    Solving for the Lagrange multiplier through brute force
    """

    K_T = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    def MF_equation_lambda(lambda_SC, T):

        k_T = K_T / (1 + np.exp(- K_T * lambda_SC / T))
        z2 = 4 * k_T * (1 - k_T)

        eq = special.digamma(0.5 + J * rho * z2 * z2 * lambda_SC /
                             (16 * T * (1 - 2 * k_T))) +\
            np.log(T) + np.log(2 * np.pi) +\
            0.5 * np.log(rho * J) - (1 - 1 / z2) / (rho * J)

        return eq

    return optimize.fsolve(MF_equation_lambda, 0, args=(Temp))


def F(s):
    """
    Returns the mean field free energy
    Function of the non-affine parameter
    """

    z2 = 4 * k(s) * (1 - k(s))
    D = np.exp(1 / (rho * J)) / np.sqrt(rho * J)

    F_orig = (s * z2 / (2 * (1 - 2 * k(s))) -
              4 * np.real(
        special.loggamma(0.5 +
                         s * J * rho * z2 * z2 / (16 * (1 - 2 * k(s))) +
                         D / (2 * np.pi * 1j * t(s))) -
        special.loggamma(0.5 +
                         s * J * rho * z2 * z2 / (16 * (1 - 2 * k(s)))))
              ) * t(s)

    F_extra = t(s) * (K(s) * s / (1 + np.exp(- K(s) * s)) -
                      np.log(1 + np.exp(- K(s) * s)))

    return (F_orig + F_extra)

def K(s):
    """
    Returns the value of the soft-constraint parameter
    """

    K_0 = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    return K_0


def k(s):
    """
    Returns the value of the modified soft-constraint parameter
    Modification comes from thermal occupation of "empty" pseudo-fermions
    """

    return K(s) / (1 + np.exp(- K(s) * s))


def t(s):
    """
    Returns normalised temperature for given value of non-affine parameter
    """

    T = (1 / (2 * np.pi)) * (1 / np.sqrt(rho * J)) *\
        np.exp((1 - 1 / (4 * k(s) * (1 - k(s)))) / (rho * J)) *\
        np.exp(- special.digamma(
               0.5 + J * rho * s * np.square(k(s) * (1 - k(s))) /
               (1 - 2 * k(s))))

    return T


def delta(s):
    """
    Returns the mean-field hybridisation field at a given non-affine parameter
    """

    k_T = k(s)
    z2 = 4 * k_T * (1 - k_T)

    d = np.pi * J * rho * z2 * s * t(s) / (8 * (1 - 2 * k_T))

    return d