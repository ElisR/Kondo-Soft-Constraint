"""
Numerically investigating the mean-field solution with constant K
Want to see if Delta = 0 is always a minimal solution
"""

import numpy as np
import scipy.special as special

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx

# Going to be varying Delta and lambda_SC at a fixed temperature


def F_star(Delta, lambda_SC, T):
    """
    Returns the value of the mean-field free energy
    """

    # Defining useful quantities
    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))
    k = k = K / (1 + np.exp(- K * lambda_SC / T))
    z2 = 4 * k * (1 - k)

    F_0 = - 4 * T * np.real(
        special.loggamma(0.5 +
                         (1j * z2 * Delta + D) / (2j * np.pi * T)) -
        special.loggamma(0.5 + (z2 * Delta / (2 * np.pi * T))))

    F_h = k * lambda_SC - T * np.log(1 + np.exp(K * lambda_SC / T))

    F = F_0 + F_h + 2 * Delta / (np.pi * J * rho)

    return F


def MF_Delta(lambda_SC, T):
    """
    Returns the Delta predicted from MF equations for a given lambda, T
    """
    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))
    k = k = K / (1 + np.exp(- K * lambda_SC / T))
    z2 = 4 * k * (1 - k)

    return np.pi * rho * J * z2 * lambda_SC / (8 * (1 - 2 * k))


def plot_F_vs_Delta_and_lambda():
    """
    Making a 2D surface plot of F against the free parameters, Delta and lambda
    """

    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.gca()

    plt.rc('text', usetex=True)

    T = 0.37

    Deltas = np.linspace(0.001, 2, 40)
    lambdas = np.linspace(0.01, 10, 40)

    Deltas_mesh, lambdas_mesh = np.meshgrid(Deltas, lambdas)
    Fs = F_star(Deltas_mesh, lambdas_mesh, T)

    #surf = ax.plot_surface(Deltas_mesh, lambdas_mesh, Fs, cmap=cmx.coolwarm,
    #                       linewidth=0, antialiased=False)

    MF_Deltas = np.zeros(np.size(lambdas))
    F_line = np.zeros(np.size(lambdas))
    for i in range(np.size(lambdas)):
        lambda_SC = lambdas[i]
        Delta = MF_Delta(lambda_SC, T)
        MF_Deltas[i] = Delta
        F_line[i] = F_star(Delta, lambda_SC, T)

    ax.plot(MF_Deltas, F_line, "k-")

    plt.xlabel(r'$ \Delta $', fontsize=20)
    plt.ylabel(r'$ \lambda $', fontsize=20)
    plt.title("T = " + str(T), fontsize=24)

    plt.show()


def main():
    global rho, J, D
    rho = 0.4
    J = 0.4
    D = np.exp(1 / (rho * J)) / np.sqrt(rho * J)

    plot_F_vs_Delta_and_lambda()


if __name__ == '__main__':
    main()
