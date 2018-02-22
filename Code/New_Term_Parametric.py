# Trying to recreate previous plots without resorting to iterative solutions

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

import matplotlib.colors as colors
import matplotlib.cm as cmx


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

    d = 2 * np.pi * J * rho *\
        np.square(k(s) * (1 - k(s))) * s * t(s) /\
        (1 - 2 * k(s))

    return d

def k(s):
    """
    Returns the value of the modified soft-constraint parameter
    Modification comes from thermal occupation of "empty" pseudo-fermions
    """

    K = 0.5 - 0.5 * np.sqrt(-0.5 * rho * J * np.log(rho * J))

    return K / (1 + np.exp(- K * s))

def plot_delta_vs_T():
    """
    Plotting delta vs T using a robust parametric plot, with extra term
    """

    ss = np.linspace(0, 40, 1000)

    ts = t(ss)
    deltas = delta(ss)

    z2s = 4 * k(ss) * (1 - k(ss))

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #plt.fill_between(np.linspace(0, np.max(ts), 10),
    #                 0, -0.5, color='#dddddd')

    # Plot the figure
    #plt.plot(ts, deltas, "k-")
    #plt.plot(ss, np.square(z2s) / (1 - 2 * k(ss)), "k-")
    plt.plot(ts, ss, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)

    ax = plt.gca()
    #ax.set_xlim([0, np.max(ts)])
    #ax.set_ylim([-0.18, 1.25])
    ax.tick_params(axis='both', labelsize=20)

    #plt.axhline(y=0, linestyle='--', color='k')

    plt.savefig("new_delta_vs_T_parametric.pdf",
                dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    # Setting various parameters of the problem

    global rho, J
    rho = 0.4
    J = 0.4

    plot_delta_vs_T()


if __name__ == '__main__':
    main()
