# Exploring the effect of the new terms in the mean-field equations

# This is making things analytically much more difficult
# Maybe the effect is negligible, however

# TODO: Write a function which solves for lambda given delta
# then looks at the change on delta and then solves for lambda
# Hopefully this effect will be monotonic and convergent but we'll see

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.misc import derivative

import matplotlib.colors as colors
import matplotlib.cm as cmx


def digamma_inv(y):
    """
    Inverse digamma function
    Returns x given y: psi(x) = y
    TODO: make this part vectorized from the beginning
    """

    x = (np.exp(y) + 0.5) if (y >= -2.22) else (-1 / (y + special.digamma(1)))

    # Using Newton-Raphson for this monotonic function
    # May have to change to a convergence condition
    for i in range(1, 20):
        x = x - (special.digamma(x) - y) / special.polygamma(1, x)

    return x


def MF_delta(T, k):
    """
    Find the value of delta predicted by the mean-field equations
    """

    digamma_inverse = np.vectorize(digamma_inv)

    psi_tilde = np.log(1 / (2 * np.pi * T))
    - 0.5 * np.log(rho * J) + (1 - 1 / z2(T, k)) / (rho * J)

    argument_tilde = digamma_inverse(psi_tilde)
    delta = np.multiply((argument_tilde - 0.5) * (2 * np.pi / z2(T, k)), T)

    return delta


def plot_graphical_solution():
    """
    Seeing how the implicit solution for lambda depends on delta
    """

    Ts = np.linspace(0.1, 1, 10)
    lambdas = np.linspace(0, 10, 100)

    linear_part = np.zeros((np.size(Ts), np.size(lambdas)))
    fermi_part = np.zeros((np.size(Ts), np.size(lambdas)))

    for i in range(np.size(Ts)):
        T = Ts[i]

        K = 0.5 - 0.5 * np.sqrt(- 0.5 * rho * J * np.log(rho * J))
        k = K / (1 + np.exp(- K * lambdas / T))
        delta = MF_delta(T, k)

        linear_part[i, :] = np.pi * J * rho * lambdas / (2 * delta)
        fermi_part[i, :] = (1 - 2 * k) / (k * (1 - k))

    # Plot the graphical solution, using colours

    cm = plt.get_cmap('inferno')
    cNorm = colors.Normalize(vmin=np.min(Ts), vmax=np.max(Ts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(np.size(Ts)):
        colorVal = scalarMap.to_rgba(Ts[i])

        plt.plot(lambdas, linear_part[i, :], "r-",
                 label=r'$ \pi J \rho \lambda_{SC} / 2 \Delta $',
                 color=colorVal)
        plt.plot(lambdas, fermi_part[i, :], "b-",
                 label=r'$ (1 - 2 \kappa) / \kappa (1 - \kappa) $',
                 color=colorVal)

    plt.xlabel(r'$ \lambda_{SC} $', fontsize=26)
    plt.ylabel(r'$ f(\lambda_{SC}) $', fontsize=26)
    # plt.legend(loc='upper right', fontsize=26, frameon=False)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)
    ax.set_ylim([0, 6])
    ax.set_xlim([0, 10])

    plt.savefig("lambda_graphical-solution.pdf", dpi=300,
                format='pdf', bbox_inches='tight')

    plt.clf()


def z2(T, k, B=0):
    """
    Returns the value of z^2 at a particular temperature
    Trying to see if a temperature dependence can remove phase transition
    """

    return 4 * k * (1 - k)


def main():
    # Setting various parameters of the problem

    global rho, J
    rho = 0.4
    J = 0.4

    plot_graphical_solution()


if __name__ == '__main__':
    main()
