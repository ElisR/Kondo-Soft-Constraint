# Trying to recreate previous plots without resorting to iterative solutions

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
import scipy.optimize as optimize

import matplotlib.colors as colors
import matplotlib.cm as cmx


def MF_equation_s(s, T):
    """
    The MF equation for lambda_SC and T
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    k_T = K / (1 + np.exp(- K * s))

    z2 = 4 * k_T * (1 - k_T)

    eq = special.digamma(0.5 + J * rho * z2 * z2 * s /
                         (16 * (1 - 2 * k_T))) +\
        np.log(T) + np.log(2 * np.pi) -\
        0.5 * np.log(rho * J) + (1 - 1 / z2) / (rho * J)

    return eq


def MF_equation_lambda(lambda_SC, T):
    """
    The MF equation again
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    K = K

    k_T = K / (1 + np.exp(- K * lambda_SC / T))

    z2 = 4 * k_T * (1 - k_T)

    eq = special.digamma(0.5 + J * rho * z2 * z2 * lambda_SC /
                         (16 * T * (1 - 2 * k_T))) +\
        np.log(T) + np.log(2 * np.pi) -\
        0.5 * np.log(rho * J) + (1 - 1 / z2) / (rho * J)

    return eq


def plot_lambda_vs_T():
    """
    Trying to nail down the form of lambda_SC against temperature
    """

    Ts = np.linspace(0.2, 4, 250)

    lambdas = np.zeros(np.size(Ts))
    ss = np.zeros(np.size(Ts))

    for i in range(np.size(Ts)):
        T = Ts[i]

        lambdas[i] = np.min(optimize.fsolve(MF_equation_lambda, 0, args=(T)))
        lambdas[i] = (lambdas[i] >= 0) * lambdas[i]
        ss[i] = lambdas[i] / T

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Both give the same thing...
    plt.plot(Ts, ss, "r-")
    plt.plot(Ts, lambdas, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \beta \lambda $', fontsize=26)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("s_vs_T_solve.pdf",
                dpi=300, format='pdf', bbox_inches='tight')


def k(s):
    """
    Returns the value of the modified soft-constraint parameter
    Modification comes from thermal occupation of "empty" pseudo-fermions
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    return K / (1 + np.exp(- K * s))


def t(s):
    """
    Returns normalised temperature for given value of non-affine parameter
    Slightly buggy?
    """

    T = (1 / (2 * np.pi)) * (1 / np.sqrt(rho * J)) *\
        np.exp((1 - 1 / (4 * k(s) * (1 - k(s)))) / (rho * J)) *\
        np.exp(- special.digamma(
               0.5 + J * rho * s * np.square(k(s) * (1 - k(s))) /
               (1 - 2 * k(s))))

    return T


def ln_t(s):
    """
    Returns the logarithm of temperature for given s
    Function for debugging
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))
    k_T = K / (1 + np.exp(- K * s))
    z2 = 4 * k_T * (1 - k_T)

    lnt = - (special.digamma(0.5 + J * rho * z2 * z2 * s /
                             (16 * (1 - 2 * k_T))) +
             np.log(2 * np.pi) +
             0.5 * np.log(rho * J) - (1 - 1 / z2) / (rho * J))

    return lnt

def delta(s):
    """
    Returns the mean-field hybridisation field at a given non-affine parameter
    """

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))
    k_T = K / (1 + np.exp(- K * s))
    z2 = 4 * k_T * (1 - k_T)

    d = np.pi * J * rho * z2 * s * t(s) / (8 * (1 - 2 * k_T))

    return d


def plot_delta_vs_T():
    """
    Plotting delta vs T using a robust parametric plot, with extra term
    """

    ss = np.linspace(0, 45, 1000)

    ts = t(ss)
    ln_ts = ln_t(ss)
    deltas = delta(ss)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #plt.fill_between(np.linspace(0, np.max(ts), 10),
    #                 0, -0.5, color='#dddddd')

    # Plot the figure
    plt.plot(ts, deltas, "r-")
    #plt.plot(np.exp(ln_ts), ss, "k-")
    #plt.plot(ts, ss, "r-")
    plt.plot(np.exp(ln_ts), deltas, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)

    ax = plt.gca()
    #ax.set_xlim([0, np.max(ts)])
    #ax.set_ylim([-0.18, 1.25])
    ax.tick_params(axis='both', labelsize=20)

    #plt.axhline(y=0, linestyle='--', color='k')


    plt.savefig("new_s_vs_T_parametric.pdf",
                dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    # Setting various parameters of the problem

    global rho, J
    rho = 0.4
    J = 0.4

    plot_delta_vs_T()
    plot_lambda_vs_T()


if __name__ == '__main__':
    main()
