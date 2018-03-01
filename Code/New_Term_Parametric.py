# Trying to recreate previous plots without resorting to iterative solutions

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
import scipy.optimize as optimize

import matplotlib.colors as colors
import matplotlib.cm as cmx


def digamma_inv(y):
    """
    Inverse digamma function
    Returns x given y: psi(x) = y
    """

    def starter(yi):
        start_pt = -1 / (yi + special.digamma(1))

        if (yi >= -2.22):
            start_pt = np.exp(yi) + 0.5

        return start_pt

    def inv(x):
        return special.digamma(x) - y

    return np.max(optimize.fsolve(inv, np.vectorize(starter)(y)))


# TODO: Fix the vectorised form of this equation
def MF_lambda_SC(Temp):

    K = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    def MF_equation_lambda(lambda_SC, T):

        k_T = K / (1 + np.exp(- K * lambda_SC / T))
        z2 = 4 * k_T * (1 - k_T)

        eq = special.digamma(0.5 + J * rho * z2 * z2 * lambda_SC /
                             (16 * T * (1 - 2 * k_T))) +\
            np.log(T) + np.log(2 * np.pi) +\
            0.5 * np.log(rho * J) - (1 - 1 / z2) / (rho * J)

        return eq

    return optimize.fsolve(MF_equation_lambda, 0, args=(Temp))

def plot_eq_vs_lambda():
    """
    Investigating the nature of the MF equation
    """

    T = 0.6

    lambdas = np.linspace(-20, 20, 250)
    lambda_soln = optimize.fsolve(MF_equation_lambda, 20, args=(T))

    eqs = np.zeros(np.size(lambdas))

    for i in range(np.size(eqs)):
        eqs[i] = MF_equation_lambda(lambdas[i], T)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(lambdas, eqs, "r-", label=("T = " + str(T)))
    plt.plot(lambda_soln, MF_equation_lambda(lambda_soln, T), "k.")

    plt.xlabel(r'$ \lambda $', fontsize=26)
    plt.ylabel(r'$ MF_{eq}(\lambda) $', fontsize=26)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)
    ax.legend()

    plt.savefig("eq_vs_lambda.pdf",
                dpi=300, format='pdf', bbox_inches='tight')


def plot_lambda_vs_T():
    """
    Trying to nail down the form of lambda_SC against temperature
    """

    Ts = np.linspace(0.2, 1.2, 250)

    lambdas = np.zeros(np.size(Ts))

    ss_parametric = np.linspace(0, 45, 1000)
    ts = t(ss_parametric)
    lambdas_parametric = np.multiply(ss_parametric, ts)

    for i in range(np.size(Ts)):
        T = Ts[i]

        lambdas[i] = MF_lambda_SC(T)
        lambdas[i] = (lambdas[i] >= 0) * lambdas[i]

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(Ts, lambdas, "r-", label="fsolve")
    plt.plot(ts, lambdas_parametric, "k--", label="parametric")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \lambda $', fontsize=26)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)
    ax.legend()

    plt.savefig("s_vs_T.pdf",
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
    deltas = delta(ss)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #plt.fill_between(np.linspace(0, np.max(ts), 10),
    #                 0, -0.5, color='#dddddd')

    # Plot the figure
    plt.plot(ts, deltas, "r-")
    plt.plot(ts, ss, "k-")

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
    #plot_eq_vs_lambda()


if __name__ == '__main__':
    main()
