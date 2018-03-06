import New_Term_Parametric as parametric_SC

import numpy as np
import matplotlib.pyplot as plt


def plot_delta_vs_T():
    """
    Plotting delta vs T using a robust parametric plot, with extra term
    """

    ss = np.linspace(0, 145, 1000)

    ts = parametric_SC.t(ss)
    deltas = parametric_SC.delta(ss)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.fill_between(np.linspace(0, np.max(ts), 10),
                     0, -0.5, color='#dddddd')

    # Plot the figure
    plt.plot(ts, deltas, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)

    ax = plt.gca()
    ax.set_xlim([0, np.max(ts)])
    ax.set_ylim([-0.18, 1.25])
    ax.tick_params(axis='both', labelsize=20)

    plt.axhline(y=0, linestyle='--', color='k')


    plt.savefig("new_delta_vs_T_parametric.pdf",
                dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def plot_F_vs_T():
    """
    Plotting the free energy against temperature
    """

    ss = np.linspace(0, 45, 1000)

    ts = parametric_SC.t(ss)
    Fs = parametric_SC.F(ss)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(ts, Fs, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ F / T_K $', fontsize=26)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("new_F_vs_T_parametric.pdf",
                dpt=300, format='pdf', bbox_inches='tight')
    plt.clf()


def plot_lambda_vs_T():
    """
    Trying to nail down the form of lambda_SC against temperature
    """

    Ts = np.linspace(0.2, 1.2, 250)

    lambdas = np.zeros(np.size(Ts))

    ss_parametric = np.linspace(0, 45, 1000)
    ts = parametric_SC.t(ss_parametric)
    lambdas_parametric = np.multiply(ss_parametric, ts)

    for i in range(np.size(Ts)):
        T = Ts[i]

        lambdas[i] = parametric_SC.MF_lambda_SC(T)
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


def plot_eq_vs_lambda():
    """
    Investigating the nature of the MF equation
    """

    T = 0.6

    lambdas = np.linspace(-20, 20, 250)

    eqs = np.zeros(np.size(lambdas))

    for i in range(np.size(eqs)):
        eqs[i] = MF_equation_lambda(lambdas[i], T)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(lambdas, eqs, "r-", label=("T = " + str(T)))

    plt.xlabel(r'$ \lambda $', fontsize=26)
    plt.ylabel(r'$ MF_{eq}(\lambda) $', fontsize=26)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)
    ax.legend()

    plt.savefig("eq_vs_lambda.pdf",
                dpi=300, format='pdf', bbox_inches='tight')


def main():
    # Setting various parameters of the problem

    global rho, J
    rho = 0.4
    J = 0.4

    plot_delta_vs_T()
    plot_F_vs_T()


if __name__ == '__main__':
    main()
