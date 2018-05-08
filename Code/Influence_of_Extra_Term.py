# Exploring the effect of the new terms in the mean-field equations

# This is making things analytically much more difficult
# Maybe the effect is negligible, however

# TODO: Write a function which solves for lambda given delta
# then looks at the change on delta and then solves for lambda
# Hopefully this effect will be monotonic and convergent but we'll see

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

    start = (np.exp(y) + 0.5) if (y >= -2.22) else (-1 / (y + special.digamma(1)))

    def inv(x):
        return special.digamma(x) - y

    return np.max(optimize.fsolve(inv, start))


def MF_delta(T, k_T):
    """
    Find the value of delta predicted by the mean-field equations
    """

    z2 = 4 * k_T * (1 - k_T)

    # This line previously had a huge error in it
    psi_tilde = - np.log(2 * np.pi) - np.log(T) - 0.5 * np.log(rho * J) + (1 - 1 / z2) / (rho * J)

    argument_tilde = digamma_inv(psi_tilde)
    delta = (argument_tilde - 0.5) * (2 * T * np.pi / z2)

    #return (delta >= 0) * delta
    return delta


def MF_lambda_SC(Temp):
    """
    Solving for the mean-field value of λ_SC at particular temperature
    i.e. Finds the root of MF_equation_lambda()
    """

    def MF_equation(lambda_SC, T):

        K_T = K(T)

        k_T = K_T / (1 + np.exp(- K_T * lambda_SC / T))

        constant_part = np.pi * J * rho * lambda_SC / 2
        difficult_part = MF_delta(T, k_T) * (1 - 2 * k_T) / (k_T * (1 - k_T))

        return constant_part - difficult_part

    MF_lambda_SC = optimize.fsolve(MF_equation, 0, args=(Temp))

    return MF_lambda_SC


def F(T, delta, lambda_SC, k_T):
    """
    Return the value of the free energy for temperature, hybridisation field
    """

    f_original = 2 * delta / (np.pi * J * rho) - 4 * T * np.real(
        np.log(special.gamma(0.5 +
                             (1j * z2(T, k_T) * delta +
                              np.exp(1 / (rho * J)) / np.sqrt(rho * J)) /
                             (2j * np.pi * T)) /
               special.gamma(0.5 + (z2(T, k_T) * delta /
                                    (2 * np.pi * T)))))

    f_extra = lambda_SC * k_T - T * np.log(1 + np.exp(K(T) * lambda_SC / T))

    f = f_original + f_extra

    return f


def MF_F(T):
    """
    Return the mean-field free energy at a given temperature
    """

    # Set the soft-constraint parameter
    K_T = K(T)

    # Calculate the Lagrange multiplier
    lambda_SC = MF_lambda_SC(T, K_T)

    # Include the temperature dependent occupation
    k_T = k(lambda_SC, T, K_T)

    # Calculate the value of the hybridisation field
    delta = MF_delta(T, k_T)

    f = F(T, delta, lambda_SC, k_T)

    return f


def K(T):
    """
    Returns the value for the soft-constraint parameter K
    This is the standard value
    """

    K_0 = 0.5 - 0.5 * np.sqrt(1 - 1 / (1 - 0.5 * rho * J * np.log(rho * J)))

    return K_0


def k(lambda_SC, T, K_T):
    """
    Returns the value of the temperature dependent κ
    """

    return K_T / (1 + np.exp(- K_T * lambda_SC / T))

def k_smooth(T):
    """
    Returns a value of (constant) kappa that seems to remove a phase transition
    """

    change = 0.018
    gradient = 4.25

    return (change / np.cosh(5 * T)) + K(T) + change * np.tanh(gradient * (T - np.exp(- special.digamma(0.5)) / (2 * np.pi)))


def plot_lambda_vs_T():
    """
    Plotting the value of lambda against temperature
    Mainly for debugging the parametric plot
    """

    Ts = np.linspace(0.2, 1.2, 250)

    lambdas = np.zeros(np.size(Ts))
    ss = np.zeros(np.size(Ts))

    for i in range(np.size(Ts)):
        T = Ts[i]

        lambdas[i] = MF_lambda_SC(T)
        ss[i] = lambdas[i] / T

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(Ts, lambdas, "k-")
    plt.plot(Ts, ss, "r-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \beta \lambda $', fontsize=26)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("s_vs_T_solve_OG.pdf",
                dpi=300, format='pdf', bbox_inches='tight')


def plot_delta_vs_T():
    """
    Plots the *new* behaviour of the order parameter with temperature
    Includes the new temperature dependence of kappa
    """

    # Measure T in units of T_K
    Ts = np.linspace(0.01, 1.5, 250)

    lambdas = np.zeros(np.size(Ts))
    ks = np.zeros(np.size(Ts))

    deltas = np.zeros(np.size(Ts))
    deltas_up = np.zeros(np.size(Ts))
    deltas_down = np.zeros(np.size(Ts))

    deltas_interp = np.zeros(np.size(Ts))

    print("K = " + str(K(0.6)))
    for i in range(np.size(Ts)):

        T = Ts[i]

        #lambda_SC = MF_lambda_SC(T, K(T))
        deltas[i] = MF_delta(T, K(T))

        #lambda_SC_up = MF_lambda_SC(T, K_exp(T))
        deltas_up[i] = MF_delta(T, K(T) + 0.018)

        #lambda_SC_down = MF_lambda_SC(T, 0.5)
        deltas_down[i] = MF_delta(T, K(T) - 0.018)

        # WORKS!?
        deltas_interp[i] = MF_delta(T, (0.0095 / np.cosh(5 * T)) + K(T) + 0.0095 * np.tanh(8 * (T - np.exp(- special.digamma(0.5)) / (2 * np.pi))))

        deltas_interp[i] = MF_delta(T, k_smooth(T))


        #lambdas[i] = lambda_SC
        #ks[i] = k(lambda_SC, T, K(T))

    #z2s = 4 * np.multiply(ks, (1 - ks))

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.fill_between(np.linspace(-0.2, np.max(Ts)+0.2, 10),
                     0, -0.5, color='#dddddd')

    plt.fill_between(Ts, deltas_up, deltas_down,
                     facecolor='blue', alpha=0.5)

    plt.plot(Ts, deltas, "k-", label=r'$ \kappa_0 $')
    plt.plot(Ts, deltas_interp, "r-", label=r'$ \kappa (T) $')

    plt.plot(Ts, deltas_down, "b:", label=r'$ \kappa_0 - \delta $')
    plt.plot(Ts, deltas_up, "b--", label=r'$ \kappa_0 + \delta $')


    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)
    plt.legend(fontsize=22, frameon=False)

    ax = plt.gca()
    ax.set_xlim([0, np.max(Ts)])
    ax.set_ylim([-0.1, 1.1 * np.max(deltas)])
    ax.tick_params(axis='both', labelsize=20)

    plt.axhline(y=0, linestyle='--', color='k')

    plt.savefig("range_delta_vs_T.pdf", dpi=300,
                format='pdf', bbox_inches='tight', transparent=True)
    plt.clf()


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

        k_T = k(lambdas, T, K(T))
        delta = MF_delta(T, k_T)

        linear_part[i, :] = np.pi * J * rho * lambdas / (2 * delta)
        fermi_part[i, :] = (1 - 2 * k_T) / (k_T * (1 - k_T))

    # Plot the graphical solution, using colours

    cm = plt.get_cmap('inferno')
    cNorm = colors.Normalize(vmin=np.min(Ts), vmax=np.max(Ts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(np.size(Ts)):
        colorVal = scalarMap.to_rgba(Ts[i])

        plt.plot(lambdas, linear_part[i, :],
                 label=r'$ \pi J \rho \lambda_{SC} / 2 \Delta $',
                 color=colorVal)
        plt.plot(lambdas, fermi_part[i, :],
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


def plot_F_vs_T():
    """
    Plotting the second derivative of free energy wrt temperature
    """

    Tc = np.exp(- special.digamma(0.5)) / (2 * np.pi)
    Ts = np.linspace(0.05, Tc + 0.2, 1001)

    MF_Fs = np.zeros(np.size(Ts))

    for i in range(np.size(Ts)):
        T = Ts[i]
        MF_Fs[i] = MF_F(T)

    # MF_d2Fs = derivative(MF_F, Ts, n=2, dx=0.001, order=3)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(Ts, MF_Fs, "r-", label="Mean-Field C")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ F / T_K $', fontsize=26)

    ax = plt.gca()
    ax.set_xlim([0, np.max(Ts)])

    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("new_F_vs_T.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    # Setting various parameters of the problem

    global rho, J
    rho = 0.4
    J = 0.4

    #plot_graphical_solution()
    plot_delta_vs_T()
    #plot_F_vs_T()
    #plot_lambda_vs_T()

if __name__ == '__main__':
    main()
