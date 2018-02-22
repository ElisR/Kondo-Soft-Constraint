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
    TODO: make this part vectorized from the beginning
    """

    x = (np.exp(y) + 0.5) if (y >= -2.22) else (-1 / (y + special.digamma(1)))

    # Using Newton-Raphson for this monotonic function
    # May have to change to a convergence condition
    for i in range(1, 20):
        x = x - (special.digamma(x) - y) / special.polygamma(1, x)

    return x


def MF_delta(T, k_T):
    """
    Find the value of delta predicted by the mean-field equations
    """

    digamma_inverse = np.vectorize(digamma_inv)

    psi_tilde = np.log(1 / (2 * np.pi * T))
    - 0.5 * np.log(rho * J) + (1 - 1 / z2(T, k_T)) / (rho * J)

    argument_tilde = digamma_inverse(psi_tilde)
    delta = np.multiply((argument_tilde - 0.5) * (2 * np.pi / z2(T, k_T)), T)

    return (delta >= 0) * delta


def MF_equation_lambda(lambda_SC, T, K_T):
    """
    Defining the MF equation defining lambda_SC
    """

    k_T = k(lambda_SC, T, K_T)

    constant_part = np.pi * J * rho * lambda_SC / 2
    difficult_part = MF_delta(T, k_T) * (1 - 2 * k_T) / (k_T * (1 - k_T))

    return constant_part - difficult_part


def MF_lambda_SC(T, K_T):
    """
    Solving for the mean-field value of λ_SC at particular temperature
    i.e. Finds the root of MF_equation_lambda()
    """

    MF_lambda_SC = optimize.brentq(MF_equation_lambda, 0, 20, args=(T, K_T))

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

    K_0 = 0.5 - 0.5 * np.sqrt(-0.5 * rho * J * np.log(rho * J))

    return K_0


def K_exp(T):
    """
    Returns an SC parameter decaying exponentially
    """

    Tc = np.exp(- special.digamma(0.5)) / (2 * np.pi)
    K_0 = 0.5 - 0.5 * np.sqrt(-0.5 * rho * J * np.log(rho * J))

    K_T = K_0 * np.exp(- T / Tc)

    return K_T


def K_grow(T):
    """
    Returns an SC
    """

    Tc = np.exp(- special.digamma(0.5)) / (2 * np.pi)
    alpha = 2 * Tc / np.log(-0.5 * rho * J * np.log(rho * J))

    K_T = 0.5 * (1 - np.exp((Tc - T) / alpha))

    return K_T


def k(lambda_SC, T, K_T):
    """
    Returns the value of the temperature dependent κ
    """

    return K_T / (1 + np.exp(- K_T * lambda_SC / T))


def plot_lambda_vs_T():
    """
    Plotting the value of lambda against temperature
    Mainly for debugging the parametric plot
    """

    Ts = np.linspace(0.01, 1.2, 250)

    lambdas = np.zeros(np.size(Ts))
    ss = np.zeros(np.size(Ts))

    for i in range(np.size(Ts)):
        T = Ts[i]

        lambdas[i] = MF_lambda_SC(T, K(T))
        ss[i] = lambdas[i] / T

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(Ts, ss, "k-")

    plt.show()


def plot_delta_vs_T():
    """
    Plots the *new* behaviour of the order parameter with temperature
    Includes the new temperature dependence of kappa
    """

    # Measure T in units of T_K
    Ts = np.linspace(0.01, 1.2, 250)

    lambdas = np.zeros(np.size(Ts))
    ks = np.zeros(np.size(Ts))

    deltas = np.zeros(np.size(Ts))
    deltas_up = np.zeros(np.size(Ts))
    deltas_down = np.zeros(np.size(Ts))

    for i in range(np.size(Ts)):

        T = Ts[i]

        lambda_SC = MF_lambda_SC(T, K(T))
        deltas[i] = MF_delta(T, k(lambda_SC, T, K(T)))

        lambda_SC_up = MF_lambda_SC(T, K_exp(T))
        deltas_up[i] = MF_delta(T, k(lambda_SC_up, T, K_exp(T)))

        lambda_SC_down = MF_lambda_SC(T, 0.5)
        deltas_down[i] = MF_delta(T, k(lambda_SC_up, T, 0.5))

        lambdas[i] = lambda_SC
        ks[i] = k(lambda_SC, T, K(T))

    z2s = 4 * np.multiply(ks, (1 - ks))

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.fill_between(np.linspace(0, np.max(Ts), 10),
                     0, -0.5, color='#dddddd')

    plt.fill_between(Ts, deltas_up, deltas_down,
                     facecolor='red', alpha=0.5)

    plt.plot(Ts, deltas, "k-")


    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)

    ax = plt.gca()
    ax.set_xlim([0, np.max(Ts)])
    ax.set_ylim([-0.1, 1.1 * np.max(deltas)])
    ax.tick_params(axis='both', labelsize=20)

    plt.axhline(y=0, linestyle='--', color='k')

    plt.savefig("new_delta_vs_T.pdf", dpi=300,
                format='pdf', bbox_inches='tight')
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

    #plot_graphical_solution()
    #plot_delta_vs_T()
    #plot_F_vs_T()
    plot_lambda_vs_T()

if __name__ == '__main__':
    main()
