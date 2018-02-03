# Plotting Δ as a function of temperature

# Defining a function for the inverse of the digamma function
# Shouldn't be too difficult to do numerically since
# psi is a monotonic function

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special


def digamma_inv(y):
    """
    Inverse digamma function
    Returns x given y: psi(x) = y
    TODO: make this part vectorized from the beginning
    """

    x = (np.exp(y) + 0.5) if (y >= -2.22) else (-1 / (y + special.digamma(1)))

    # Using Newton-Raphson for this monotonic function
    # May have to change to a convergence condition
    for i in range(1, 10):
        x = x - (special.digamma(x) - y) / special.polygamma(1, x)

    return x

def MF_Δ(T, ρ, J, z2):
    """
    Find the value of delta predicted by the mean-field equations
    """

    digamma_inverse = np.vectorize(digamma_inv)
    psi_tilde = np.log(1 / (2 * np.pi * T)) - 0.5 * np.log(ρ * J) + (1 - 1 / z2) / (ρ * J)
    argument_tilde = digamma_inverse(psi_tilde)
    Δ = np.multiply((argument_tilde - 0.5) * (2 * np.pi / z2), T)

    return Δ

def plot_Δ_vs_T(ρ, J, z2):
    """
    Plots the behaviour of the order parameter with temperature
    """

    # Measure T in units of T_K
    T = np.linspace(0.01, 1.2, 100)

    # Calculate the order parameter in units of T_K
    Δ = MF_Δ(T, ρ, J, z2)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot the figure
    plt.plot(T, Δ, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)

    ax = plt.gca()
    ax.set_xlim([0, np.max(T)])
    ax.tick_params(axis='both', labelsize=20)

    plt.axhline(y=0, linestyle='--', color='k')

    plt.savefig("delta_vs_T.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def F(T, Δ, ρ, J, z2):
    """
    Return the value of the free energy for temperature, hybridisation field
    """

    f = 2 * Δ / (np.pi * J * ρ) - 4 * T * np.real(
        np.log(special.gamma(0.5 +
                             (1j * z2 * Δ +
                              np.exp(1 / (ρ * J)) / (ρ * J)) /
                             (2j * np.pi * T)) /
               special.gamma(0.5 + (z2 * Δ /
                                    (2 * np.pi * T)))))

    return f


def plot_F_vs_Δ(ρ, J, z2):
    """
    Plots the free energy as a function of Δ at various temperatures
    """

    # Measure T and Δ in units of T_K
    Ts = np.array([0.1, 0.6, 0.95, 1.13, 1.5])
    Δs = np.linspace(0, 1.0, 100)

    Fs = np.zeros((np.size(Ts), np.size(Δs)))

    MF_Δs = MF_Δ(Ts, ρ, J, z2)

    for i in range(np.size(Ts)):
        T = Ts[i]
        Fs[i, :] = F(T, Δs, ρ, J, z2)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(np.size(Ts)):
        plt.plot(Δs, Fs[i, :], "k-", label=str(Ts[i]))

    plt.xlabel(r'$ \Delta / T_K $', fontsize=26)
    plt.ylabel(r'$ F(\frac{\Delta}{T_K}) / T_K $', fontsize=26)

    plt.text(0.6 * np.max(Δs), 1.0 * np.max(Fs),
             r'$ \rho = ' + str(ρ) + r'\quad J = ' + str(J) + r'$',
             fontsize=26)

    ax = plt.gca()
    ax.set_xlim([0, np.max(Δs)])
    ax.set_ylim([0.9 * np.min(Fs), 1.1 * np.max(Fs)])
    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("F_vs_delta", dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()

def main():
    # Setting various parameters of the problem

    ρ = 1
    J = 2
    z2 = 1 / (1 - 0.5 * ρ * J * np.log(ρ * J))

    plot_Δ_vs_T(ρ, J, z2)
    plot_F_vs_Δ(ρ, J, z2)


if __name__ == '__main__':
    main()
