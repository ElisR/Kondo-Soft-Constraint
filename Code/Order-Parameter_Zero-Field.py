# Plotting Δ as a function of temperature

# Defining a function for the inverse of the digamma function
# Shouldn't be too difficult to do numerically since
# psi is a monotonic function

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


def MF_Δ(T):
    """
    Find the value of delta predicted by the mean-field equations
    """

    digamma_inverse = np.vectorize(digamma_inv)

    psi_tilde = np.log(1 / (2 * np.pi * T))
    - 0.5 * np.log(ρ * J) + (1 - 1 / z2(T)) / (ρ * J)

    argument_tilde = digamma_inverse(psi_tilde)
    Δ = np.multiply((argument_tilde - 0.5) * (2 * np.pi / z2(T)), T)

    return Δ


def plot_Δ_vs_T():
    """
    Plots the behaviour of the order parameter with temperature
    """

    # Measure T in units of T_K
    T = np.linspace(0.01, 1.2, 100)

    # Calculate the order parameter in units of T_K
    Δ = MF_Δ(T)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.fill_between(np.linspace(0, np.max(T), 10),
                     0, -0.5, color='#dddddd')

    # Plot the figure
    plt.plot(T, Δ, "k-")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \Delta / T_K $', fontsize=26)

    ax = plt.gca()
    ax.set_xlim([0, np.max(T)])
    ax.set_ylim([-0.18, 1.25])
    ax.tick_params(axis='both', labelsize=20)

    plt.axhline(y=0, linestyle='--', color='k')

    plt.savefig("delta_vs_T.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def F(T, Δ):
    """
    Return the value of the free energy for temperature, hybridisation field
    """

    f = 2 * Δ / (np.pi * J * ρ) - 4 * T * np.real(
        np.log(special.gamma(0.5 +
                             (1j * z2(T) * Δ +
                              np.exp(1 / (ρ * J)) / np.sqrt(ρ * J)) /
                             (2j * np.pi * T)) /
               special.gamma(0.5 + (z2(T) * Δ /
                                    (2 * np.pi * T)))))

    return f


def MF_F(T):
    """
    Return the mean-field free energy at a given temperature
    Sets Δ=0 if Δ<0
    """

    Δ = MF_Δ(T)
    Δ0 = (Δ >= 0) * Δ

    f = F(T, Δ0)

    return f


def z2(T, B=0):
    """
    Returns the value of z^2 at a particular temperature
    Trying to see if a temperature dependence can remove phase transition
    """

    z2_inverse = (1 - 0.5 * ρ * J * np.log(ρ * J))# - ρ * J * np.log(T) * np.tanh(4 * T)

    return 1 / z2_inverse


def plot_F_vs_Δ():
    """
    Plots the free energy as a function of Δ at various temperatures
    """

    # Measure T and Δ in units of T_K
    # Ts = np.array([0.4, 0.6, 0.95, 1.13, 1.3])
    Tc = np.exp(- special.digamma(0.5)) / (2 * np.pi)
    Ts = np.linspace(Tc - 0.1, Tc + 0.1, 11)
    Δs = np.linspace(-0.2, 1.2, 160)

    Fs = np.zeros((np.size(Ts), np.size(Δs)))

    MF_Δs = MF_Δ(Ts)
    # MF_Δs = (MF_Δs >= 0) * MF_Δs
    MF_Fs = F(Ts, MF_Δs)

    for i in range(np.size(Ts)):
        T = Ts[i]
        Fs[i, :] = F(T, Δs)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.fill_between(np.linspace(np.min(Δs), 0, 10),
                     1.2 * np.max(Fs), 0.8 * np.min(Fs), color='#dddddd')
    plt.axvline(x=0, linestyle='--', color='k')

    cm = plt.get_cmap('inferno')
    cNorm = colors.Normalize(vmin=np.min(Ts), vmax=np.max(Ts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for i in range(np.size(Ts)):
        colorVal = scalarMap.to_rgba(Ts[i])
        plt.plot(Δs, Fs[i, :], "-", color=colorVal, label=str(Ts[i]))

    plt.plot(MF_Δs, MF_Fs, "r.", label="Mean-Field")

    plt.xlabel(r'$ \Delta / T_K $', fontsize=26)
    plt.ylabel(r'$ F(\frac{\Delta}{T_K}) / T_K $', fontsize=26)

    plt.text(0.5 * np.max(Δs), 1.0 * np.max(Fs),
             r'$ \rho = ' + str(ρ) + r'\quad J = ' + str(J) + r'$',
             fontsize=26)

    ax = plt.gca()
    ax.set_xlim([np.min(Δs), np.max(Δs)])
    ax.set_ylim([np.min(Fs) - 0.1 * (np.max(Fs) - np.min(Fs)),
                 np.max(Fs) + 0.1 * (np.max(Fs) - np.min(Fs))])
    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("F_vs_delta.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def plot_d2F_vs_T():
    """
    Plotting the second derivative of free energy wrt temperature
    """

    Tc = np.exp(- special.digamma(0.5)) / (2 * np.pi)
    Ts = np.linspace(Tc - 0.1, Tc + 0.1, 101)

    MF_d2Fs = derivative(MF_F, Ts, n=2, dx=0.001, order=3)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(Ts, MF_d2Fs, "r-", label="Mean-Field C")

    plt.xlabel(r'$ T / T_K $', fontsize=26)
    plt.ylabel(r'$ \frac{d^2(F / T_K)}{d(T / T_K)^2} $', fontsize=26)

    ax = plt.gca()
    ax.set_xlim([np.min(Ts), np.max(Ts)])

    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("d2F_vs_T.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.clf()


def main():
    # Setting various parameters of the problem

    global ρ, J
    ρ = 0.4
    J = 0.4
    #z2 = 1 / (1 - 0.5 * ρ * J * np.log(ρ * J))

    plot_Δ_vs_T()
    plot_F_vs_Δ()
    plot_d2F_vs_T()


if __name__ == '__main__':
    main()
