"""
Numerically investigating the mean-field solution with constant K
Want to see if Delta = 0 is always a minimal solution
"""

import numpy as np
import scipy.special as special

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
from matplotlib.widgets import Slider

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
    fig.canvas.set_window_title('F vs Delta and Lambda')
    ax = fig.gca(projection='3d')

    T = 0.6

    Deltas = np.linspace(0.001, 2, 40)
    lambdas = np.linspace(0.01, 12, 60)

    Deltas_mesh, lambdas_mesh = np.meshgrid(Deltas, lambdas)
    Fs = F_star(Deltas_mesh, lambdas_mesh, T)

    surf = ax.plot_surface(Deltas_mesh, lambdas_mesh, Fs, cmap=cmx.Spectral,
                           linewidth=0, antialiased=False)

    MF_Deltas = MF_Delta(lambdas, T)
    F_line = F_star(MF_Deltas, lambdas, T)

    plt.plot(MF_Deltas, lambdas, F_line, "k-")

    ax.set_xlabel(r'$ \Delta $', fontsize=16)
    ax.set_ylabel(r'$ \lambda_{SC} $', fontsize=16)
    ax.set_zlabel(r'$ F(\Delta, \lambda_{SC}) $', fontsize=16)
    plt.title("T = " + f'{T:.2f}', fontsize=24)

    plt.show()

def plot_F_vs_Delta():
    """
    Making a 2D surface plot of F against the free parameters, Delta and lambda
    """

    fig = plt.figure()
    fig.canvas.set_window_title('F vs Delta')

    plot_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
    T_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

    T0 = 0.37

    Deltas = np.linspace(0.001, 2, 40)
    lambdas = np.linspace(0.01, 9, 40)

    MF_Deltas = MF_Delta(lambdas, T0)
    F_line = F_star(MF_Deltas, lambdas, T0)

    plt.axes(plot_ax)
    l, = plt.plot(MF_Deltas, F_line, "k-")
    plot_ax.set_xlim([0, 1.3])

    plt.xlabel(r'$ \Delta $', fontsize=20)
    plt.ylabel(r'$ F(\Delta) $', fontsize=20)
    plt.title("T = " + f'{T0:.2f}', fontsize=24)

    T_slider = Slider(T_ax, 'T', 0.01, 1.0, valinit=T0, color='grey')

    def update(val):
        T = T_slider.val
        MF_Deltas = MF_Delta(lambdas, T)
        F_line = F_star(MF_Deltas, lambdas, T)
        l.set_ydata(F_line)
        l.set_xdata(MF_Deltas)

        plot_ax.set_ylim([np.min(F_line) - 0.01, np.max(F_line) + 0.01])

        plt.title("T = " + f'{T:.2f}', fontsize=24)
        fig.canvas.draw_idle()

    T_slider.on_changed(update)

    plt.show()


def main():
    global rho, J, D
    rho = 0.4
    J = 0.4
    D = np.exp(1 / (rho * J)) / np.sqrt(rho * J)

    plot_F_vs_Delta_and_lambda()


if __name__ == '__main__':
    main()
