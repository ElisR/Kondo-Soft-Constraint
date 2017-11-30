import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special


def Tc(b):
    # One of the parametric functions for defining the critical temperature
    # Involves the analytic continuation of the logarithm of gamma
    t = (1 / (2 * np.pi)) *\
        np.abs(np.exp(- (4 * np.pi * 1j *
                      special.loggamma((1 / 2) + (b / (4 * np.pi * 1j)))) /
                      b))

    return t


def plot_phase_boundary():
    # Plotting the phase boundary of the large N Kondo model
    # Arises from averaging free energy
    # Uses parametric plot

    # Prepare the array for parametric plot
    b = np.linspace(0.01, 200, 1000)
    x = b * Tc(b)
    y = Tc(b)

    # Fix the edge
    x = np.append(x, max(x))
    y = np.append(y, 0)
    right = np.linspace(max(x), 5.8, 10)

    fig = plt.figure(figsize=(8, 8))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot the figure
    plt.plot(x, y, "k-")

    # Add phase regions
    plt.fill_between(x, y, color='#d9ffb3')
    maxy = plt.ylim()[1]
    plt.fill_between(x, y, 1.2, color='#8cccff')
    plt.fill_between(right, 1.2, color='#8cccff')

    # Label the phases
    plt.text(2.2, 0.55, r'$ | \Delta | > 0 $', fontsize=22)
    plt.text(4.1, 0.97, r'$ \Delta = 0 $', fontsize=22)

    # Make improvements to the figure
    plt.xlabel(r'$ \frac{T}{T_K} $', fontsize=22)
    plt.ylabel(r'$ \frac{g \mu_B B}{T_K} $', fontsize=22)

    ax = plt.gca()
    ax.set_xlim([0, 5.8])
    ax.set_ylim([0, 1.2])

    plt.savefig("phase_diagram.pdf", dpi=300, format='pdf')
    plt.clf()


def main():
    plot_phase_boundary()


if __name__ == "__main__":
    main()
