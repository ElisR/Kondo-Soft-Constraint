import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special


def Tc(b):
    """
    Defining the parametric T in terms of difficult expressions
    """

    K = 0.5 - 0.5 * np.sqrt(-0.5 * rho * J * np.log(rho * J))

    z2 = K * (2 - K)

    t = (1 / (2 * np.pi)) * (1 / np.sqrt(rho * J)) *\
        np.exp((1 - 1 / z2) / (rho * J)) *\
        np.abs(np.exp(- 4j * np.pi * special.loggamma(0.5 + b / (4j * np.pi)) / b))

    return t


def plot_new_phase_boundary():

    # Prepare the array for parametric plot
    b = np.linspace(0.01, 200, 1000)

    x = b * Tc(b)
    y = Tc(b)

    #right = np.linspace(max(x), 5.8, 10)

    fig = plt.figure(figsize=(8.4, 8.4))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot the figure
    plt.plot(x, y, "k-")

    # Add phase regions
    #plt.fill_between(x, y, color='#d9ffb3')
    #maxy = plt.ylim()[1]
    #plt.fill_between(x, y, 1.2, color='#8cccff')
    #plt.fill_between(right, 1.2, color='#8cccff')

    # Label the phases
    #plt.text(2.2, 0.55, r'$ | \Delta | > 0 $', fontsize=26)
    #plt.text(4.1, 0.97, r'$ \Delta = 0 $', fontsize=26)

    # Make improvements to the figure
    plt.ylabel(r'$ \frac{T}{T_K} $', fontsize=26)
    plt.xlabel(r'$ \frac{g \mu_B B}{T_K} $', fontsize=26)

    ax = plt.gca()
    #ax.set_xlim([0, 5.8])
    #ax.set_ylim([0, 1.2])
    ax.tick_params(axis='both', labelsize=20)

    plt.savefig("new_phase_diagram.pdf", dpi=300,
                format='pdf', bbox_inches='tight')
    plt.clf()

def main():

    # Defining the parameters of the system
    global rho, J
    rho = 0.4
    J = 0.4

    plot_new_phase_boundary()


if __name__ == "__main__":
    main()
