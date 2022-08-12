# Quantum Tunneling Simulation for 1D Time-Depentend Schroedinger Equation
# Simulates quantum tunneling of an wave packet across a potential barrier
# Wave packet as a 1D Gaussian function
# Solved using the Finite-Difference Time-Domain (FD-TD) method

# Code inspired by Quantum Tunneling (https://github.com/nishantsule/Quantum-Tunneling) done by Nishant Sule (https://github.com/nishantsule).

# Import Python modules

import numpy as np                   # Mathematical functions for Python
import matplotlib.pyplot as plt      # Plot functions for Python
import scipy.constants as constants  # Physical constants

class QuantumTunnelingFDTD:
    def __init__(self, V0, bw, ke, me, sig):

        ## Convert Units
        self.V0 = V0 * constants.value('electron volt')
        self.bw = bw * constants.value('Angstrom star')

        self.ke = ke * constants.value('electron volt')
        self.me = me * constants.m_e

        # Numerical definition of the Wave Function
        ## Initial Wave Vector
        self.k0 = np.sqrt(2 * self.ke * self.me / (constants.hbar ** 2))

        # Initial velocity of the wave function
        self.vel = constants.hbar * self.k0 / self.me

        # Define simulation parameters for the Quantum Tunneling Simulation
        ## Initial spread of Gaussian Wave function
        self.sig = sig * constants.value('Angstrom star')

        ## Define grid cell size
        self.dx = np.minimum((self.bw / 25.0), (self.sig / 25.0))

        ## Define simulation domain
        length = 40 * np.maximum(self.bw, self.sig)  # Simulation domain length
        self.ll = int(length / self.dx)              # Total number of grid points in the domain
        self.lx = np.linspace(0.0, length, self.ll)  # Initial position of the wave vector along x

        ## Simulation time step size
        self.dt = 0.9 * constants.hbar / ((constants.hbar ** 2 / (self.me * self.dx ** 2)) + (self.V0 / 2.0))
        self.tt = int(0.35 * length / self.vel / self.dt) # Total number of time steps in the simulation

        # Numerical Definition of the Potential Energy Barrier
        ## Build potential energy array in the simulation domain
        self.Vx = np.zeros(self.ll)

        ## Build potential energy barrier array
        bwgrid = int(self.bw / (2.0 * self.dx))
        bposgrid = int(self.ll / 2.0)
        bl = bposgrid - bwgrid
        br = bposgrid + bwgrid

        ## Include the potential energy barrier in the potential energy array
        self.Vx[bl:br] = self.V0

        # Numerical Definition of the Wave Function
        ## Build wave function arrays for real and imaginary parts and its magnitude
        self.psi_real = np.zeros((self.ll))
        self.psi_im   = np.zeros((self.ll))
        self.psi_mag  = np.zeros(self.ll)

        ## Describe the wave function using a Gaussian function
        ### Define Gaussian parameters
        ac = 1.0 / np.sqrt((np.sqrt(np.pi)) * self.sig)
        x0 = bl * self.dx - 6 * self.sig

        self.psi_gauss = ac * np.exp(-(self.lx - x0) ** 2 / (2.0 * self.sig ** 2))

        ### Build the wave function arrays replacing by the built Gaussian function
        self.psi_real = self.psi_gauss * np.cos(self.k0 * self.lx)
        self.psi_im   = self.psi_gauss * np.sin(self.k0 * self.lx)
        self.psi_mag  = self.psi_real ** 2 + self.psi_im ** 2

        # Define Finite-Difference Time-Domain coefficients
        self.c1 = constants.hbar * self.dt / (2.0 * self.me * self.dx ** 2)
        self.c2 = self.dt / constants.hbar

    ## Function to update Finite-Difference Time-Domain constants
    def update_fdtd_coeffs(self):
        self.psi_im[1:self.ll - 1]  = ( self.c1 * (self.psi_real[2:self.ll] - 2.0 * self.psi_real[1:self.ll - 1] + \
                                        self.psi_real[0:self.ll - 2]) - self.c2 * self.Vx[1:self.ll - 1] * \
                                        self.psi_real[1:self.ll - 1] + self.psi_im[1:self.ll - 1])

        self.psi_real[1:self.ll - 1] = (-self.c1 * (self.psi_im[2:self.ll] - 2.0 * self.psi_im[1:self.ll - 1] + \
                                         self.psi_im[0:self.ll - 2]) + self.c2 * self.Vx[1:self.ll - 1] * \
                                         self.psi_im[1:self.ll - 1] + self.psi_real[1:self.ll - 1])

        self.psi_mag = self.psi_real ** 2 + self.psi_im ** 2

def run_quantum_tunneling_simulation(V0_in, bw_in, ke_in, me_in=1.0, sig_in=1.0):
    # Create Quantum Tunneling FDTD object
    qtunnel = QuantumTunnelingFDTD(V0_in, bw_in, ke_in, me_in, sig_in)

    # Function to Plot Initial Conditions
    def plot_initial_conditions():
        plt.ioff()
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        ax0.set_xlabel('position ($\AA$)')
        ax0.set_ylabel('$\Psi$')
        ax0.set_title('Initial wave functions (normalized)')
        ax0.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_mag / np.amax(qtunnel.psi_mag), label='$|\Psi|^2$')
        ax0.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.Vx / np.amax(qtunnel.Vx), label='barrier')
        ax0.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_im / np.amax(qtunnel.psi_im), label='$\Im[\Psi]$', alpha=0.5)
        ax0.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_real / np.amax(qtunnel.psi_real), label='$\Re[\Psi]$', alpha=0.5)
        ax0.legend()
        fig0.tight_layout()
        fig0.canvas.draw()

        return fig0

    # Function to print Quantum Tunneling Settings
    def print_tunneling_settings():
        print('')
        print('Potential barrier = ', round(qtunnel.V0 / constants.value('electron volt'), 2), 'eV')
        print('Potential barrier width = ', round(qtunnel.bw / constants.value('Angstrom star'), 2), 'A')
        print('(The boundary of the simulation domain is assumed to be an infinite barrier)')
        print('Wave packet energy = ', round(qtunnel.ke / constants.value('electron volt'), 2), 'eV')
        print('Wave packet spread = ', round(qtunnel.sig / constants.value('Angstrom star'), 2), 'A')
        print('')
        print('Grid size = ', '%.2e' % (qtunnel.dx / constants.value('Angstrom star')), 'A')
        print('Time step = ', "%.2e" % (qtunnel.dt * 1e15), 'fs')

    # Function to Calculate Wave Packet propagation by the Finite-Difference Time-Domain Method
    def run_fdtd_simulation():
        # Define Finite-Difference Time-Domain coefficients
        c1 = constants.hbar * qtunnel.dt / (2.0 * constants.m_e * qtunnel.dx ** 2)
        c2 = qtunnel.dt / constants.hbar

        ## Define Plot object for time steps
        plt.ion()
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('position ($\AA$)')
        ax1.set_ylabel('norm magnitude')
        fig1.show()
        fig1.canvas.draw()

        for nn in range(0, qtunnel.tt):
            # Update FDTD coefficients
            qtunnel.update_fdtd_coeffs()

            if nn % 50 == 0:
                # Update Plot
                tstr = 'Time = ' + str(round(nn * qtunnel.dt * 1e15, 4)) + ' fs'
                ax1.clear()
                ax1.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_mag / np.amax(qtunnel.psi_mag), label='$|\Psi|^2$')
                ax1.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.Vx / np.amax(qtunnel.Vx), label='barrier')
                ax1.legend()
                ax1.set_title(tstr)
                ax1.set_xlabel('position ($\AA$)')
                ax1.set_ylabel('normalized magnitude')
                fig1.canvas.draw()

    # Function to Plot Final Conditions
    def plot_final_conditions():
        # Plot final wave function
        plt.ioff()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('position ($\AA$)')
        ax2.set_ylabel('$\Psi$')
        ax2.set_title('Final wave functions (normalized)')
        ax2.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_mag / np.amax(qtunnel.psi_mag), label='$|\Psi|^2$')
        ax2.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.Vx / np.amax(qtunnel.Vx), label='barrier')
        ax2.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_im / np.amax(qtunnel.psi_im), label='$\Im[\Psi]$', alpha=0.5)
        ax2.plot(qtunnel.lx / constants.value('Angstrom star'), qtunnel.psi_real / np.amax(qtunnel.psi_real), label='$\Re[\Psi]$', alpha=0.5)
        ax2.legend()
        fig2.tight_layout()
        fig2.canvas.draw()

        return fig2

    print_tunneling_settings()
    fig_initial = plot_initial_conditions()
    run_fdtd_simulation()
    fig_final = plot_final_conditions()

    return fig_initial, fig_final
