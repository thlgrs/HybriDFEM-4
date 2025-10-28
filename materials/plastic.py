import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from materials.base import Material

class Plastic_Mat(Material):

    def __init__(self, E, nu, fy, corr_fact=1, shear_def=True):

        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_p'] = 0.
        self.tag = 'PLAST'
        self.commit()

    def plot_stress_strain(self):

        eps = np.linspace(0, 2 * self.strain['e_y'], 100)
        eps = np.append(eps, np.linspace(2 * self.strain['e_y'], -2 * self.strain['e_y'], 200))
        eps = np.append(eps, np.linspace(-2 * self.strain['e_y'], 3 * self.strain['e_y'], 300))
        eps = np.append(eps, np.linspace(3 * self.strain['e_y'], -3 * self.strain['e_y'], 400))
        gamma = np.zeros(len(eps))

        sig = np.zeros(len(eps))
        e_p = np.zeros(len(eps))

        for i in range(len(eps)):
            self.set_elongs(eps[i], gamma[i])
            self.update()
            sig[i] = self.stress['s']
            e_p[i] = self.strain['e_p']

        plt.figure(None, figsize=(6, 6))
        plt.plot(eps, sig, label='eps')
        plt.plot(e_p, sig, label='e_p')
        plt.grid(True)
        plt.legend()

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]

        s_tr = self.stiff0['E'] * (self.strain['e'] - self.strain['e_p'])
        f_tr = abs(s_tr) - (self.stress['f_y'])

        # Elastic step
        if f_tr <= 0:
            self.stress['s'] = s_tr
            self.stiff['E'] = deepcopy(self.stiff0['E'])

        # Plastic step
        else:
            d_g = f_tr / (self.stiff0['E'])

            self.stress['s'] = (1 - d_g * self.stiff0['E'] / abs(s_tr)) * s_tr
            # self.strain['e_p'] += d_g * np.sign(s_tr)

            self.stiff['E'] = 0.

        # Shear behaviour is linear elastic
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']

class Plastic_Stiffness_Deg(Material):

    def __init__(self, E, nu, fy, corr_fact=1, shear_def=True):

        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.stiff['E_el'] = E
        self.strain['e_y'] = fy / E
        self.tag = 'STIFFDEG'
        self.commit()

    def plot_stress_strain(self):

        eps = np.linspace(0, 2 * self.strain['e_y'], 100)
        eps = np.append(eps, np.linspace(2 * self.strain['e_y'], 0, 200))
        eps = np.append(eps, np.linspace(0, 4 * self.strain['e_y'], 300))
        eps = np.append(eps, np.linspace(4 * self.strain['e_y'], 0, 400))
        gamma = np.zeros(len(eps))

        sig = np.zeros(len(eps))

        for i in range(len(eps)):
            self.set_elongs(eps[i], gamma[i])
            self.update()
            sig[i] = self.stress['s']

        plt.figure(None, figsize=(6, 6))
        plt.plot(eps * 100, sig / 1e6, label='eps')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Strain [%]')
        plt.ylabel('Stress [MPa]')

        self.revert_commit()

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]
        # Elastic step
        if abs(self.strain['e']) <= self.strain['e_y']:
            self.stress['s'] = self.stiff['E_el'] * self.strain['e']
            self.stiff['E'] = self.stiff['E_el']

        # Plastic step
        else:
            # print('Plastic')
            self.stress['s'] = np.sign(self.strain['e']) * self.stress['f_y']
            self.stiff['E_el'] = self.stress['f_y'] / abs(self.strain['e'])
            self.strain['e_y'] = abs(self.strain['e'])
            self.stiff['E'] = 0.
        # Shear behaviour is linear elastic
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']
