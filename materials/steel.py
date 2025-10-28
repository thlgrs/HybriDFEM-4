import sys
import numpy as np
from materials.base import Material

class steel_EC(Material):

    def __init__(self, E, fyd, eps_ud=0.04, fail_crit=False, alpha=0.00357):

        nu = 0.0

        super().__init__(E, nu)

        self.stress['f_yd'] = fyd
        self.strain['e_ud'] = eps_ud
        self.strain['e_y'] = self.stress['f_yd'] / self.stiff0['E']
        self.fail_crit = fail_crit
        self.alpha = alpha
        # self.k = k

        self.nameTag = "Eurocode ULS steel model"

        self.commit()

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material

        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-self.strain['e_ud'], self.strain['e_ud'], 1000)
        gamma = np.linspace(-2.5 * self.strain['e_y'], 2.5 * self.strain['e_y'], 1000)
        sigma = np.zeros(1000)
        tau = np.zeros(1000)

        for i in range(1000):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']

        plt.figure(figsize=(5, 5), dpi=200)
        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')
        plt.plot(gamma * 100, tau / 1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        if self.fail_crit:
            plt.scatter(-4, -self.stress['f_yd'] / 1e6, color='red', marker='*', s=100, label='Failure')
            plt.scatter(4, self.stress['f_yd'] / 1e6, color='red', marker='*', s=100)

        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        """
        Updates the values of current strains to eps and gamma in the
        self.strains dictionary
        Updates the values of current stresses corresponding to eps and gamma
        in the self.stresses dictionary, according to a bilinear asymmetric
        σ − ε and linear τ − γ relationship

        """
        super().update(dL)

        if abs(self.strain['e']) < self.strain['e_y']:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = self.stiff0['E']

        else:

            eps_p = abs(self.strain['e']) - self.strain['e_y']
            f_p = self.alpha * self.stiff0['E'] * eps_p
            self.stress['s'] = (self.stress['f_yd'] + f_p) * np.sign(self.strain['e'])
            self.stiff['E'] = self.stiff0['E'] * self.alpha

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']

        if self.fail_crit:
            if abs(self.strain['e']) > self.strain['e_ud']:
                eps = self.strain['e']
                sys.exit(f'Steel has failed in tension: eps_s = {eps}')

class steel_tensionchord(Material):

    def __init__(self, fy, fu, ey, eu, fail_crit=False):

        nu = 0.0
        super().__init__(fy / ey, nu)

        self.stiff0['E'] = fy / ey
        self.stiff['E'] = fy / ey
        self.stiff['E_sh'] = (fu - fy) / (eu - ey)

        self.stress['f_y'] = fy
        self.stress['f_u'] = fu
        self.strain['e_u'] = eu
        self.strain['e_y'] = ey

        self.fail_crit = fail_crit
        self.tag = 'STC'

        self.nameTag = "Tension Chord steel model"

        self.commit()

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material

        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-self.strain['e_u'], self.strain['e_u'], 1000)
        gamma = np.linspace(-2.5 * self.strain['e_y'], 2.5 * self.strain['e_y'], 1000)
        sigma = np.zeros(1000)
        tau = np.zeros(1000)

        for i in range(1000):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']

        plt.figure(figsize=(5, 5), dpi=200)
        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')
        plt.plot(gamma * 100, tau / 1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        if self.fail_crit:
            # print(self.stress['f_u'])
            plt.scatter(-self.strain['e_u'] * 100, -self.stress['f_u'] / 1e6, color='red', marker='*', s=100,
                        label='Failure')
            plt.scatter(self.strain['e_u'] * 100, self.stress['f_u'] / 1e6, color='red', marker='*', s=100)

        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        """
        Updates the values of current strains to eps and gamma in the
        self.strains dictionary
        Updates the values of current stresses corresponding to eps and gamma
        in the self.stresses dictionary, according to a bilinear asymmetric
        σ − ε and linear τ − γ relationship

        """
        super().update(dL)

        if abs(self.strain['e']) < self.strain['e_y']:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = self.stiff0['E']
            self.yielded = False

        else:
            eps_p = abs(self.strain['e']) - self.strain['e_y']
            f_p = self.stiff['E_sh'] * eps_p
            self.stress['s'] = (self.stress['f_y'] + f_p) * np.sign(self.strain['e'])
            self.stiff['E'] = self.stiff['E_sh']
            self.yielded = True

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']
