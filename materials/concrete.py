import sys
import numpy as np
from materials.base import Material

class concrete_EC(Material):

    def __init__(self, Ec, fcd, fct=0., f_res=0., eps_c=0.002, eps_cu=0.0035, fail_crit=False):

        nu = 0.0

        super().__init__(Ec, nu)

        self.stress['f_cd'] = fcd
        self.stress['f_ct'] = fct
        self.stress['f_res'] = f_res
        self.strain['e_c'] = eps_c
        self.strain['e_t'] = self.stress['f_ct'] / self.stiff0['E']
        self.strain['e_cu'] = eps_cu
        self.fail_crit = fail_crit

        self.nameTag = "Eurocode ULS concrete model"

        self.commit()

    def to_ommit(self):
        if self.strain['e'] > self.strain['e_t']:
            return True
        return False

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material

        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-self.strain['e_cu'], 2 * self.strain['e_t'], 500)
        gamma = np.linspace(-self.strain['e_cu'], 2 * self.strain['e_t'], 500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

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
        if self.fail_crit:
            plt.scatter(-self.strain['e_cu'] * 100, -self.stress['f_cd'] / 1e6, color='red', marker='*', s=100,
                        label='Failure')
        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
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

        if self.strain['e'] > 1e-10:

            if self.strain['e'] > 11 * self.strain['e_t'] / 10:

                self.stress['s'] = self.stress['f_res']
                self.stress['t'] = 0
                self.stiff['E'] = 0
                self.stiff['G'] = 0

            elif self.strain['e'] > self.strain['e_t']:

                self.stress['t'] = self.stiff0['G'] * self.strain['g']
                self.stiff['E'] = 0 * (self.stress['f_res'] - self.stress['f_ct']) / self.strain['e_t']
                self.stress['s'] = self.stress['f_ct'] + self.stiff['E'] * (self.strain['e'] - self.strain['e_t'])
                self.stiff['G'] = self.stiff0['G']

            else:
                self.stress['s'] = self.stiff0['E'] * self.strain['e']
                self.stress['t'] = self.stiff0['G'] * self.strain['g']
                self.stiff['E'] = self.stiff0['E']
                self.stiff['G'] = self.stiff0['G']

        else:
            if self.strain['e'] >= - self.strain['e_c']:
                self.stress['s'] = (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] ** 2 \
                                   + 2 * (self.stress['f_cd'] / self.strain['e_c']) * self.strain['e']
                self.stiff['E'] = 2 * (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] \
                                  + 2 * (self.stress['f_cd'] / self.strain['e_c'])
            else:
                self.stress['s'] = - self.stress['f_cd']
                self.stiff['E'] = 0

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']

        if self.fail_crit:
            if self.strain['e'] < -self.strain['e_cu']:
                eps = self.strain['e']
                sys.exit(f'Concrete has failed in compression: eps_c = {eps}')

class concrete_tensionchord(Material):

    def __init__(self, fc, Ec):

        super().__init__(Ec, 0.0)

        self.cracked = False
        self.randomized = False
        # import random
        # self.stress['fct'] =  random.gauss(1, 0.1) * fc
        # print(self.stress['fct'])
        self.stress['fct'] = fc
        self.strain['e_ct'] = self.stress['fct'] / Ec
        self.tag = 'CTC'

        self.commit()

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material

        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-4 * self.strain['e_ct'], 2 * self.strain['e_ct'], 1000)
        sigma = np.zeros(1000)

        for i in range(1000):

            if i == 0:
                d_eps = eps[0]
            else:
                d_eps = eps[i] - eps[i - 1]

            self.update([d_eps, 0])
            sigma[i] = self.stress['s']

        plt.figure(figsize=(5, 5), dpi=200)
        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        plt.xlabel(r'Strain $\varepsilon$ [\%]')
        plt.ylabel(r'Stress $\sigma$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        super().update(dL)

        if not self.randomized:
            import random
            self.stress['fct'] = random.gauss(1, 0.02) * self.stress['fct']
            self.strain['e_ct'] = self.stress['fct'] / self.stiff0['E']
            self.randomized = True

        if self.strain['e'] <= self.strain['e_ct']:
            self.stress['s'] = self.strain['e'] * self.stiff0['E']
            self.stiff['E'] = self.stiff0['E']
        else:
            self.stress['s'] = 0.
            self.stiff['E'] = 0.
            self.cracked = True

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']

class popovics_concrete(Material):

    def __init__(self, fc, ec, ecu, n, ft, et, etu, b, gc=None):

        nu = 0.0
        E_0 = fc * n / (ec * (n - 1))

        super().__init__(E_0, nu)

        self.stress['fc'] = - fc
        self.stress['ft'] = ft

        self.n = n
        self.b = b

        # if gc is not None:
        #     if fc*ec*np.pi / gc < 0 or fc*ec*np.pi / gc > 1:
        #         print('Dissipation energy is not large enough')

        #     self.n = np.pi / np.arcsin(fc*ec*np.pi / gc)

        self.strain['ec'] = - ec
        self.strain['et'] = et
        self.strain['ecu'] = - ecu
        self.strain['etu'] = etu

        self.tag = 'PPC'
        self.nameTag = 'Popovics concrete model'

        self.commit()

    def plot_stress_strain(self):

        import matplotlib.pyplot as plt

        eps = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        gamma = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        E = np.zeros(500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']
            E[i] = self.stiff['E']

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        # plt.figure(figsize=(5,5), dpi=200)

        # plt.plot(eps*100, E/1e9, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        # plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        # plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        # plt.title('Stress-strain relation \n' + self.nameTag)
        # plt.axhline(y=0, color='k', linewidth=.7)
        # plt.axvline(x=0, color='k', linewidth=.7)
        # plt.legend()
        # plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        super().update(dL)

        # Compression
        if self.strain['e'] <= 0:

            if self.strain['e'] < self.strain['ecu']:
                self.stress['s'] = 0.
                self.stiff['E'] = 0.
            else:
                r_e = self.strain['e'] / self.strain['ec']
                self.stress['s'] = (self.stress['fc'] * r_e) * (self.n / (self.n - 1 + (r_e) ** self.n))
                self.stiff['E'] = - self.stress['fc'] * self.n ** 2 * r_e ** self.n / (
                        self.strain['ec'] * (self.n + r_e ** self.n - 1) ** 2) \
                                  + self.stress['fc'] * self.n / (self.strain['ec'] * (self.n + r_e ** self.n - 1))
        else:

            if self.strain['e'] > self.strain['etu']:
                self.stress['s'] = 0.
                self.stiff['E'] = 0.
            elif self.strain['e'] > self.strain['et']:
                self.stress['s'] = self.stress['ft'] * self.b ** (
                        (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))
                self.stiff['E'] = (self.stress['ft'] * np.log(self.b) * (self.b) ** (
                        (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))) \
                                  / (self.strain['etu'] - self.strain['et'])
            else:
                self.stiff['E'] = self.stress['ft'] / self.strain['et']
                self.stress['s'] = self.stiff['E'] * self.strain['e']

        self.stress['t'] = self.stiff['G'] * self.strain['g']

class KSP_concrete(Material):

    def __init__(self, fc, ec, fu, ecu, ft, et, etu, b, gc=None, gt=None):

        nu = 0.0
        E_0 = 2 * fc / ec

        super().__init__(E_0, nu)

        self.stress['fc'] = - fc
        self.stress['fu'] = - fu
        self.stress['ft'] = ft

        self.b = b

        if gc is not None:
            R = fu / fc
            ecu = (1 / (1 + R)) * (2 * gc / fc - fc / E_0 + (1 + R) * ec + R ** 2 * fc / E_0)

        if gt is not None:
            etu = (np.log(b) / (ft * (b - 1))) * (gt) + et

        #     self.n = np.pi / np.arcsin(fc*ec*np.pi / gc)

        self.strain['ec'] = - ec
        self.strain['et'] = et
        self.strain['ecu'] = - ecu
        self.strain['etu'] = etu

        self.tag = 'KSP'
        self.nameTag = 'Kent-Scott-Park concrete model'

        self.commit()

    def plot_stress_strain(self):

        import matplotlib.pyplot as plt

        eps = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        gamma = np.linspace(1.2 * self.strain['ecu'], 1.2 * self.strain['etu'], 500)
        E = np.zeros(500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

            if i == 0:
                d_eps = eps[0]
                d_gam = gamma[0]
            else:
                d_eps = eps[i] - eps[i - 1]
                d_gam = gamma[i] - gamma[i - 1]

            self.update([d_eps, d_gam])
            sigma[i] = self.stress['s']
            tau[i] = self.stress['t']
            E[i] = self.stiff['E']

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, sigma / 1e6, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(5, 5), dpi=200)

        plt.plot(eps * 100, E / 1e9, color='black', linewidth=0.9, label=r'$\sigma(\varepsilon)$')

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
        plt.axhline(y=0, color='k', linewidth=.7)
        plt.axvline(x=0, color='k', linewidth=.7)
        plt.legend()
        plt.grid(True)

        self.revert_commit()

    def update(self, dL):

        super().update(dL)

        # Compression
        if self.strain['e'] <= 0:

            if self.strain['e'] < self.strain['ecu']:
                self.stress['s'] = self.stress['fu']
                self.stiff['E'] = 0.
            elif self.strain['e'] < self.strain['ec']:
                self.stiff['E'] = (self.stress['fu'] - self.stress['fc']) / (self.strain['ecu'] - self.strain['ec'])
                self.stress['s'] = self.stiff['E'] * (self.strain['e'] - self.strain['ec']) + self.stress['fc']
            else:
                a = - self.stress['fc'] / self.strain['ec'] ** 2
                b = - 2 * a * self.strain['ec']
                self.stress['s'] = a * self.strain['e'] ** 2 + b * self.strain['e']
                self.stiff['E'] = 2 * a * self.strain['e'] + b

        else:

            if self.strain['e'] > self.strain['etu']:
                self.stress['s'] = 0.
                self.stiff['E'] = 0.
            elif self.strain['e'] > self.strain['et']:
                self.stress['s'] = self.stress['ft'] * self.b ** (
                        (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))
                self.stiff['E'] = (self.stress['ft'] * np.log(self.b) * (self.b) ** (
                        (self.strain['e'] - self.strain['et']) / (self.strain['etu'] - self.strain['et']))) \
                                  / (self.strain['etu'] - self.strain['et'])
            else:
                self.stiff['E'] = self.stress['ft'] / self.strain['et']
                self.stress['s'] = self.stiff['E'] * self.strain['e']

        self.stress['t'] = self.stiff['G'] * self.strain['g']

class concrete_EC_softening(Material):

    def __init__(self, Ec, fcd, alpha=-0.1, eps_c=0.002, fail_crit=False, gt=None):

        nu = 0.0

        super().__init__(Ec, nu)

        self.stress['f_cd'] = fcd
        self.strain['e_c'] = eps_c
        self.alpha = alpha
        self.fail_crit = fail_crit

        if gt is not None:  # Use fracture energy to regularize ultimate strain
            e_t = fcd / Ec
            e_f = e_t + 2 * gt / fcd
            self.alpha = - fcd / (Ec * (e_f - e_t))

        self.nameTag = "Eurocode ULS concrete model with softening"

        self.commit()

    def to_ommit(self):

        return False

    def plot_stress_strain(self):

        """
        Displays the σ − ε and τ − γ curve of the material

        """
        import matplotlib.pyplot as plt

        eps = np.linspace(-10 * self.strain['e_c'], 0, 500)
        gamma = np.linspace(-10 * self.strain['e_c'], 0, 500)
        sigma = np.zeros(500)
        tau = np.zeros(500)

        for i in range(500):

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

        # plt.plot(gamma*100, tau/1e6, color='black', linewidth=0.7, label=r'$\tau(\gamma)$', linestyle='dashed')
        plt.xlabel(r'Strain $\varepsilon$ or $\gamma$ [\%]')
        plt.ylabel(r'Stress $\sigma$ or $\tau$ [MPa]')
        plt.title('Stress-strain relation \n' + self.nameTag)
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

        if self.strain['e'] > self.tol_disp:

            self.stress['s'] = 0.
            self.stress['t'] = 0.
            self.stiff['E'] = 0.
            self.stiff['G'] = 0.

        else:
            if self.strain['e'] >= - self.strain['e_c']:
                self.stress['s'] = (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] ** 2 \
                                   + 2 * (self.stress['f_cd'] / self.strain['e_c']) * self.strain['e']
                self.stiff['E'] = 2 * (self.stress['f_cd'] / self.strain['e_c'] ** 2) * self.strain['e'] \
                                  + 2 * (self.stress['f_cd'] / self.strain['e_c'])
            else:
                self.stress['s'] = - self.stress['f_cd'] - (abs(self.strain['e']) - self.strain['e_c']) * (
                        self.alpha * self.stiff0['E'])
                self.stiff['E'] = self.alpha * self.stiff0['E']

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']
