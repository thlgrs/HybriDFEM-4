import numpy as np
from materials.base import Material

class Mixed_Hardening_Mat(Material):
    def __init__(self, E, nu, fy, H, r=0.0, corr_fact=1, shear_def=True):
        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_p'] = 0.0
        self.state['alpha'] = 0.0  # kinematic
        self.state['kappa'] = 0.0  # isotropic
        self.H = H  # total hardening modulus
        self.r = r  # isotropic/kinematic ratio (0 = kin., 1 = iso.)
        self.tag = 'MIXED_HARD'

        self.commit()

    # def

    def update(self, dL):
        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]

        E = self.stiff0['E']
        H = self.H
        r = self.r

        # conv

        # Trial stress
        sig_tr = self.stress['s'] + E * dL[0]
        alpha = self.state['alpha']
        kappa = self.state['kappa']

        sig_red_tr = sig_tr - alpha
        phi = abs(sig_red_tr) - (self.stress['f_y'] + kappa)

        phi_tol = 1e-8
        if phi <= phi_tol:
            # Elastic step
            self.stress['s'] = sig_tr
            self.stiff['E'] = E
            # print("phi:", phi, "stiffness:", self.stiff['E'])  # <-- here
            self.state['alpha'] = alpha
            self.state['kappa'] = kappa

        else:
            # Plastic step
            h_eff = E + H
            miu = phi / h_eff
            sign_red = np.sign(sig_red_tr)

            # === Debug prints ===
            # print(f"phi: {phi:.4e}, miu: {miu:.4e}, sig_tr: {sig_tr:.4e}, sig_red_tr: {sig_red_tr:.4e}, alpha: {alpha:.4e}, kappa: {kappa:.4e}, stiff[E]: {self.stiff['E']:.4e}")

            self.stress['s'] = sig_tr - miu * E * sign_red
            self.state['alpha'] = alpha + (1 - r) * H * miu * sign_red
            self.state['kappa'] = kappa + r * H * miu
            self.strain['e_p'] += miu * sign_red
            self.stiff['E'] = (E * H) / h_eff

        # Always elastic in shear
        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = self.stiff0['G']