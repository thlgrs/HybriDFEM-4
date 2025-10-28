from materials.base import Material

class PlaneStress(Material):
    """
    Isotropic linear elastic material in plane stress conditions.
    Inherits from Material and extends it for 2D continuum behavior.
    """

    def __init__(self, E, nu, rho, corr_fact=1, shear_def=True):
        # Initialize parent class (keeps 1D interface behavior intact)
        super().__init__(E, nu, corr_fact, shear_def)

        # Store additional FE-specific parameters
        self.nu = nu
        self.rho = rho
        self.tag = 'PLANE_STRESS'

        # Extend stress dict for 2D continuum (Voigt notation: σ_xx, σ_yy, τ_xy)
        self.stress['sigma_xx'] = 0.0
        self.stress['sigma_yy'] = 0.0
        self.stress['tau_xy'] = 0.0

        # Extend strain dict for 2D continuum (Voigt notation: ε_xx, ε_yy, γ_xy)
        self.strain['epsilon_xx'] = 0.0
        self.strain['epsilon_yy'] = 0.0
        self.strain['gamma_xy'] = 0.0

        # Commit initial state
        self.commit()

    @property
    def D(self) -> np.ndarray:
        """
        Constitutive matrix for plane stress (3x3).
        Relates stress to strain: {σ} = [D]{ε}
        """
        E, nu = self.stiff['E'], self.nu
        c = E / (1 - nu ** 2)
        return c * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

    def update_2D(self, strain_vector):
        """
        Update 2D stress state based on strain increment or total strain.

        Parameters
        ----------
        strain_vector : np.ndarray
            Strain vector in Voigt notation [ε_xx, ε_yy, γ_xy]
        """
        # Compute stress from strain using constitutive law
        stress_vector = self.D @ strain_vector

        # Update stress state
        self.stress['sigma_xx'] = stress_vector[0]
        self.stress['sigma_yy'] = stress_vector[1]
        self.stress['tau_xy'] = stress_vector[2]

        # Update strain state
        self.strain['epsilon_xx'] = strain_vector[0]
        self.strain['epsilon_yy'] = strain_vector[1]
        self.strain['gamma_xy'] = strain_vector[2]

    def get_forces_2D(self):
        """
        Return current stress vector in Voigt notation.

        Returns
        -------
        np.ndarray
            Stress vector [σ_xx, σ_yy, τ_xy]
        """
        return np.array([
            self.stress['sigma_xx'],
            self.stress['sigma_yy'],
            self.stress['tau_xy']
        ])

    def get_strain_2D(self):
        """
        Return current strain vector in Voigt notation.

        Returns
        -------
        np.ndarray
            Strain vector [ε_xx, ε_yy, γ_xy]
        """
        return np.array([
            self.strain['epsilon_xx'],
            self.strain['epsilon_yy'],
            self.strain['gamma_xy']
        ])

    def get_k_tan_2D(self):
        """
        Return tangent stiffness matrix (for linear elastic: D = D_tangent).

        Returns
        -------
        np.ndarray
            Tangent constitutive matrix (3x3)
        """
        return self.D

    def get_k_init_2D(self):
        """
        Return initial stiffness matrix.

        Returns
        -------
        np.ndarray
            Initial constitutive matrix (3x3)
        """
        E, nu = self.stiff0['E'], self.nu
        c = E / (1 - nu ** 2)
        return c * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])


class PlaneStrain(Material):
    """
    Isotropic linear elastic material in plane strain conditions.
    Inherits from Material and extends it for 2D continuum behavior.
    """

    def __init__(self, E, nu, rho, corr_fact=1, shear_def=True):
        # Initialize parent class (keeps 1D interface behavior intact)
        super().__init__(E, nu, corr_fact, shear_def)

        # Store additional FE-specific parameters
        self.nu = nu
        self.rho = rho
        self.tag = 'PLANE_STRAIN'

        # Extend stress dict for 2D continuum (Voigt notation: σ_xx, σ_yy, τ_xy)
        self.stress['sigma_xx'] = 0.0
        self.stress['sigma_yy'] = 0.0
        self.stress['tau_xy'] = 0.0

        # Extend strain dict for 2D continuum (Voigt notation: ε_xx, ε_yy, γ_xy)
        self.strain['epsilon_xx'] = 0.0
        self.strain['epsilon_yy'] = 0.0
        self.strain['gamma_xy'] = 0.0

        # Commit initial state
        self.commit()

    @property
    def D(self) -> np.ndarray:
        """
        Constitutive matrix for plane strain (3x3).
        Relates stress to strain: {σ} = [D]{ε}
        """
        E, nu = self.stiff['E'], self.nu
        c = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        return c * np.array([
            [1, nu / (1 - nu), 0],
            [nu / (1 - nu), 1, 0],
            [0, 0, (1 - 2 * nu) / (2 * (1 - nu))]
        ])

    def update_2D(self, strain_vector):
        """
        Update 2D stress state based on strain increment or total strain.

        Parameters
        ----------
        strain_vector : np.ndarray
            Strain vector in Voigt notation [ε_xx, ε_yy, γ_xy]
        """
        # Compute stress from strain using constitutive law
        stress_vector = self.D @ strain_vector

        # Update stress state
        self.stress['sigma_xx'] = stress_vector[0]
        self.stress['sigma_yy'] = stress_vector[1]
        self.stress['tau_xy'] = stress_vector[2]

        # Update strain state
        self.strain['epsilon_xx'] = strain_vector[0]
        self.strain['epsilon_yy'] = strain_vector[1]
        self.strain['gamma_xy'] = strain_vector[2]

    def get_forces_2D(self):
        """
        Return current stress vector in Voigt notation.

        Returns
        -------
        np.ndarray
            Stress vector [σ_xx, σ_yy, τ_xy]
        """
        return np.array([
            self.stress['sigma_xx'],
            self.stress['sigma_yy'],
            self.stress['tau_xy']
        ])

    def get_strain_2D(self):
        """
        Return current strain vector in Voigt notation.

        Returns
        -------
        np.ndarray
            Strain vector [ε_xx, ε_yy, γ_xy]
        """
        return np.array([
            self.strain['epsilon_xx'],
            self.strain['epsilon_yy'],
            self.strain['gamma_xy']
        ])

    def get_k_tan_2D(self):
        """
        Return tangent stiffness matrix (for linear elastic: D = D_tangent).

        Returns
        -------
        np.ndarray
            Tangent constitutive matrix (3x3)
        """
        return self.D

    def get_k_init_2D(self):
        """
        Return initial stiffness matrix.

        Returns
        -------
        np.ndarray
            Initial constitutive matrix (3x3)
        """
        E, nu = self.stiff0['E'], self.nu
        c = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        return c * np.array([
            [1, nu / (1 - nu), 0],
            [nu / (1 - nu), 1, 0],
            [0, 0, (1 - 2 * nu) / (2 * (1 - nu))]
        ])