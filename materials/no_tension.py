from materials.base import Material

class NoTension_Mat(Material):

    def __init__(self, E, nu, corr_fact=1, shear_def=True):
        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)
        self.tag = 'NTMAT'

    def to_ommit(self):
        if self.strain['e'] > 0:
            return True
        return False
