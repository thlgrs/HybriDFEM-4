# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""

import numpy as np
from copy import deepcopy
from materials.base import Material

class Bilinear_Mat(Material):

    def __init__(self, E, nu, fy, alpha, corr_fact=1, shear_def=True):

        super().__init__(E, nu, corr_fact=corr_fact, shear_def=shear_def)

        self.stress['f_y'] = fy
        self.strain['e_y'] = fy / E
        self.a = alpha
        self.tag = 'BILIN'
        self.commit()

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]

        # Elastic step:
        d_e = abs(self.strain['e']) - self.strain['e_y']

        if d_e <= 0:
            self.stress['s'] = self.stiff0['E'] * self.strain['e']
            self.stiff['E'] = deepcopy(self.stiff0['E'])

        else:
            d_f = d_e * self.a * self.stiff0['E']

            self.stress['s'] = (self.stress['f_y'] + d_f) * np.sign(self.strain['e'])
            self.stiff['E'] = self.a * self.stiff0['E']

        self.stress['t'] = self.stiff0['G'] * self.strain['g']
        self.stiff['G'] = deepcopy(self.stiff0['G'])
