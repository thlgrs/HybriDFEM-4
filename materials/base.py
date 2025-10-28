# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:24:41 2024

@author: ibouckaert
"""
import numpy as np
from copy import deepcopy

class Material:

    def __init__(self, E, nu, corr_fact=1, shear_def=True):

        self.stiff = {}
        self.stiff0 = {}
        self.stress = {}
        self.strain = {}
        self.state = {}

        self.stiff['E'] = E
        self.stiff0['E'] = E

        self.tag = 'LINEL'

        self.chi = corr_fact
        self.shear_def = shear_def

        if self.shear_def:
            self.stiff['G'] = (1 / self.chi) * E / (2 * (1 + nu))
            self.stiff0['G'] = (1 / self.chi) * E / (2 * (1 + nu))
        else:
            self.stiff['G'] = 1e6 * (1 / self.chi) * E / (2 * (1 + nu))
            self.stiff0['G'] = 1e6 * (1 / self.chi) * E / (2 * (1 + nu))

        self.stress['s'] = 0
        self.stress['t'] = 0
        self.strain['e'] = 0
        self.strain['g'] = 0

        self.tol_disp = 1e-20

        self.commit()

    def copy(self):

        return deepcopy(self)

    def commit(self):

        self.stress_conv = deepcopy(self.stress)
        self.strain_conv = deepcopy(self.strain)
        self.stiff_conv = deepcopy(self.stiff)
        self.state_conv = deepcopy(self.state)

    def revert_commit(self):

        self.stress = deepcopy(self.stress_conv)
        self.strain = deepcopy(self.strain_conv)
        self.stiff = deepcopy(self.stiff_conv)
        self.state = deepcopy(self.state_conv)

    def get_forces(self):

        return np.array([self.stress['s'], self.stress['t']])

    def set_elongs(self, eps, gamma):

        self.strain['e'] = eps
        self.strain['g'] = gamma

    def update(self, dL):

        self.strain['e'] += dL[0]
        self.strain['g'] += dL[1]

        self.stress['s'] = self.stiff['E'] * self.strain['e']
        self.stress['t'] = self.stiff['G'] * self.strain['g']

    def get_k_tan(self):

        return (self.stiff['E'], self.stiff['G'], 0, 0)

    def get_k_init(self):

        return (self.stiff0['E'], self.stiff0['G'], 0, 0)

    def to_ommit(self):

        return False
