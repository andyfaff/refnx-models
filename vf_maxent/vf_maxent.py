"""
A Component for maximum entropy of a volume fraction profile
"""
__author__ = 'Andrew Nelson'
__copyright__ = "Copyright 2019, Andrew Nelson"
__license__ = "3 clause BSD"

import numpy as np

from refnx.reflect import ReflectModel, Structure, Component, SLD, Slab
from refnx.analysis import (Bounds, Parameter, Parameters,
                            possibly_create_parameter)


class VFMaxEnt(Component):
    """
    """

    def __init__(self, G, sld, betas, total_thickness, alpha, rough,
                 name=''):
        super(VFMaxEnt, self).__init__(name=name)

        self.G = possibly_create_parameter(G, name='Dry thickness')

        if isinstance(sld, SLD):
            self.sld = sld
        else:
            self.sld = SLD(sld)

        # betas are the vf of each of the 'slabs' in the VFP
        self.betas = Parameters(name='betas')
        for i, beta in enumerate(betas):
            p = possibly_create_parameter(
                beta,
                name='%s - spline beta[%d]' % (name, i))
            p.range(0.00001, 1)
            self.betas.append(p)

        self.alpha = possibly_create_parameter(alpha,
                                               name='Regularising parameter')
        self.rough = possibly_create_parameter(rough,
                                               name='Roughness')
        self.total_thickness = possibly_create_parameter(total_thickness,
                                               name='total_thickness')

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.G, self.sld.parameters, self.betas, self.alpha,
                  self.rough, self.total_thickness])
        return p

    @property
    def slab_thickness(self):
        return float(self.total_thickness) / len(self.betas)

    @property
    def adsorbed_amount(self):
        betas = np.array(self.betas)
        return np.sum(betas * self.slab_thickness)

    def logp(self):
        # prior for the profile

        # prior for adsorbed amount
        logp = self.G.logp(self.adsorbed_amount)

        # Shannon-Jaynes entropy prior for regularisation
        betas = np.array(self.betas)
        mj = float(self.G / self.total_thickness)
        S = betas - mj - betas * np.log(betas/mj)
        return logp + np.sum(float(self.alpha) * S)

    def slabs(self, structure=None):
        num_slabs = len(self.betas)

        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = self.slab_thickness
        slabs[:, 1] = float(self.sld.real)
        slabs[:, 2] = float(self.sld.imag)
        slabs[:, 3] = float(self.rough)
        slabs[:, 4] = 1 - np.array(self.betas)
        slabs[0, 3] = 4

        return slabs
