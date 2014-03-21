#!/usr/bin/env/python

"""
Prototype implementation of aftershock models
"""

import numpy as np


class BaseAftershockModel(object):
    """
    Base Class for implementation of Aftershock Models
    """
    def __init__(self, config):
        """
        """
        self.params = config
        self.rate = None

class ReasenbergJones1989(BaseAftershockModel):
    """
    Implements the Reasenberg & Jones (1989) model for aftershock rate
    """
    def get_rates(self, magnitudes, elapsed_time, mainshock):
        """
        """
        self.rate = np.zeros(len(magnitudes), dtype=float)
        self.rate = 10.0 ** (self.params['a'] + self.params['b'] *
            (mainshock - magnitudes)) * ((elapsed_time + self.params['c']) **
            self.params['p'])


class ETAS(BaseAftershockModel):
    """
    Implements the ETAS (Ogata, 1999) model for calculating aftershock rate
    """
    def get_rates(self, magnitudes, elapsed_time, mainshock):
        """
        """
        self.rate = np.zeros(len(magnitudes), dtype=float)


    def omori_utsu(self, mag, t_val, t_i, mmin)):
        """

        """
        value = self.params['K'] / ((self.params['c'] + t_val - t_i) **
                                    self.params['p'])
        return value * (10.0 ** (self.params['alpha'] * (mag - mmin)))


class ShapiroEtAl2010(BaseAftershockModel):
    """

    """
    def get_rates(self, magnitudes, elapsed_time):
        """

        """
        self.rates = np.log10(self.params['Fluid Volume']) -\
                self.params['b'] * magnitudes +\
                self.params['Sigma']
        self.rates = np.array([np.sum(self.rates[i:]) 
                               for i in range(0, len(self.rates))])



class ConvertitoEtAl2012(BaseAftershockModel):
    """

    """
    def get_rates(self, magnitudes, elapsed_time=None):
        """
        """

