#!/usr/bin/env/python

"""
Hazard Calculation Tools for Time-Dependent Effects
"""
import numpy as np
from scipy.stats import Poisson
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.source.non_parametric import \
    NonParametricSeismicSource
from openquake.hazardlib.gsim import get_available_gsims


def build_time_dependent_model(source_model, params, catalogue=None):
    """

    """
    time_dependent_model = []
    if not catalogue:
        # No time-dependence
        return source_model
    for source in source_model:
        # Get the ruptures
        nhypos = len(source.hypocenter_distribution.data) # Number hypocentres
        nmags = len(source.get_annual_occurrence_rates()) # Number mags
        rupture_set = list(source.iter_ruptures())
        nrupts = source.count_ruptures()
        rate_grid = np.empty([nrupts / nmags, nmags], dtype=float)
        hypocentres = np.empty([len(rupture_set), 3])
        hloc = 0
        for rupture in enumerate(rupture_set):
            for jloc in range(0, nmags):
            hypocentres[hloc, :] = np.array([rupture.hypocentre.longitude,
                                             rupture.hypocentre.latitude,
                                             rupture.hypocentre.depth])



