#!/usr/bin/env/python

"""
General codes associated with the Space-Time ETAS model
"""

import random
import numpy as np
from copy import deepcopy
from math import fabs, asin, radians, sin, sqrt, log10
from openquake.nrmllib.hazard.parsers import SourceModelParser
from openquake.commonlib.source import NrmlHazardlibConverter


# Get source model from file

def get_source_model(ifile, inv_time, rupt_mesh_spacing=1.0, mfd_width=0.1,
        area_discretisation=10.0):
    """

    """
    parser = SourceModelParser(ifile)
    model = parser.parse()
    sources = list(model.sources)
    converter = NrmlHazardlibConverter(inv_time, rupt_mesh_spacing, mfd_width,
                                       area_discretisation)
    src_model = []
    for source in sources:
        src_model.append(converter(source))
    return src_model


def _get_tstar(params):
    """
    Returns the characteristic time of aftershocks
    """
    nval = params["n"]
    return params["c"] * ((nval / (1. - nval)) ** (1. / (params["p"] - 1.)))


def _get_branching_ratio(params):
    """
    Returns the branching ratio
    """
    theta = params["p"] - 1.0
    denom = (params["alpha"] - params["b"]) / ((params["c"] ** theta) / theta)
    return (params["K"] * params["b"]) / denom



def _get_kval(params):
    """
    Returns the unscaled K-value
    """
    if "K" in params.keys() and params["K"]:
        return params["K"]

    if np.fabs(params["alpha"] - params["b"]) < 1E-17:
        #x_1 = (params["p"] - 1.0) / (params["c"] ** (params["p"] - 1.0))
        denom = params["b"] * np.log(10.0) * (params["mmax"] - params["m0"]) /\
            (params["p"] - 1.0) / (params["c"] ** (params["p"] - 1.0))
    else:
        x_1 = 1. - (10.0 ** ((params["alpha"] - params["b"]) *
                             (params["mmax"] - params["m0"])))
        #x_1 = (params["p"] - 1.0) * x_1
        denom = params["b"] / (params["b"] - params["alpha"]) /\
            (params["p"] - 1.) / (params["c"] ** (params["p"] - 1.0))
        denom = denom * x_1
    return params["n"] / denom

def _get_aftershock_productivity(params):
    """
    Returns total aftershock productivity (K-value)
    """
    params["K"] = _get_kval(params)
    scale = 10.0 ** (-params["b"] * (params["md"] - params["m0"]))
    params["K"] *= scale
    return params


def _get_number_direct_aftershocks(params, mag):
    """
    Returns the number of direct aftershocks
    """
    params = _get_aftershock_productivity(params)
    return params["K"] * (10.0 ** ((params["alpha"] * (mag - params["m0"]))))/\
        (params["p"] - 1.0) / (params["c"] ** (params["p"] - 1.0))

def _get_missing_event_correction(params):
    """
    Returns 'drho'
    """
    x_1 = 1.0 - (10.0 ** ((params["alpha"] - params["b"]) *
                          (params["md"] - params["m0"])))
    x_1 = (10.0 ** (params["b"] * (params["md"] - params["m0"]))) * x_1
    params["drho"] = (params["K"] * params["b"]) /\
        (params["b"] - params["alpha"]) * x_1
    return params

def total_aftershock_rate(magnitude, params):
    """

    """
    exponent = 10.0 ** (params["alpha"] * (magnitude - params["m0"]))
    return (params["K"] * exponent) + params["drho"]

def _get_next_event_time(params, magnitude, event_time, max_time):
    """

    """
    rho = total_aftershock_rate(magnitude, params)
    sample = random.uniform(0., 1.)
    if fabs(params["p"] - 1.) < 1E-16:
        dtime = np.exp(
            np.log(params["c"] + aft_time) - ((1. / rho) * np.log(sample))
            ) - params["c"] - event_time
    else:
        x_1 = np.exp((rho / (1.0 - params["p"])) *\
            ((event_time + params["c"]) ** (1.0 - params["p"])))
        if sample < x_1:
            return None
        else:
            x_2 = ((event_time + params["c"]) ** (1.0 - params["p"])) - \
                    ((1.0 - params["p"]) * np.log(sample) / rho)
            dtime = x_2 ** (1.0 / (1.0 - params["p"])) - params["c"] - event_time
    return event_time + dtime


def _get_aftershock_distance(magnitude, params, sample):
    """

    """
    x_1 = ((1.0 - sample) ** (-1. / params["mu"])) - 1.0
    return (params["d"] * x_1) * (10.0 ** (params["gamma"] * magnitude))


def _get_aftershock_location(location, magnitude, params):
    """

    """
    rhyp = _get_aftershock_distance(magnitude,
                                    params,
                                    random.uniform(0., 1.))
    # Minimum angle
    theta1 = -asin(min(1.0, radians(location.depth / rhyp)))
    theta2 = asin(min(1.0, radians((params["zmax"] - location.depth) / rhyp)))
    theta = random.uniform(theta1, theta2)
    dz = rhyp * sin(theta)
    repi = sqrt((rhyp ** 2.) - (dz ** 2.))
    azimuth = np.random.uniform(0., 360.) # Isotropic
    return location.point_at(repi, dz, azimuth)

class SimpleEvent(object):
    """
    Simple holder for basic rupture information
    """
    def __init__(self, event_time, hypocentre, magnitude,
            rake, typology, surface, trt, mainshock_id, is_mainshock):
        """
        """
        self.time = event_time
        self.hypocentre = hypocentre
        self.mag = magnitude
        self.rake = rake
        self.typology = typology
        self.surface = surface
        self.trt = trt
        self.mainshock_id = mainshock_id
        self.is_mainshock = is_mainshock


def get_aftershock_catalogue(event, max_time, params):
    """

    """
    # Get total aftershock rate
    rho = total_aftershock_rate(magnitude, params)
    # Get event time list
    aftershock_sequence = []
    #print magnitude, event_time, location
    aft_time = _get_next_event_time(params,
                                    event.magnitude,
                                    event.time,
                                    max_time)
    while aft_time and (aft_time <= max_time):
        #time_list.append(aft_time)
        aft_locn =_get_aftershock_location(location, magnitude, params)
        #loc_list.append(aft_locn)
        aft_mag = np.inf
        while aft_mag > params["mmax"]:
            aft_mag = params["m0"] - (log10(random.uniform(0., 1.)) /
                                      params["b"])
        #mag_list.append(aft_mag)
        #print aft_mag, aft_time, aft_locn
        aft_time = _get_next_event_time(params, rho, aft_time, max_time)
        aft_surface = PlanarSurface(1.0, event.surface.strike,
            event.surface.dip, aft_locn, aft_locn, aft_locn, aft_locn)
        aftershock = SimpleEvent(aft_time, aft_locn, aft_mag, event.rake,
            event.typology, aft_surface, event.trt, event.mainshock_id, False)
        aftershock_sequence.append(aftershock)
    return aftershock_sequence


def convert_rates(source, max_time):
    """
    Convert the probabilities of occurence from annual to the time
    period under consideration
    """
    ratio = max_time / 365.25

    if isinstance(source.mfd, TruncatedGRMFD):
        rate = 10.0 ** (source.mfd.a_val -
                        source.mfd.b_val * source.mfd.min_mag)
        rate = rate * ratio
        source.mfd.a_val = log10(rate) + source.mfd.b_val * source.mfd.min_mag
    else:
        source.mfd.occurrence_rates = [val * ratio
                                       for val in source.mfd.occurrence_rates]
    return source


SECS_PER_DAY = 24. * 3600.
SECS_PER_YEAR = 365.25 * SECS_PER_DAY

def get_event_times(source, max_time):
    """
    Returns a list of event times occuring up until the source time
    """
    # Convert maximum time to seconds
    max_seconds = max_time * SECS_PER_DAY
    # Convert source rates to rates per second
    #if isinstance(source.mfd, TruncatedGRMFD):
    #    source.mfd.a_val = source.mfd.a_val - np.log10(SECS_PER_YEAR)
    #else:
    #    source.mfd.occurrence_rates = [val / SECS_PER_YEAR
    #                                   for val in source.mfd.occurrence_rates]
    annual_rate = np.sum(np.array([val[1]
                         for val in source.get_annual_occurrence_rates()]))
    # Get rate per second
    rate = annual_rate / SECS_PER_YEAR
    # Sample event times
    event_times = []
    end_time = random.expovariate(rate)
    while end_time < max_seconds:
        print max_seconds, rate, end_time
        event_times.append(end_time)
        end_time += random.expovariate(rate)
    return event_times


def get_background_events(source, rupture_set, max_time, start_time=0.0):
    """
    Returns a set of event times derived from the background source
    """
    # Convert annual rates to rates in event_time (in days)
    event_times = get_event_times(source, max_time)
    if len(event_times) == 0:
        # No ruptures occured
        return []
    else:
        # Sample ruptures (with replacement)
        sampler = np.random.randint(0, len(rupture_set), len(event_times))
    catalogue = []
    for iloc, locn in enumerate(sampler):
        rup = rupture_set[locn]
        event = SimpleEvent(event_times[iloc], rup.hypocenter, rup.mag,
                            rup.rake, rup.source_typology, rup.surface,
                            rup.tectonic_region_type, iloc, True)
        catalogue.append(event)
    return catalogue


def get_etas_catalogue_for_event(event, params, max_time, max_number_events):
    """
    Get an aftershock sequence for a catalogue, based on the ETAS model
    """
    catalogue = [event]
    extend_list = True
    iloc = 0
    while (iloc < len(catalogue)) and (len(catalogue) <= max_number_events):
        # Get aftershocks
        aftershocks = get_aftershock_catalogue(catalogue[iloc], max_time,
                                               params)
        if len(aftershocks > 0):
            catalogue.extend(aftershocks)
        iloc += 1
    return catalogue


def get_etas_catalogue_for_source(source, params, max_time, start_time,
        max_number_events):
    """
    Constructs the full Space-Time ETAS aftershock sequence for an event
    """
    # Generate ruptures
    rupture_set = list(source.iter_ruptures())
    background_catalogue = get_background_events(source,
                                                 rupture_set,
                                                 max_time,
                                                 start_time)
    print "Background Catalogue Contains %s Events" % len(background_catalogue)
    catalogue = []
    for iloc, mainshock in enumerate(background_catalogue):
        # Get event sequence
        sequence = get_etas_catalogue_for_event(mainshock,
                                                params,
                                                max_time,
                                                max_number_events)
        print "Mainshock #1 Produces %s Aftershocks" %(iloc, len(sequence))
        catalogue.extend(sequence)
    print "Total Catalogue Contains %s event" % len(catalogue)
    print "Sorting ..."
    time_list = np.array([event.time for event in catalogue])
    idx = np.argsort(time_list)
    return [catalogue[loc] for loc in idx]

