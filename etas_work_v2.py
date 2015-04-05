#!/usr/bin/env/python

"""
General codes associated with the Space-Time ETAS model
"""

import random
import numpy as np
from datetime import timedelta
from copy import deepcopy
from math import fabs, asin, radians, sin, sqrt, log10
from openquake.commonlib.source import parse_source_model
from openquake.commonlib.sourceconverter import SourceConverter
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD
from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.geo.geodetic import distance
from openquake.hazardlib.source.rupture import Rupture
from hmtk.seismicity.catalogue import Catalogue

class SpatialDensityFunction(object):
    """
    Class to distribute the spatial density of an event given an initial
    set of parameters
    """
    def __init__(self, params, lons, lats, depths):
        """

        """
        self.params = params
        self.mesh = np.column_stack([lons, lats, depths])
        self.npts = np.shape(self.mesh)[0]
        self._build_constants()

    @classmethod
    def from_mesh(cls, params, mesh):
        """
        Instantiate the class from an openquake Mesh object
        """

        if len(mesh.lons.shape) > 1:
            n_y, n_x = mesh.lons.shape
            npts = n_y * n_x
            lons = np.reshape(np.copy(mesh.lons), [npts, 1]).flatten()
            lats = np.reshape(np.copy(mesh.lats), [npts, 1]).flatten()
            depths = np.reshape(np.copy(mesh.depths), [npts, 1]).flatten()
        else:
            npts = mesh.lons.shape[0]
            lons = np.copy(mesh.lons)
            lats = np.copy(mesh.lats)
            if mesh.depths is not None:
                depths = np.copy(mesh.depths)
            else:
                depths = np.zeros_like(lons)

        return cls(params, lons, lats, depths)

    def _build_constants(self):
        """
        Builds any constants from the parameters and adds them to the
        parameters dictionary
        """
        pass

    def calculate(self, hypo_lon, hypo_lat, hypo_depth, magnitude=None):
        """

        """
        raise NotImplementedError

class BachHainzlDensity(SpatialDensityFunction):
    """

    """

    def _build_constants(self):
        """
        The numerator of the model is constant from the parameters
        denom = (q - 1) d ** (2 * (q - 1))
        """
        self.params["numer"] = (self.params["q"] - 1.0) * (self.params["d"] **
            (2.0 * (self.params["q"] - 1.0)))


    def calculate(self, hypo_lon, hypo_lat, hypo_depth, magnitude=None):
        """

        """
        d_val = distance(self.mesh[:, 0], self.mesh[:, 1], self.mesh[:, 2],
                          hypo_lon, hypo_lat, hypo_depth)
        denominator = (d_val ** 2.0) + (self.params["d"] ** 2.)
        return self.params["numer"] /\
            (np.pi * (denominator ** self.params["q"]))


def omoris_law(days, kval, cval, pval):
    """
    Separate implementation of Omori's law - for testing
    """
    return kval / ((cval + days) ** pval)


class SimpleETAS(object):
    """
    Test module for simple ETAS (no spatial dependence)
    """
    def __init__(self, params, catalogue, spatial_model=None):
        """
   
        """
        self.params = _check_params(params)
        assert isinstance(catalogue, Catalogue)
        self.catalogue = catalogue
        _ = self.catalogue.add_datetime()
        self.start_time = None
        self.end_time = None
        self.rates = None
        self.spatial_model = spatial_model
        if self.spatial_model:
            self.npts = self.spatial_model.npts
        else:
            self.npts = 0

    def _check_params(self, params):
        """
        Checks to ensure all required parameters are available
        """
        for param in ["K", "c", "p", "alpha", "mu", "b", "time_window"]:
            assert param in params.keys()
        return params

    def get_probability_in_time_window(self, start_time, window_length, mmin,
            mmax, bin_width):
        """
        :param start_time:
            Start time as instance of datetime.datetime object
        """
        mag_dist = self._setup_rates(start_time,
                                     window_length,
                                     mmin,
                                     mmax,
                                     bin_width)
        for iloc in range(0, self.catalogue.get_number_events()):
            aftershock_rate = self.get_magnitude_rate(
                self.catalogue.data["datetime"][iloc],
                self.catalogue.data["magnitude"][iloc],
                mmin)
            aftershock_rate = self._distribute_rate(aftershock_rate,
                                                    mag_dist,
                                                    bin_width)
            if self.spatial_model:
                spatial_density = self.spatial_model.calculate(
                    self.catalogue.data["longitude"][iloc],
                    self.catalogue.data["latitude"][iloc],
                    self.catalogue.data["depth"][iloc],
                    self.catalogue.data["magnitude"][iloc])
            else:
                spatial_density = 1.0
            for jloc, mag_rate in enumerate(aftershock_rate):
                self.rates[:, jloc] += (mag_rate * spatial_density)
        return self.rates

    def _setup_rates(self, start_time, window_length, mmin, mmax, bin_width):
        """

        """
        self.start_time = start_time
        mag_dist = np.arange(mmin, mmax + bin_width, bin_width)
        total_rate = (self.params["mu"] / self.params["time_window"]) *\
           window_length / float(self.npts)
        self.rates = np.tile(
            self._distribute_rate(total_rate, mag_dist, bin_width),
            [self.npts, 1])
        self.end_time = start_time + timedelta(days = int(window_length))
        return mag_dist

    def _distribute_rate(self, rate, magnitudes, bin_width):
        """
        Given an input activity rate, distribute between Mmin and Mmax using
        the truncated Gutenberg-Ricter distribution 
        """
        beta = self.params["b"] * np.log(10.0)
        m_0 = magnitudes[0]
        m_u = magnitudes[-1]
        central_mags = magnitudes[:-1] + (bin_width / 2.0)
        f_m = beta * np.exp(-beta * (central_mags - m_0)) /\
                (1.0 - np.exp(-beta * (m_u - m_0))) * bin_width
        return rate * f_m

    def get_magnitude_rate(self, event_time, magnitude, mmin):
        """
        Returns the rate of events for an aftershock sequence of a given
        magnitude
        """
        magnitude_scale = 10.0 ** (self.params["alpha"] * (magnitude - mmin))
        days = np.arange(float((self.start_time - event_time).days),
                         float((self.end_time - event_time).days) + 1.0,
                         1.0)
        return magnitude_scale * np.sum(omoris_law(days,
                                                   self.params["K"],
                                                   self.params["c"],
                                                   self.params["p"]))



def etas_adjusted_magnitude_rate(etas_rates, magnitude, bin_width, beta,
        mmin, mmax):
    """

    """
    f_m = beta * np.exp(-beta * (magnitude - mmin))  /\
        (1.0 - np.exp(-beta * (mmax - mmin)))
    return etas_rates * f_m * bin_width

def distance_probability(dval, qval, probs):
    """

    """
    numer = (qval - 1.0) * (dval ** (qval - 1.0))
    return np.sqrt(((numer / (np.pi * probs)) ** (1. / qval)) - (dval ** 2.0))

class RunETASSimulation(object):
    """

    """
    def __init__(self, params, source_model, start_time, time_window,
            catalogue=None, bin_width=0.1, threshold_days=1000.0):
        """
        From a source model generate a stochastic event set based on an
        ETAS simulation
        """
        self.params = self._check_params(params)
        self.source_model = source_model
        self.catalogue = catalogue
        self.erf = []
        self.upper_seismogenic_depth = np.inf
        self.lower_seismogenic_depth = -np.inf
        for source in source_model:
            if source.upper_seismogenic_depth < self.upper_seismogenic_depth:
                self.upper_seismogenic_depth = source.upper_seismogenic_depth
            if source.lower_seismogenic_depth < self.lower_seismogenic_depth:
                self.lower_seismogenic_depth = source.lower_seismogenic_depth
            self.erf.extend(list(source.iter_ruptures))
        # Adjust rates in the ERF from annual to daily rate
        self.mmin = np.inf
        self.mmax = -np.inf
        self.nrupts = len(self.erf)
        self.bin_width = bin_wdith
        # Setup arrays
        self.hypos = np.empty([self.nrupts, 3])
        self.mags = np.empty(self.nrupts, 3)
        self.mu = np.empty(self.nrupts)
        for iloc, erf in enumerate(self.erf):
            erf.occurrence_rate /= 365.25
            self.hypos[iloc, :]= np.array([rup.hypocenter.longitude,
                                           rup.hypocenter.latitude,
                                           rup.hypocenter.depth])
            self.mags[iloc] = erf.mag
            self.mu[iloc] = erf.occurrence_rate
            if erf.mag < self.mmin:
                self.mmin = erf.mag
            if erf.mag > self.mmax:
                self.mmax = erf.mag


        self.time_window = time_window
        self.start_time = start_time
        self.end_time = self.start_time + timedelta(time_window)
        if self.catalogue:
            _ = self.catalogue.add_datetime()
            self.update_erf_from_catalogue(self)


    def _check_params(self, params):
        """
        Checks to ensure all required parameters are available
        """
        for param in ["K", "c", "p", "alpha", "b", "d", "q"]:
            assert param in params.keys()
        param["beta"] = param["b"] * log(10.0)
        self.params["numer"] = (self.params["q"] - 1.0) * (self.params["d"] **
            (2.0 * (self.params["q"] - 1.0)))
        return params
    
     def update_erf_from_catalogue(self):
        """
        """
        for iloc in self.catalogue.get_number_events():
            magnitude_rate = self.get_magnitude_rate(
                self.catalogue.data["datetime"][iloc],
                self.catalogue.data["magnitude"][iloc])
            if magnitude_rate
                # Skip
                continue
            # Get distance rates
            dist_rates = self.get_distance_rates(hypos[:, 0], hypos[:, 1],
                hypos[:, 2],
                self.catalogue.data["longitude"][iloc],
                self.catalogue.data["latitude"][iloc],
                self.catalogue.data["depth"][iloc])
            magnitude_rate *= dist_rates
            total_rate = etas_adjusted_magnitude_rates(magnitude_rate,
                                                       self.mags,
                                                       self.bin_width,
                                                       self.params["beta"],
                                                       self.mmin,
                                                       self.mmax)
            self.mu += total_rate
        for iloc, erf in enumerate(self.erf):
            erf.occurrence_rate = self.mu[iloc]


    def generate_ses(self):
        """
        """
        background_ses = list(self.get_background_ses())
        # Add rupture dates
        sample_times = np.random.uniform(0.,
                                         self.time_window,
                                         len(background_ses))
        sample_times.sort()
        for iloc, event in enumerate(background_ses):
            setattr(event, "time",
                    self.start_time + time_delta(sample_times[iloc]))
            setattr(event, "event_index", 0)

        # Generate event set
        ses = []
        for iloc, event in enumerate(background_ses):
            # Sample number of aftershocks
            number_aftershocks, aftershock_times = \
                self.sample_number_aftershocks(event.mag, event.time)
            if number_aftershocks > 0:
                # Generate a sample time and sample location


        

    def sample_number_aftershocks(self, magnitude, event_time):
        """

        """
        d_t = float((self.end_time - event_time).days)
        d_t = np.arange(0., d_t, 1.0)
        aftershock_rate = (self.params["K"] * np.exp(magnitude - self.mmin)) /\
            ((d_t + self.params["c"]) ** self.params["p"])
        sample = np.random.poisson(aftershock_rate)
        aftershock_times = []
        if np.any(sample > 0):
            for iloc, nsamples in enumerate(samples):
                if nsamples > 0:
                    days = np.random.uniform(d_t[iloc],
                                             d_t[iloc] + 1.,
                                             nsamples)
                    for jloc in xrange(nsamples):

                        aftershock_times.append(
                            event_time + timedelta(days[jloc]))
            return np.sum(sample), aftershock_times
        else
            return 0, []
        

    def generate_aftershocks(self, number_aftershocks, aftershock_times,
            main_event):
        """

        """
        aftershock_set = []
        distances = distance_probability(
            self.params["q"],
            self.params["p"],
            np.random.uniform(0., 1., number_aftershocks))
        # Assumes isotropy in model
        azimuths = np.random.uniform(0., 360., number_aftershocks)
        # Get inclinations - random between limits
        ud_limit = (main_event.hypocenter.depth -
            self.upper_seimogenic_depth) / distances
        ud_limit[ud_limit > 1.] = 1.0
        ld_limit = (self.lower_seismogenic_depth -
             main_event.hypocenter.depth) / distances
        ld_limit[ld_limit > 1.] = 1.0
        ud_limit = np.arcsin(ud_limit)
        ld_limit = np.arcsin(ld_limit)
        for iloc in xrange(number_aftershocks):
            # Get location
            # Get inclination
            inclination = random.uniform(-ud_limit[iloc], ld_limit[iloc])
            vertical_distance = distances[iloc] * np.sin(inclination)
            horizontal_distance = np.sqrt(distances[iloc] ** 2.0 -
                                          vertical_distance ** 2.0)
            hypo_loc = main_event.hypocenter.point_at(horizontal_distance,
                                                      vertical_distance,
                                                      azimuth[iloc])




    def get_background_ses(self):
        """

        """
        for rupture in self.erf:
            for i in xrange(rupture.sample_number_of_occurrences()):
                yield rupture

    def sample_distance_probabilities(self, 

    def get_magnitude_rate(self, event_time, magnitude):
        """
        Returns the rate of events for an aftershock sequence of a given
        magnitude
        """
        magnitude_scale = 10.0 ** (self.params["alpha"] *
                                   (magnitude - self.mmin))
        days = np.arange(float((self.start_time - event_time).days),
                         float((self.end_time - event_time).days) + 1.0,
                         1.0)
        if days > self.threshold_days:
            return None
        return magnitude_scale * np.sum(omoris_law(days,
                                                   self.params["K"],
                                                   self.params["c"],
                                                   self.params["p"]))

    def get_distance_rates(self, lons, lats, depths, hypo_lon, hypo_lat,
            hypo_depth)
        d_val = distance(lons, lats, depths, hypo_lon, hypo_lat, hypo_depth)
        denominator = (d_val ** 2.0) + (self.params["d"] ** 2.)
        return self.params["numer"] /\
            (np.pi * (denominator ** self.params["q"]))

SPATIAL_MODELS = {"BachHainzl": BachHainzlDensity}
#class TimeDependentSourceModel(SimpleETAS):
#    """
#    Builds a time-dependent source model from a time-independent source
#    and adds a short-term aftershock model
#    """
#    def __init__(self, source_model, params, catalogue,
#            spatial_model_name="BachHainzl"):
#        """
#        """
#        full_erf = []
#        for nsrc, source in enumerate(source_model):
#            mag_dict = OrderedDict([
#                ("{:.3f}".format(mag), iloc) for iloc, (mag, rate) in
#                enumerate(source.get_annual_occurrence_rates()])
#            erf = list(source.iter_ruptures())
#            nrupts = len(erf)
#            nmags = len(mag_dict)
#            npp = len(source.hypocenter_distribution.data) *\
#                    len(source.nodal_plane_distribution.data)
#            rates = np.zeros([nrupts / nmags, nmags])
#            hypos = np.empty([nrupts / nmags, nmags])
#            # Pre-allocate ruptures
#            ruptures = [[None for i in range(0, nmags)]
#                         for j in range(0, nrupts / nmags)]
#            # Group rupture by location
#            idx = np.array([[ival] for ival in range(0, npp)])
#            idx = np.tile(idx, [1, nmags])
#            for jval in range(0, nmags):
#                idx[:, jval] += (nmags * j)
#            hidx = np.arange(0, npp)
#            for iloc in range(0, nrupts / nmags):
#                for kloc in range(0, npp):
#                    for jloc in range(0, nmags)
#                        rupture = erf[idx[kloc, jloc]]
#                        if kloc == 0:
#                            hypos[hidx[kloc], :] = np.array([
#                                rupture.hypocenter.longitude,
#                                rupture.hypocenter.latitude,
#                                rupture.hypocenter.depth])
#                        rates[hidx[kloc], jloc] += rupture.occurrence_rate
#                        ruptures[hidx[kloc], jloc] = deepcop(rupture)
#                hidx += npp
#                idx += (npp * nmags)
#            full_erf.extend(ruptures)
#            if nsrc == 0:
#                full_hypos = hypos.copy()
#                full_rates = rates.copy()
#            else:
#        
#
#
#
#            
#
#
#
#
#            for rupture in erf:
#                # Gather ruptures for a single location
#                for mctr in range(0, nmags):
#                    if mctr == 0:
#                        hypo_loc
#
#
#                idx_inner = np.arange(idx[0], idx[-1], )
#                for jloc in range(0, len(idx_inner)):
#
#                idx += (npp * nmags)
#            
#            mloc = mag_dict["{:.3f}".format(rupture.mag)]
#
#                rates[hloc, mloc] += rupture.occurrence_rate
#
#
#                hypos[hloc, :] = np.array([rupture.hypocenter.longitude,
#                                           rupture.hypocenter.latitude,
#                                           rupture.hypocentre.depth])
#
#            for rupture in enumerate(erf):
#
#
#                
#                if (iloc > 0) and (hloc 
#
#                if  (iloc % npp) == 0:
#                    hloc = 0
#                    ploc += 
#
#
#
#
#
#
#
#
#        # Generate mesh and rates from ERF
#        hypo_list = np.empty([len(erf), 3])
#        rates = np.empty(len(erf))
#        mags = np.empty(len(erf))
#        for iloc, rupture in enumerate(erf):
#            hypo_list[iloc, :] = np.array([rupture.hypocentre.longitude,
#                                           rupture.hypocentre.latitude,
#                                           rupture.hypocentre.depth])
#            rates[iloc] = rupture.occurrence_rate
#            mags[iloc] = rupture.mag
#        spatial_model = SPATIAL_MODELS[spatial_model_name](params,
#                                                           hypo_list[:, 0],
#                                                           hypo_list[:, 1],
#                                                           hypo_list[:, 2])
#        super(TimeDependentSourceModel, self).__init__(params,
#                                                       catalogue,
#                                                       spatial_model)
#        self.rates = rates
#
#    def _check_params(self, params):
#        """
#        Checks to ensure all required parameters are available
#        """
#        for param in ["K", "c", "p", "alpha", "b"]:
#            assert param in params.keys()
#        return params
#
#    def _setup_rates(self, start_time, window_length, mmin, mmax, bin_width):
#        """
#
#        """
#        self.start_time = start_time
#        # Convert rates from annual to days
#        self.rates /= 365.25
#        self.end_time = start_time + timedelta(days = int(window_length))
#        return mag_dist
#
#











class InducedETAS(SimpleETAS):
    """
    Modification of the ETAS class to permit the possibility of a change
    in baseline rate
    """
    def _check_params(self, params):
        """
        Checks to ensure all required parameters are available
        """
        for param in ["K", "c", "p", "alpha", "mu", "b", "time_window", "cf"]:
            if not param in params.keys():
                raise ValueError("Induced ETAS requires the parameter %s"
                                 % param)
        return params

    def _setup_rates(self, start_time, window_length, mmin, mmax, bin_width):
        """
        :param list window_length:
             In this example window_length corresponds to flow a list or
             array of expected flow_rates per day

           
        """
        self.start_time = start_time
        mag_dist = np.arange(mmin, mmax + bin_width, bin_width)
        total_rate = 0.0
        for flow_rate in enumerate(window_length):
            total_rate += ((self.params["mu"] / self.params["time_window"]) +
                (self.params["cf"] * flow_rate))
        total_rate /= float(self.npts)
        self.rates = np.tile(
            self._distribute_rate(total_rate, mag_dist, bin_width),
            [self.npts, 1])
        self.end_time = start_time + timedelta(days = int(window_length))
        return mag_dist

class ShapiroRates(SimpleETAS):
    """

    """
    def _check_params(self, params):
        """
        Checks to ensure all required parameters are available
        """
        for param in ["b", "sigma", "p",  "time_window", "q", "psi", "cmax"]:
            if not param in params.keys():
                raise ValueError("Shapiro Model requires the parameter %s"
                                 % param)
        return params

    def get_probability_in_time_window(self, start_time, cum_vol,
            window_length, mmin, mmax, bin_width):
        """
        :param start_time:
            Start time as instance of datetime.datetime object
        """
        self.start_time = start_time
        mag_dist = np.arange(mmin, mmax + bin_width, bin_width)
        total_rate = 0.0
        for volume in cum_vol:
            # Evaluate total rate from Shapiro's formula
            total_rate += 10.0 ** (np.log10(volume)  + self.params["sigma"])
 
        total_rate /= float(self.npts)
        if window_length > len(cum_vol):
            # Time window extends beyond shut-in
            d_t = np.arange(1, window_length - len(cum_vol) + 2, 1)
            injection_seis_rate = (self.params["q"] * self.params["psi"]) /\
                self.params["cmax"]
            total_rate += (injection_seis_rate / (d_t ** self.params["p"]))
        self.rates = np.tile(
            self._distribute_rate(total_rate, mag_dist, bin_width),
            [self.npts, 1])
        return self.rates

