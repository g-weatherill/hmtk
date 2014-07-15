#!/usr/bin/env/python

"""
Simple unit tests for the ETAS tools
"""
import unittest
import numpy as np
import hmtk.etas_work as etas


class TestETASParams(unittest.TestCase):
    """

    """
    def setUp(self):
        """
        """
        self.params = {"b": 1.0,
                       "p": 1.2,
                       "c": 0.01,
                       "n": 0.95,
                       "alpha": 0.8,
                       "m0": 0.0,
                       "md": 2.0,
                       "mmax": 8.0,
                       "d": 0.001,
                       "mu": 3.0,
                       "gamma": 0.5,
                       "mu_rate": 10.0}

    def test_tstar(self):
        """

        """
        self.assertAlmostEqual(etas._get_tstar(self.params), 24761.0, 1)

    def test_get_kval_alpha_b_not_equal(self):
        """

        """
        self.assertAlmostEqual(etas._get_kval(self.params), 0.01551786, 7)

    def test_get_kval_alpha_equal(self):
        """

        """
        self.params["alpha"] = 1.0
        self.assertAlmostEqual(etas._get_kval(self.params), 4.10627383E-3, 7)
        self.params["alpha"] = 0.8

    def test_aftershock_productivity(self):
        """

        """
        params = etas._get_aftershock_productivity(self.params)
        self.assertAlmostEqual(params["K"], 0.0001551786, 7)

    def test_get_direct_number_aftershocks(self):
        """
        """
        self.params["K"] = None
        self.assertAlmostEqual(
            etas._get_number_direct_aftershocks(self.params),
            122.97083282, 7)

    def test_get_missing_event_correction(self):
        """

        """
        params = etas._get_missing_event_correction(self.params),
        self.assertAlmostEqual(self.params["drho"], 4.67004552E-2, 7)

