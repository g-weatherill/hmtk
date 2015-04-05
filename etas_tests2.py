#!/usr/bin/env/python
"""
Tests for the simple space-time ETAS function
"""
import numpy as np
import matplotlib.pyplot as plt
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon
from hmtk.seismicity.catalogue import Catalogue
from hmtk.parsers.catalogue.csv_catalogue_parser import CsvCatalogueParser
import etas_work_v2 as ets

# Build the polygon

area1 = Polygon([Point(10.0, 10.0),
                 Point(10.0, 11.0),
                 Point(11.0, 11.0),
                 Point(11.0, 10.0)])

mesh1 = area1.discretize(2.0)

test_cat = "demo_cat1/etas_testcat1.csv"
parser = CsvCatalogueParser(test_cat)
cat1 = parser.read_file()
plt.plot(mesh1.lons, mesh1.lats, "o")
plt.plot(cat1.data["longitude"], cat1.data["latitude"], "rs")

params = {"K": 0.053,
          "c": 0.002,
          "p": 1.06,
          "alpha": 0.94,
          "mu": 0.08,
          "b": 1.0,
          "time_window": 1.0}




