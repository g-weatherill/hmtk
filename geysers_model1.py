#!/usr/bin/env/python

import numpy as np
from decimal import Decimal
from openquake.hazardlib.mfd.truncated_gr import TruncatedGRMFD
from openquake.hazardlib.scalerel.point import PointMSR
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.source.area import AreaSource
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon
from openquake.hazardlib.pmf import PMF


geometry1 = Polygon([Point(-122.70080432106884, 38.83050716337448),
                     Point(-122.6511265434277, 38.76896603584888),
                     Point(-122.6511265434277, 38.70964928642661),
                     Point(-122.73936020819332, 38.71039074579439),
                     Point(-122.77495025784667, 38.76155144217109),
                     Point(-122.70080432106884, 38.83050716337448)])

geometry2 = Polygon([Point(-122.92843234697679, 38.88908245342897),
                     Point(-122.79941841698336, 38.88908245342897),
                     Point(-122.70080432106884, 38.83050716337448),
                     Point(-122.77495025784667, 38.760809982803316),
                     Point(-122.92991526571234, 38.800848788663345),
                     Point(-122.92991526571234, 38.800848788663345),
                     Point(-122.92843234697679, 38.88908245342897)])

upper_seismo_depth = 0.
lower_seismo_depth = 12.0

# Nodal plane distribution is a required attribute, but it will not influence here!
npd = PMF([(Decimal('1.0'), NodalPlane(0.,90., 0.))])
npd2 = PMF([(Decimal('0.5'), NodalPlane(0.,90., 0.)),
            (Decimal('0.5'), NodalPlane(90.0, 90.0, 0.0))])

# Define distribute hypocentral depth probabilities uniformly through upper 20 km of crust
hypos = np.arange(1., 11., 1.)
hdd = PMF([(Decimal('0.1'), hypo) for hypo in hypos])
    
src1 = AreaSource("001",
                  "Geysers 1",
                  "EGS",
                  TruncatedGRMFD(2.0, 6.0, 0.1, 4.4, 1.4),
                  1.0,
                  PointMSR(),
                  1.0,
                  PoissonTOM(1.0),
                  upper_seismo_depth,
                  lower_seismo_depth,
                  npd2,
                  hdd,
                  geometry1,
                  10)

src2 = AreaSource("002",
                  "Geysers 2",
                  "EGS",
                   TruncatedGRMFD(2.0, 6.0, 0.1, 4.8, 1.1),
                   1.0,
                   PointMSR(),
                   1.0,
                   PoissonTOM(1.0),
                   upper_seismo_depth,
                   lower_seismo_depth,
                   npd,
                   hdd,
                   geometry2,
                   1.0)
