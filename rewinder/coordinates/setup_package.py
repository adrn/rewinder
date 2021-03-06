# Licensed under a 3-clause BSD style license - see PYFITS.rst
from __future__ import absolute_import

from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['sources'].append('rewinder/coordinates/gal_hel.pyx')

    return [Extension('rewinder.coordinates._gal_hel', **cfg)]

# def get_package_data():
#     return {'biff': ['src/*.h', 'data/*.dat.gz', 'data/*.coeff']}
