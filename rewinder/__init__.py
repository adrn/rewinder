"""
Use stellar tidal streams to infer the gravitational potential
around the Milky Way.
"""

from ._astropy_init import *

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    # HACK: this should be a configuration setting
    from gary.units import galactic
    usys = galactic
    del galactic
