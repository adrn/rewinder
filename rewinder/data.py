# coding: utf-8

""" Class for representing observed dynamical object """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from scipy.stats import norm
from gary.inference import NormalPrior

__all__ = ['RewinderData']

_names = ['l','b','d','mul','mub','vr']

class RewinderData(object):
    """
    Position and velocity data. Pass in infinite errors/uncertainties for a
    given coordinate if there is no data.

    Parameters
    ----------
    lbd : iterable
        Galactic coordinates and distance.
    pmv : iterable
        Proper motions and radial velocity.
    units : `gary.units.UnitSystem`
        The unit system to use.
    lbd_err : iterable
    pmv_err : iterable
    frozen : list (optional)
        A list of coordinates to freeze. In this context, this means coordinates
        to exclude from the prior and data likelihood computation.

    """

    def __init__(self, lbd, pmv, units,
                 lbd_err=None, pmv_err=None, frozen=list()):

        l,b,d = lbd
        mul,mub,vr = pmv

        # validate input quantities
        shape = None
        coords = [l,b,d,mul,mub,vr]
        for c in coords:
            if not hasattr(c, 'unit'):
                raise TypeError("Coordinate and velocity information must be Astropy"
                                "Quantity objects.")

            if shape is None:
                shape = c.shape
            else:
                if c.shape != shape:
                    raise ValueError("All input coordinates must have same shape.")

        # validate frozen parameter names
        for name in frozen:
            if name not in _names:
                raise ValueError("Parameter name '{}' is not a valid coordinate. Valid"
                                 " names are: {}".format(name,','.join(_names)))

        # validate input uncertainties
        if lbd_err is None:
            lbd_err = [None, None, None]
        l_err,b_err,d_err = lbd_err

        if pmv_err is None:
            pmv_err = [None, None, None]
        mul_err,mub_err,vr_err = lbd_err

        errs = [l_err,b_err,d_err,mul_err,mub_err,vr_err]
        for name,err in zip(_names, errs):
            if name in frozen:
                continue

            if err is None:
                raise ValueError("Uncertainty is None for '{}' but that parameter"
                                 " is not frozen.".format(name))

            if not hasattr(err, 'unit'):
                raise TypeError("Coordinate and velocity uncertainties must be Astropy"
                                "Quantity objects.")

        # containers
        n = l.shape[0]
        self.data = np.ones((n,6))*np.nan
        self.err = np.ones((n,6))*np.nan

        # TODO: right now this makes the assumption that each coordinate has Gaussian,
        #       uncorrelated uncertainties?
        for i,name,c,err in zip(range(6),_names,coords,errs):
            self.data[:,i] = c.decompose(units).value
            if err is None:
                self.err[:,i] = np.zeros(n)
            else:
                self.err[:,i] = err.decompose(units).value

        # priors on coordinates
        default_priors = np.zeros_like(self.data)
        default_priors[:,0] = np.zeros(n) - np.log(2*np.pi)
        default_priors[:,1] = np.log(np.cos(self.data[:,1])/2)

        # distance range: 1-100 kpc
        default_priors[:,2] = -np.log(np.log(100.)) - np.log(self.data[:,2])

        p = NormalPrior(0., 0.306814 / self.data[:,2])  # 300 km/s at d
        default_priors[:,3] = p.logpdf(self.data[:,3])
        default_priors[:,4] = p.logpdf(self.data[:,4])

        p = NormalPrior(0., 0.306814)  # 300 km/s
        default_priors[:,5] = p.logpdf(self.data[:,5])

        self.default_priors = default_priors

    def log_data_prob(self, x):
        """ Compute the (log-)probability of phase-space positions (in heliocentric
            observed coordinates), given the data and uncertainties.

        """
        _dist = norm(self.data, self.err)
        lp = _dist.logpdf(x)
        for i in range(6):
            lp[np.isnan(lp[:,i]),i] = self.default_priors[np.isnan(lp[:,i]),i]

        return lp.sum(axis=1)
