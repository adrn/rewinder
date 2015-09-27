# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
from gary.inference import ModelParameter

__all__ = ['RewinderPotential']

class RewinderPotential(object):
    """
    A thin wrapper around gary potential objects to support
    statistical parameters.

    Parameters
    ----------
    Potential : `gary.potential.PotentialBase`
        The potential class.
    units : `gary.units.UnitSystem`
        The unit system to use.
    priors : dict
        A dictionary of prior objects for each potential parameter.
    frozen : dict
        If any of the parameters should be frozen (held fixed),
        specify the value to fix them to.
    """

    def __init__(self, Potential, units, priors=dict(), frozen=dict()):

        self.Potential = Potential
        self.units = units
        self.frozen = frozen
        self.priors = priors

        self.parameters = OrderedDict()
        for name,prior in priors.items():
            self.parameters[name] = ModelParameter(name=name, prior=prior)

    def get_obj(self, **parameter_values):
        """
        Given values for the potential parameters being varied, return
        a potential object with the given parameters plus any other
        frozen parameters.
        """
        pars = dict(parameter_values.items() + self.frozen.items())
        return self.Potential(units=self.units, **pars)

    def ln_prior(self, **kwargs):
        """ Evaluate the value of the log-prior over the potential parameters. """

        lp = 0.
        for k,prior in self.priors.items():
            lp += prior.logpdf(kwargs[k])

        return lp
