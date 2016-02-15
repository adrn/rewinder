""" Pure-python implementation of the rewinder likelihood function """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.coordinates as coord
from astropy import log as logger
import numpy as np
import gary.coordinates as gc
from gary.integrate import LeapfrogIntegrator
from gary.dynamics.core import combine
from scipy.stats import norm

# Project
from .rewinder import usys

# acceleration for integrator for the specified potential
def selfgravity_acceleration(t, w, potential, mass):
    this_acc = potential.acceleration(w[:3], t=t) # (3,2)
    dx = w[:3,1:] - w[:3,0:1]
    kep_acc = -potential.G*mass(t)*dx / np.linalg.norm(dx,axis=0)**3
    this_acc[:,1:] += kep_acc
    return np.vstack((w[3:], this_acc))

def progenitor_rot_matrix(prog_x, prog_v):
    R = np.zeros((3,3,prog_x.shape[1]))

    x1_hat = prog_x
    x3_hat = np.cross(x1_hat, prog_v)
    x2_hat = -np.cross(x1_hat, x3_hat)

    x1_hat /= np.linalg.norm(x1_hat, axis=0)
    x2_hat /= np.linalg.norm(x2_hat, axis=0)
    x3_hat /= np.linalg.norm(x3_hat, axis=0)

    R[0] = x1_hat
    R[1] = x2_hat
    R[2] = x3_hat

    return R

def rewinder_likelihood(dt, nsteps, potential, progen, stream, progen_mass,
                        K_mean, K_disp, betas, self_gravity=False):
    """
    Evaluate the Rewinder likelihood function.

    Parameters
    ----------
    dt : numeric
        Timestep for integration.
    nsteps : int
        Number of steps to integrate for.
    potential : :class:`gary.potential.PotentialBase`
        The gravitational potential of the host system.
    progen : :class:`gary.dynamics.CartesianPhaseSpacePosition`
        The present-day position and velocity of the progenitor in
        Galactocentric, cartesian coordinates.
    stream : :class:`gary.dynamics.CartesianPhaseSpacePosition`
        The present-day position and velocity of the progenitor in
        Galactocentric, cartesian coordinates.
    progen_mass : numeric, callable
        Either a single numeric value, or a function that evaluates
        the mass at a given time.
    K : array_like
        Matrix of scale parameters in the Fardal et al. (2015)
        parametrization of tidal stripping.
    betas : array_like
        Array of 1 or -1 for leading / trailing tail for each star.
    self_gravity : bool (optional)
        Turn on the selfgravity of the progenitor system.
    """

    # combine progenitor and stream star initial conditions
    w0 = combine((progen, stream))

    # default is Leapfrog
    Integrator = gi.LeapfrogIntegrator

    # a function to get mass of satellite at a given time
    try:
        progen_mass(0.)
        progen_mass_func = progen_mass
    except TypeError: # object not callable, constant mass
        progen_mass_func = lambda t: progen_mass

    # use different acceleration if accounting for self-gravity of the progenitor
    if self_gravity:
        integrator = Integrator(selfgravity_acceleration,
                                func_units=galactic,
                                func_args=(potential, progen_mass_func))
        orbits = integrator.run(w0, dt=dt, nsteps=nsteps)

    else:
        orbits = potential.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                                           Integrator=Integrator)

    # satellite mass
    progen_mass = progen_mass_func(t)

    # progenitor orbit
    progen = orbits[:,0]
    progen_x = progen.pos.decompose(usys).value
    progen_v = progen.vel.decompose(usys).value
    R = progenitor_rot_matrix(progen_x, progen_v)

    # stream stars
    stream = orbits[:,1:]
    stream_x = stream.pos.decompose(usys).value
    stream_v = stream.vel.decompose(usys).value

    dx = stream_x - progen_x
    dx = stream_v - progen_v

    rot_x = np.einsum('ijk,jk->ik', R, dx)
    rot_v = np.einsum('ijk,jk->ik', R, dv)

    # compute tidal radius estimate
    d = np.sqrt(np.sum(progen_x**2, axis=0))
    q = np.ascontiguousarray(progen_x.T)
    Om = np.linalg.norm(np.cross(progen_x, progen_v, axis=0) / d**2)
    f = Om*Om - potential.c_instance.d2_dr2(w, potential.G)
    r_tide = (potential.G*progen_mass / f)**(1/3.)
    v_scale = np.sqrt(potential.G * progen_mass / r_tide)

    ll = 0.

    # beta=-1 is trailing tail, so need to negate
    ll += norm.logpdf(dx[0], loc=-betas*K_mean[0]*r_tide, scale=K_disp[0]*r_tide)
    ll += norm.logpdf(dx[1], loc=K_mean[1]*r_tide, scale=K_disp[1]*r_tide)
    ll += norm.logpdf(dx[2], loc=K_mean[2]*r_tide, scale=K_disp[2]*r_tide)

    ll += norm.logpdf(dv[0], loc=K_mean[3]*v_scale, scale=K_disp[3]*v_scale)
    ll += norm.logpdf(dv[1], loc=K_mean[4]*v_scale, scale=K_disp[4]*v_scale)
    ll += norm.logpdf(dv[2], loc=K_mean[5]*v_scale, scale=K_disp[5]*v_scale)









    # compute an estimate of the jacobian
    Rsun = 8.
    R2 = (w[:,1:,0] + Rsun)**2 + w[:,1:,1]**2 + w[:,1:,2]**2
    x2 = w[:,1:,2]**2 / R2
    log_jac = np.log(R2*R2 * np.sqrt(1.-x2))

    return g + log_jac, w, ixes
    # return g, w, ixes
