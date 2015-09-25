# encoding: utf-8

""" Fast transformation from heliocentric (Galactic) coordinates to
    galacocentric cartesian coordinates.
"""

from __future__ import division

import sys
import numpy as np
cimport numpy as np

import cython
cimport cython

from libc.math cimport M_PI

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double abs(double x)

cdef TWOPI = 2*M_PI

def gal_to_hel(np.ndarray[double, ndim=2] X,
               double Rsun=8.,
               double Vcirc=(220*0.0010227121650537077)): # km/s to kpc/Myr
    """ Assumes Galactic units: kpc, Myr, radian, M_sun """

    cdef int ii
    cdef int nparticles = X.shape[0]
    cdef double[:,:] O = np.empty((nparticles, 6))

    cdef double x, y, z, vx, vy, vz, d_xy
    cdef double l,b,d,mul_cosb,mub,vr

    for ii in range(nparticles):
        # transform to heliocentric cartesian
        x = X[ii,0] + Rsun
        y = X[ii,1]
        z = X[ii,2]
        vx = X[ii,3]
        vy = X[ii,4] - Vcirc
        vz = X[ii,5]

        # transform from cartesian to spherical
        d = sqrt(x*x + y*y + z*z)
        d_xy = sqrt(x*x + y*y)
        l = atan2(y, x)
        b = atan2(z, d_xy)

        if l < 0:
            l = l+TWOPI

        # transform cartesian velocity to spherical
        vr = (vx*x + vy*y + vz*z) / d # kpc/Myr
        mul_cosb = (x*vy-vx*y) / (d_xy*d) # rad / Myr
        mub = -(z*(x*vx + y*vy) - d_xy*d_xy*vz) / (d*d * d_xy) # rad / Myr

        O[ii,0] = l
        O[ii,1] = b
        O[ii,2] = d
        O[ii,3] = mul_cosb
        O[ii,4] = mub
        O[ii,5] = vr

    return np.array(O)

def hel_to_gal(np.ndarray[double, ndim=2] O,
               double Rsun=8.,
               double Vcirc=(220*0.0010227121650537077)): # km/s to kpc/Myr
    """ Assumes Galactic units: kpc, Myr, radian, M_sun """

    cdef int ii
    cdef int nparticles = O.shape[0]
    cdef double[:,:] X = np.empty((nparticles, 6))

    cdef double l,b,d,mul_cosb,mub,vr
    cdef double x, y, z, vx, vy, vz
    cdef double cosl, sinl, cosb, sinb

    for ii in range(nparticles):
        l = O[ii,0]
        b = O[ii,1]
        d = O[ii,2]
        mul_cosb = O[ii,3]
        mub = O[ii,4]
        vr = O[ii,5]

        cosb = cos(b)
        sinb = sin(b)
        cosl = cos(l)
        sinl = sin(l)

        # transform from spherical to cartesian
        x = d*cosl*cosb
        y = d*sinl*cosb
        z = d*sinb

        # transform spherical velocity to cartesian
        # mul_cosb = -mul_cosb
        # mub = -mub

        # vx = x/d*vr + y*mul + z*cos(l)*mub
        # vy = y/d*vr - x*mul + z*sin(l)*mub
        # vz = z/d*vr - d*cos(b)*mub
        vx = vr*cosl*cosb - d*sinl*mul_cosb - d*cosl*sinb*mub
        vy = vr*sinl*cosb + d*cosl*mul_cosb - d*sinl*sinb*mub
        vz = vr*sinb + d*cosb*mub

        x = x - Rsun
        vy = vy + Vcirc

        X[ii,0] = x
        X[ii,1] = y
        X[ii,2] = z
        X[ii,3] = vx
        X[ii,4] = vy
        X[ii,5] = vz

    return np.array(X)
