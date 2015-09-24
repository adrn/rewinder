# coding: utf-8

""" Fast coordinate transformation from Galactocentric to Heliocentric """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import gary.coordinates as gc
from gary.units import galactic

# Project
from .. import gal_to_hel, hel_to_gal

n = 100

def test_hel_gal():
    np.random.seed(42)

    l = np.random.uniform(0.,360.,size=n)*u.degree
    b = np.random.uniform(-90.,90.,size=n)*u.degree
    d = np.random.uniform(0.,100.,size=n)*u.kpc
    mul = np.random.normal(0., 300., size=n)*u.km/u.s/d
    mub = np.random.normal(0., 300., size=n)*u.km/u.s/d
    vr = np.random.normal(0., 300., size=n)*u.km/u.s

    mul = mul.to(u.mas/u.yr,equivalencies=u.dimensionless_angles())
    mub = mub.to(u.mas/u.yr,equivalencies=u.dimensionless_angles())

    VCIRC = 240.*u.km/u.s
    VLSR = [9.,10.,11.] * u.km/u.s

    c = coord.Galactic(l=l, b=b, distance=d)
    vxyz = gc.vhel_to_gal(c, (mul,mub), vr,
                          vcirc=VCIRC, vlsr=VLSR)

    xyz = c.transform_to(coord.Galactocentric).cartesian.xyz

    x,y,z = xyz.decompose(galactic).value
    vx,vy,vz = vxyz.decompose(galactic).value

    X = hel_to_gal(np.vstack((l.decompose(galactic).value,
                              b.decompose(galactic).value,
                              d.decompose(galactic).value,
                              mul.decompose(galactic).value,
                              mub.decompose(galactic).value,
                              vr.decompose(galactic).value)).T)
    x1,y1,z1,vx1,vy1,vz1 = X.T

    assert np.allclose(x1, x)
    assert np.allclose(y1, y)
    assert np.allclose(z1, z)
    assert np.allclose(vx1, vx)
    assert np.allclose(vy1, vy)
    assert np.allclose(vz1, vz)

# def test_gal_hel():
#     np.random.seed(42)

#     xyz = np.random.uniform(-100.,100.,size=(3,n))*u.kpc
#     vxyz = np.random.normal(0.,300.,size=(3,n))*u.km/u.s

#     lbd,(mul,mub,vr) = sc.gal_xyz_to_hel(xyz, vxyz, vlsr=[0.,0.,0.]*u.km/u.s)
#     l = lbd.l.decompose(usys).value
#     b = lbd.b.decompose(usys).value
#     d = lbd.distance.decompose(usys).value
#     mul = mul.decompose(usys).value
#     mub = mub.decompose(usys).value
#     vr = vr.decompose(usys).value

#     X = gal_to_hel(np.vstack((xyz.decompose(usys).value,
#                               vxyz.decompose(usys).value)).T)
#     l1,b1,d1,mul1,mub1,vr1 = X.T

#     assert np.allclose(l1, l)
#     assert np.allclose(b1, b)
#     assert np.allclose(d1, d)
#     assert np.allclose(mul1, mul)
#     assert np.allclose(mub1, mub)
#     assert np.allclose(vr1, vr)
