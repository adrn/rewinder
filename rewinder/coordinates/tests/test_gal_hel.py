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
    """
    Note: slight offsets between Astropy / gary transformation and
    this transformation are expected because this assumes (l,b)=(0,0)
    is the Galactic center. Astropy uses a more modern measurement of
    the position of the GC.
    """
    np.random.seed(42)

    l = np.random.uniform(0.,360.,size=n)*u.degree
    b = np.random.uniform(-90.,90.,size=n)*u.degree
    d = np.random.uniform(0.,100.,size=n)*u.kpc
    mul_cosb = np.random.normal(0., 300., size=n)*u.km/u.s/d * np.cos(b)
    mub = np.random.normal(0., 300., size=n)*u.km/u.s/d
    vr = np.random.normal(0., 300., size=n)*u.km/u.s

    mul_cosb = mul_cosb.to(u.mas/u.yr, equivalencies=u.dimensionless_angles())
    mub = mub.to(u.mas/u.yr, equivalencies=u.dimensionless_angles())

    RSUN = 8.*u.kpc
    VCIRC = 240.*u.km/u.s
    VLSR = [0,0,0.] * u.km/u.s
    gc_frame = coord.Galactocentric(z_sun=0.*u.pc,
                                    galcen_distance=RSUN)

    c = coord.Galactic(l=l, b=b, distance=d)
    vxyz = gc.vhel_to_gal(c, (mul_cosb,mub), vr,
                          vcirc=VCIRC, vlsr=VLSR,
                          galactocentric_frame=gc_frame)
    xyz = c.transform_to(gc_frame).cartesian.xyz

    x,y,z = xyz.decompose(galactic).value
    vx,vy,vz = vxyz.decompose(galactic).value

    X = hel_to_gal(np.vstack((l.decompose(galactic).value,
                              b.decompose(galactic).value,
                              d.decompose(galactic).value,
                              mul_cosb.decompose(galactic).value,
                              mub.decompose(galactic).value,
                              vr.decompose(galactic).value)).T,
                   Vcirc=VCIRC.decompose(galactic).value,
                   Rsun=RSUN.decompose(galactic).value)
    x1,y1,z1,vx1,vy1,vz1 = X.T

    assert np.allclose(x1, x, rtol=1E-2)
    assert np.allclose(y1, y, rtol=1E-2)
    assert np.allclose(z1, z, rtol=1E-2)
    assert np.allclose(vx1, vx, rtol=1E-2)
    assert np.allclose(vy1, vy, rtol=1E-2)
    assert np.allclose(vz1, vz, rtol=1E-2)

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
