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

n = 128

def test_roundtrip():
    np.random.seed(42)

    X = np.random.normal(size=(n,6))
    O = gal_to_hel(X)
    X_trans = hel_to_gal(O)
    np.testing.assert_allclose(X, X_trans)

    O = np.random.normal(size=(n,6))
    O[:,0] = np.random.uniform(0, 2*np.pi, size=n)
    O[:,1] = np.random.uniform(-np.pi/2, np.pi/2, size=n)
    O[:,2] = np.random.uniform(0, 100, size=n)
    X = hel_to_gal(O)
    O_trans = gal_to_hel(X)
    np.testing.assert_allclose(O, O_trans)

def test_hel_gal():
    """
    Note: slight offsets between Astropy / gary transformation and
    this transformation are expected because this assumes (l,b)=(0,0)
    is the Galactic center. Astropy uses a more modern measurement of
    the position of the GC.
    """
    np.random.seed(42)

    RSUN = 8.*u.kpc
    VCIRC = 240.*u.km/u.s
    VLSR = [0,0,0.] * u.km/u.s
    gc_frame = coord.Galactocentric(z_sun=0.*u.pc,
                                    galcen_distance=RSUN)

    l = np.random.uniform(0.,360.,size=n)*u.degree
    b = np.random.uniform(-90.,90.,size=n)*u.degree
    d = np.random.uniform(0.,100.,size=n)*u.kpc
    mul_cosb = np.random.normal(0., 300., size=n)*u.km/u.s/d * np.cos(b)
    mub = np.random.normal(0., 300., size=n)*u.km/u.s/d
    vr = np.random.normal(0., 300., size=n)*u.km/u.s

    mul_cosb = mul_cosb.to(u.mas/u.yr, equivalencies=u.dimensionless_angles())
    mub = mub.to(u.mas/u.yr, equivalencies=u.dimensionless_angles())

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

def test_gal_hel():
    """
    Note: slight offsets between Astropy / gary transformation and
    this transformation are expected because this assumes (l,b)=(0,0)
    is the Galactic center. Astropy uses a more modern measurement of
    the position of the GC.
    """
    np.random.seed(42)

    RSUN = 8.*u.kpc
    VCIRC = 240.*u.km/u.s
    VLSR = [0,0,0.] * u.km/u.s

    xyz = np.random.uniform(-10.,10.,size=(3,n))*u.kpc
    vxyz = np.random.normal(0.,100.,size=(3,n))*u.km/u.s

    c = coord.Galactocentric(coord.CartesianRepresentation(xyz),
                             z_sun=0.*u.pc,
                             galcen_distance=RSUN)

    gal_c = c.transform_to(coord.Galactic)

    mul_cosb,mub,vr = gc.vgal_to_hel(gal_c, vxyz, vcirc=VCIRC, vlsr=VLSR,
                                     galactocentric_frame=c)

    l = gal_c.l.decompose(galactic).value
    b = gal_c.b.decompose(galactic).value
    d = gal_c.distance.decompose(galactic).value
    mul_cosb = mul_cosb.decompose(galactic).value
    mub = mub.decompose(galactic).value
    vr = vr.decompose(galactic).value

    X = gal_to_hel(np.vstack((xyz.decompose(galactic).value,
                              vxyz.decompose(galactic).value)).T,
                   Vcirc=VCIRC.decompose(galactic).value,
                   Rsun=RSUN.decompose(galactic).value)
    l1,b1,d1,mul_cosb1,mub1,vr1 = X.T

    assert np.allclose(l1, l, rtol=1E-2)
    assert np.allclose(b1, b, rtol=1E-2)
    assert np.allclose(d1, d, rtol=1E-2)
    assert np.allclose(mul_cosb1, mul_cosb, rtol=1E-2)
    assert np.allclose(mub1, mub, rtol=1E-2)
    assert np.allclose(vr1, vr, rtol=1E-2)
