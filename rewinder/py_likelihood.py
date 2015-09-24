# coding: utf-8

""" Pure-python implementation of my likelihood function """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np
from astropy import log as logger

def get_basis(prog_w, theta=0.):
    """ Compute the instantaneous orbital plane basis at each timestep for the
        progenitor system orbit.
    """
    basis = np.zeros((len(prog_w),3,3))

    prog_x = prog_w[...,:3].copy()
    prog_v = prog_w[...,3:].copy()

    x1_hat = prog_x
    x3_hat = np.cross(x1_hat, prog_v)
    x2_hat = -np.cross(x1_hat, x3_hat)

    x1_hat /= np.linalg.norm(x1_hat, axis=-1)[...,np.newaxis]
    x2_hat /= np.linalg.norm(x2_hat, axis=-1)[...,np.newaxis]
    x3_hat /= np.linalg.norm(x3_hat, axis=-1)[...,np.newaxis]

    if theta != 0.:
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        R = np.array([[costheta, sintheta,0],[-sintheta, costheta,0],[0,0,1]])

        x1_hat = x1_hat.dot(R)
        x2_hat = x2_hat.dot(R)

    basis[...,0] = x1_hat
    basis[...,1] = x2_hat
    basis[...,2] = x3_hat
    return basis

def rewinder_likelihood(dt, nsteps, potential, prog_xv, star_xv, m0, mdot,
                        alpha, betas, theta, selfgravity=False):

    # full array of initial conditions for progenitor and stars
    w0 = np.vstack((prog_xv,star_xv))

    # integrate orbits
    t,w = potential.integrate_orbit(w0.copy(), dt=dt, nsteps=nsteps)
    dw = w[:,1:] - w[:,0:1]

    t = t[:-1]
    w = w[:-1]

    # satellite mass
    sat_mass = -mdot*t + m0
    # GMprog = potential.G * sat_mass

    # compute approximations of tidal radius and velocity dispersion from mass enclosed
    menc = potential.mass_enclosed(w[:,0,:3].copy())  # progenitor position orbit
    f = 1.
    E_scale = f * (sat_mass / menc)**(1/3.)

    # compute naive tidal radius and velocity dispersion
    rtide = E_scale * np.linalg.norm(w[:,0,:3], axis=-1)  # progenitor orbital radius
    vdisp = E_scale * np.linalg.norm(w[:,0,3:], axis=-1)  # progenitor orbital velocity

    # write like this to allow for more general dispersions...probably want a covariance matrix
    sigmas = np.zeros((nsteps,1,6))
    sigmas[:,0,0] = rtide
    sigmas[:,0,1] = 2*rtide
    sigmas[:,0,2] = rtide

    sigmas[:,0,3] = 2*vdisp
    sigmas[:,0,4] = vdisp
    sigmas[:,0,5] = vdisp

    # get the instantaneous orbital plane basis vectors (x1,x2,x3)
    basis = get_basis(w[:,0], theta)

    # star orbits relative to progenitor
    dw = w[:,1:] - w[:,0:1]

    # project orbits into new basis
    w123 = np.zeros_like(dw)
    for i in range(3):
        w123[...,i] = np.sum(dw[...,:3] * basis[...,i][:,np.newaxis], axis=-1)
        w123[...,i+3] = np.sum(dw[...,3:] * basis[...,i][:,np.newaxis], axis=-1)

    w123[...,0] += alpha*betas[np.newaxis]*rtide[:,np.newaxis]

    ixes = np.sum((w123/sigmas)**2,axis=-1).argmin(axis=0)

    # # not in new basis
    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,2,figsize=(16,8))

    # axes[0].plot(dw[...,0], dw[...,1], marker='.', alpha=0.01, color='b', linestyle='none')
    # axes[1].plot(dw[...,3], dw[...,4], marker='.', alpha=0.01, color='b', linestyle='none')

    # ix = 4
    # ixes = np.sqrt(np.sum((w123[:,ix]/sigmas[:,0])**2,axis=-1)).argmin()
    # axes[0].plot(dw[:ix2,ix,0], dw[:ix2,ix,1], marker='.', alpha=0.75, color='k', linestyle='none')
    # axes[1].plot(dw[:ix2,ix,3], dw[:ix2,ix,4], marker='.', alpha=0.75, color='k', linestyle='none')

    # axes[0].set_xlim(-0.5,0.5)
    # axes[1].set_xlim(-0.005,0.005)
    # axes[0].set_ylim(-0.5,0.5)
    # axes[1].set_ylim(-0.005,0.005)

    # in basis
    # import matplotlib.pyplot as plt
    # fig,axes = plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True)

    # axes[0].plot(w123[:,:,0]/sigmas[:,:,0], w123[:,:,1]/sigmas[:,:,1],
    #              marker='.', alpha=0.1, color='k', linestyle='none')
    # axes[1].plot(w123[:,:,3]/sigmas[:,:,3], w123[:,:,4]/sigmas[:,:,4],
    #              marker='.', alpha=0.1, color='k', linestyle='none')

    # axes[0].set_xlim(-2,2)
    # axes[0].set_ylim(-2,2)
    # axes[0].set_xlabel("$x_1$", fontsize=24)
    # axes[0].set_ylabel("$x_2$", fontsize=24)

    # axes[1].set_xlabel("$v_{x_1}$", fontsize=24)
    # axes[1].set_ylabel("$v_{x_2}$", fontsize=24)
    # plt.show()
    # sys.exit(0)

    # sigmoid business
    # A = 0.99
    # l1 = 25.*sigmas[...,0]
    # x1 = 1/l1 * np.log((1-A)/A)
    # l2 = 25.*sigmas[...,1]
    # x2 = 1/l2 * np.log((1-A)/A)
    # S1 = -np.log(1 + np.exp(betas[None]*-l1*(w123[...,0]-x1)))
    # S2 = -np.log(1 + np.exp(betas[None]*-l2*(w123[...,1]-x2)))

    # gaussian likelihood
    g = -0.5*np.log(2*np.pi) - np.log(sigmas) - 0.5*(w123/sigmas)**2
    g = g.sum(axis=-1)
    # g += S1 + S2

    # # plot each orbit on separate panel
    # import matplotlib.pyplot as plt
    # for i in range(len(ixes)):
    #     fig,all_axes = plt.subplots(2,4,figsize=(20,12))

    #     i1,i2 = ixes[i]-int(500./abs(dt)), ixes[i]+int(500./abs(dt))
    #     if i1 < 0:
    #         i1 = 0.
    #     if i2 > (len(t)-1):
    #         i2 = len(t)-1

    #     c = t[i1:i2]
    #     # c = g[i1:i2,i]

    #     axes = all_axes[0]
    #     axes[0].scatter(dw[i1:i2,i,0], dw[i1:i2,i,1], marker='.', alpha=0.5, c=c)
    #     axes[1].scatter(dw[i1:i2,i,3], dw[i1:i2,i,4], marker='.', alpha=0.5, c=c)
    #     axes[2].scatter(dw[i1:i2,i,0], dw[i1:i2,i,2], marker='.', alpha=0.5, c=c)
    #     axes[3].scatter(dw[i1:i2,i,3], dw[i1:i2,i,5], marker='.', alpha=0.5, c=c)

    #     axes[0].set_xlim(-1,1)
    #     axes[0].set_ylim(-1,1)
    #     axes[2].set_xlim(-1,1)
    #     axes[2].set_ylim(-1,1)
    #     axes[1].set_xlim(-0.01,0.01)
    #     axes[1].set_ylim(-0.01,0.01)
    #     axes[3].set_xlim(-0.01,0.01)
    #     axes[3].set_ylim(-0.01,0.01)

    #     axes = all_axes[1]
    #     cb = axes[0].scatter(w123[i1:i2,i,0]/sigmas[i1:i2,0,0], w123[i1:i2,i,1]/sigmas[i1:i2,0,1],
    #                          marker='.', alpha=0.5, c=c)
    #     axes[1].scatter(w123[i1:i2,i,3]/sigmas[i1:i2,0,3], w123[i1:i2,i,4]/sigmas[i1:i2,0,4],
    #                     marker='.', alpha=0.5, c=c)
    #     axes[2].scatter(w123[i1:i2,i,0]/sigmas[i1:i2,0,0], w123[i1:i2,i,2]/sigmas[i1:i2,0,2],
    #                     marker='.', alpha=0.5, c=c)
    #     axes[3].scatter(w123[i1:i2,i,3]/sigmas[i1:i2,0,3], w123[i1:i2,i,5]/sigmas[i1:i2,0,5],
    #                     marker='.', alpha=0.5, c=c)

    #     cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    #     fig.colorbar(cb, cax=cbar_ax)

    #     for ax in axes:
    #         lim = 3.
    #         ax.set_xlim(-lim,lim)
    #         ax.set_ylim(-lim,lim)
    #     axes[0].set_xlabel("$x_1$", fontsize=24)
    #     axes[0].set_ylabel("$x_2$", fontsize=24)

    #     axes[1].set_xlabel("$v_{x_1}$", fontsize=24)
    #     axes[1].set_ylabel("$v_{x_2}$", fontsize=24)

    #     axes[2].set_xlabel("$x_1$", fontsize=24)
    #     axes[2].set_ylabel("$x_3$", fontsize=24)

    #     axes[3].set_xlabel("$v_{x_1}$", fontsize=24)
    #     axes[3].set_ylabel("$v_{x_3}$", fontsize=24)

    #     fig.tight_layout()

    #     fig.savefig("/Users/adrian/Downloads/{}.png".format(i))

    #     plt.close('all')

    # # plt.show()
    # sys.exit(0)

    # # plot the god damn orbit in real space
    # import matplotlib.pyplot as plt
    # for i in [0,1]:
    #     fig,axes = plt.subplots(1,3,figsize=(16,6))
    #     axes[0].plot(w[:,0,0], w[:,0,1], marker=None, c='k')
    #     axes[0].plot(w[0,0,0], w[0,0,1], marker='o', c='k')
    #     axes[0].plot(w[:,i+1,0], w[:,i+1,1], marker=None, c='b')
    #     axes[0].plot(w[0,i+1,0], w[0,i+1,1], marker='o', c='b')

    #     axes[1].plot(w[:,0,0], w[:,0,2], marker=None, c='k')
    #     axes[1].plot(w[0,0,0], w[0,0,2], marker='o',c='k')
    #     axes[1].plot(w[:,i+1,0], w[:,i+1,2], marker=None, c='b')
    #     axes[1].plot(w[0,i+1,0], w[0,i+1,2], marker='o',c='b')

    #     axes[2].plot(w[:,0,1], w[:,0,2], marker=None, c='k')
    #     axes[2].plot(w[0,0,1], w[0,0,2], marker='o',c='k')
    #     axes[2].plot(w[:,i+1,1], w[:,i+1,2], marker=None, c='b')
    #     axes[2].plot(w[0,i+1,1], w[0,i+1,2], marker='o',c='b')

    #     fig.savefig("/Users/adrian/Downloads/wtf/{}/{}_{}.png".format(i, i, potential['halo'].parameters['M']))

    # compute an estimate of the jacobian
    Rsun = 8.
    R2 = (w[:,1:,0] + Rsun)**2 + w[:,1:,1]**2 + w[:,1:,2]**2
    x2 = w[:,1:,2]**2 / R2
    log_jac = np.log(R2*R2 * np.sqrt(1.-x2))

    return g + log_jac, w, ixes
    # return g, w, ixes
