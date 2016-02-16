"""
    Draw the probabilistic graphical model (PGM) for Rewinder.

    Run this from the top-level of the project (e.g., above scripts/)
"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as pl

try:
    import daft
except ImportError:
    raise ImportError("Daft is required to generate a PGM image. You can "
                      "install with 'pip install daft'")

plot_path = 'plots'
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

def main():

    pl.rcParams['text.usetex'] = True

    pgm = daft.PGM([6, 6], origin=[-4, -3])

    star_plate = daft.Plate([-1.1, -2.25, 2.2, 2.2],
                            label=r"$i=1...N_s$")
    pgm.add_plate(star_plate)

    prog_plate = daft.Plate([-1.2, -2.75, 2.4, 4.7],
                            label=r"$j=1...N_p$")
    pgm.add_plate(prog_plate)

    y_off = star_plate.rect[1] / 2.
    pgm.add_node(daft.Node("star data helio", r"$\boldsymbol{D}_i$",
                           -0.5, -0.5+y_off, observed=True))
    pgm.add_node(daft.Node("star data error", r"$\boldsymbol{\sigma}_i$",
                           0.5, -0.5+y_off, observed=True))
    pgm.add_node(daft.Node("unbinding time", r"$\tau_i$",
                           0.5, 0.5+y_off))
    w_i = daft.Node("star data galacto", r"$\boldsymbol{W}_i$", -0.5, 0.5+y_off)
    pgm.add_node(w_i)

    pgm.add_edge("star data galacto", "star data helio")
    pgm.add_edge("star data error", "star data helio")
    pgm.add_edge("unbinding time", "star data galacto")

    pgm.add_node(daft.Node("prog data helio", r"$\boldsymbol{D}_j$",
                           -0.5, w_i.y+2., observed=True))
    pgm.add_node(daft.Node("prog data error", r"$\boldsymbol{\sigma}_j$",
                           0.5, w_i.y+2., observed=True))
    pgm.add_node(daft.Node("prog data galacto", r"$\boldsymbol{W}_j$",
                           -0.5, w_i.y+1.))
    pgm.add_node(daft.Node("prog ev time", r"$T_j$",
                           0.5, w_i.y+1.))
    pgm.add_edge("prog data galacto", "star data galacto")
    pgm.add_edge("prog data error", "prog data helio")
    pgm.add_edge("prog data galacto", "prog data helio")
    pgm.add_edge("prog ev time", "star data galacto")

    pgm.add_node(daft.Node("sun true", r"$\boldsymbol{W}_\odot$",
                           -1.9, w_i.y+2.))
    pgm.add_node(daft.Node("sun data", r"$\boldsymbol{G}_\odot$",
                           -2.9, w_i.y+2., observed=True))
    pgm.add_node(daft.Node("sun err", r"$\boldsymbol{\sigma}_\odot$",
                           -2.9, w_i.y+3, observed=True))
    pgm.add_edge("sun true", "prog data helio")
    pgm.add_edge("sun true", "star data helio")
    pgm.add_edge("sun true", "sun data")
    pgm.add_edge("sun err", "sun data")

    pgm.add_node(daft.Node("potential", r"$\boldsymbol{\Phi}$",
                           -1.9, y_off+1.1))
    pgm.add_node(daft.Node("K", r"$\boldsymbol{K}$",
                           -1.9, y_off+0.5))
    pgm.add_node(daft.Node("S", r"$\boldsymbol{\rm \Sigma}$",
                           -1.9, y_off-0.1))
    pgm.add_edge("potential", "star data galacto")
    pgm.add_edge("K", "star data galacto")
    pgm.add_edge("S", "star data galacto")

    pgm.render()
    pgm.figure.savefig(os.path.join(plot_path, "weaklensing.pdf"))
    pgm.figure.savefig(os.path.join(plot_path, "weaklensing.png"), dpi=150)

if __name__ == "__main__":
    main()
