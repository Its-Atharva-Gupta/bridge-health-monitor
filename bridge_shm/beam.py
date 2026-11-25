"""Beam physics for the simulator: a simply supported plank whose stiffness (EI)
can be reduced in "damage zones".

static()  - sag, slope (what the MPUs tilt by) and bending moment under a point load
modes()   - first two vibration modes (frequency ratios + mode shapes), finite elements

Why physics instead of guessed multipliers: every sensor then responds to the
SAME damage consistently (e.g. a strain gauge only notices damage right under it,
while the MPU tilts notice damage anywhere), so comparing sensor setups is fair.
"""
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh

from . import config as C


def ei_factor(x, zones):
    """Stiffness multiplier along the beam: 1 = healthy, 0.7 = 30% weaker."""
    f = np.ones_like(np.asarray(x, float))
    for x0, x1, fac in zones:
        f[(x >= x0) & (x < x1)] *= fac
    return f


def static(F, x_load, zones=(), EI=1.0, L=C.SPAN_MM, n=2701):
    """Point load F at x_load. Returns x, deflection (down = +), slope, moment."""
    x = np.linspace(0.0, L, n)
    M = np.where(x <= x_load, F * (L - x_load) / L * x, F * x_load / L * (L - x))
    kappa = M / (EI * ei_factor(x, zones))
    s = cumulative_trapezoid(kappa, x, initial=0.0)
    S = cumulative_trapezoid(s, x, initial=0.0)
    c = S[-1] / L                       # makes deflection zero at both supports
    return x, -S + c * x, -s + c, M


@lru_cache(maxsize=64)
def modes(zones=(), n_el=108, sensor_mass_ratio=0.15, L=C.SPAN_MM):
    """First two bending modes. Returns (omega1, omega2, node_x, shape1, shape2).

    Uniform plank (mass 1 per length, EI 1 unless reduced) plus the two MPU boards
    as point masses (each `sensor_mass_ratio` x plank mass). Only ratios matter:
    the simulator scales them to the healthy frequency.
    """
    le = L / n_el
    nd = 2 * (n_el + 1)
    K = np.zeros((nd, nd))
    Mm = np.zeros((nd, nd))
    xc = (np.arange(n_el) + 0.5) * le
    f = ei_factor(xc, zones)
    l = le
    k0 = np.array([[12, 6 * l, -12, 6 * l], [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                   [-12, -6 * l, 12, -6 * l], [6 * l, 2 * l * l, -6 * l, 4 * l * l]]) / l ** 3
    m0 = l / 420 * np.array([[156, 22 * l, 54, -13 * l], [22 * l, 4 * l * l, 13 * l, -3 * l * l],
                             [54, 13 * l, 156, -22 * l], [-13 * l, -3 * l * l, -22 * l, 4 * l * l]])
    for e in range(n_el):
        i = slice(2 * e, 2 * e + 4)
        K[i, i] += f[e] * k0
        Mm[i, i] += m0
    for xs in (C.MPU_X_MM, L - C.MPU_X_MM):
        Mm[2 * int(round(xs / le)), 2 * int(round(xs / le))] += sensor_mass_ratio * L
    keep = [d for d in range(nd) if d not in (0, 2 * n_el)]       # w = 0 at both supports
    w2, vec = eigh(K[np.ix_(keep, keep)], Mm[np.ix_(keep, keep)])
    full = np.zeros((nd, 2))
    full[keep] = vec[:, :2]
    nodes = np.arange(n_el + 1) * le
    shapes = []
    for k in range(2):
        w = full[0::2, k]
        w = w / w[np.argmax(np.abs(w))]
        if k == 1 and np.interp(C.MPU_X_MM, nodes, w) < 0:        # sign: left half up = +
            w = -w
        shapes.append(w)
    return float(np.sqrt(w2[0])), float(np.sqrt(w2[1])), nodes, shapes[0], shapes[1]


def shape_at(nodes, shape, x):
    return float(np.interp(x, nodes, shape))
