#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
from .estimate import estimate
from .gradient import dzdc

__all__ = (
  'update_calibration',
)


def update_calibration(obs, ref, src, exp, cal):
    ''' Updates of the calibration parameters

    Arguments:
        obs: measurements
        ref: reference catalog
        src: source parameters
        exp: exposure parameters
        cal: calibration parameters

    Returns:
      The updated calibration parameters.
    '''
    c = estimate(src, exp, cal)[:, 2]
    o = obs[:, 2]
    s = obs[:, 3]
    S = (1 / s**2).reshape(-1, 1)

    Dc = dzdc(src[:, 1:], exp[:, 2:], cal, exp[:, 1])

    N = Dc.T @ (S * Dc)
    b = Dc.T @ ((o - c) / s**2)

    cfac = jax.scipy.linalg.cho_factor(N)
    delta = jax.scipy.linalg.cho_solve(cfac, b)
    return cal + delta
