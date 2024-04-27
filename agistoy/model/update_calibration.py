#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from .estimate import estimate
from .gradient import dzdc

__all__ = (
  'update_calibration',
)


def update_calibration_inner(obs, ref, src, exp, _cal):
    c = estimate(src, exp, _cal)[:, 2]
    o = obs[:, 2]
    s = obs[:, 3]
    S = (1 / s**2).reshape(-1, 1)

    cid = _cal[0]
    tx = exp[exp[:, 2] == cid, 0]
    ex = exp[exp[:, 2] == cid, 3:]

    Dc = dzdc(src[:, 1:], ex, _cal[1:], tx)

    N = Dc.T @ (S * Dc)
    b = Dc.T @ ((o - c) / s**2)

    cfac = jax.scipy.linalg.cho_factor(N)
    delta = jax.scipy.linalg.cho_solve(cfac, b)
    print(N, b, delta, cfac)

    return _cal.at[1:].set(_cal[1:] + delta)


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
    return jnp.vstack(
        [update_calibration_inner(obs, ref, src, exp, _) for _ in cal])
