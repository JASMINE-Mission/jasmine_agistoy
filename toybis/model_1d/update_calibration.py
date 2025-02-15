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
    _exp = exp[exp[:, 1] == _cal[0], :]

    c = estimate(src, _exp, _cal)[:, 3]
    o = obs[obs[:, 2] == _cal[0], 3]
    s = obs[obs[:, 2] == _cal[0], 4]
    S = jnp.array([0.1, 0.0001, 0.0001])
    p = _cal[1:]

    tx = _exp[:, 2]

    Dc = dzdc(src[:, 1:], _exp[:, 3:], _cal[1:], tx)

    N = Dc.T @ ((1 / s**2).reshape(-1, 1) * Dc) + jnp.diag(1 / S**2)
    b = Dc.T @ ((o - c) / s**2) - p / S**2

    cfac = jax.scipy.linalg.cho_factor(N)
    delta = jax.scipy.linalg.cho_solve(cfac, b)

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
