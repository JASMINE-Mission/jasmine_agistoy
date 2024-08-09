#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from .estimate import estimate
from .gradient import dzde

__all__ = (
  'update_exposure',
)


def update_exposure_inner(obs, ref, src, _exp, cal):
    c = estimate(src, _exp, cal)[:, 3]
    o = obs[obs[:, 1] == _exp[0]][:, 3]
    s = obs[obs[:, 1] == _exp[0]][:, 4] + ref[:, 4]
    cx = cal[cal[:, 0] == _exp[1], 1:].ravel()

    De = dzde(src[:, 1:], _exp[3:], cx, _exp[2])

    N = De.T @ jnp.diag(1.0 / s**2) @ De
    b = De.T @ ((o - c) / s**2)

    cfac = jax.scipy.linalg.cho_factor(N)
    delta = jax.scipy.linalg.cho_solve(cfac, b)

    return _exp.at[3:].set(_exp[3:] + delta)


def update_exposure(obs, ref, src, exp, cal):
    ''' Updates of the exposure parameters

    Arguments:
        obs: measurements
        ref: reference catalog
        src: source parameters
        exp: exposure parameters
        cal: calibration parameters

    Returns:
      The updated exposure parameters.
    '''
    return jnp.vstack([
        update_exposure_inner(obs, ref, src, _, cal) for _ in exp])
