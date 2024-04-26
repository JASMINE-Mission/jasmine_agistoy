#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from .estimate import estimate_for_exposure
from .gradient import dzde

__all__ = (
  'update_exposure',
)


def update_exposure_inner(obs, ref, src, exp, cal):
    c = estimate_for_exposure(src, exp, cal)[:, 2]
    o = obs[obs[:, 1] == exp[0]][:, 2]
    s = obs[obs[:, 1] == exp[0]][:, 3] + ref[:, 3]

    De = dzde(src[:, 1:], exp[2:], cal, exp[1])

    N = De.T @ jnp.diag(1.0 / s**2) @ De
    b = De.T @ ((o - c) / s**2)

    cfac = jax.scipy.linalg.cho_factor(N)

    delta = jax.scipy.linalg.cho_solve(cfac, b)
    return exp.at[2:].set(exp[2:] + delta)


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
