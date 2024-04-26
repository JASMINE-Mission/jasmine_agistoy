#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from .estimate import estimate_for_source
from .gradient import dzds

__all__ = (
  'update_source',
)


def update_source_inner(obs, ref, src, exp, cal):
    c = estimate_for_source(src, exp, cal)[:, 2]
    o = obs[obs[:, 0] == src[0]][:, 2]
    s = obs[obs[:, 0] == src[0]][:, 3]
    S = ref[int(src[0])][3:6]
    p = src[1:] - ref[int(src[0])][0:3]

    Ds = dzds(src[1:], exp[:, 2:], cal, exp[:, 1])

    N = Ds.T @ ((1 / s**2).reshape(-1, 1) * Ds) + jnp.diag(1 / S**2)
    b = Ds.T @ ((o - c) / s**2) - p / S**2

    cfac = jax.scipy.linalg.cho_factor(N)
    delta = jax.scipy.linalg.cho_solve(cfac, b)

    return src.at[1:].set(src[1:] + delta)


def update_source(obs, ref, src, exp, cal):
    ''' Updates of the source parameters

    Arguments:
        obs: measurements
        ref: reference catalog
        src: source parameters
        exp: exposure parameters
        cal: calibration parameters

    Returns:
      The updated source parameters.
    '''
    return jnp.vstack(
        [update_source_inner(obs, ref, _, exp, cal) for _ in src])
