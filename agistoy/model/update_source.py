#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from .estimate import estimate
from .gradient import dzds

__all__ = (
  'update_source',
)


def update_source_inner(obs, ref, _src, exp, cal):
    c = estimate(_src, exp, cal)[:, 2]
    o = obs[obs[:, 0] == _src[0]][:, 2]
    s = obs[obs[:, 0] == _src[0]][:, 3]
    S = ref[ref[:, 0] == _src[0]][0][4:7]
    p = _src[1:] - ref[int(_src[0])][1:4]

    _ = []
    for cn in range(cal.shape[0]):
        cid = cal[cn, 0]
        tx = exp[exp[:, 2] == cid, 0]
        ex = exp[exp[:, 2] == cid, 3:]
        cx = cal[cn, 1:]
        _.append(dzds(_src[1:], ex, cx, tx))
    Ds = jnp.vstack(_)

    N = Ds.T @ ((1 / s**2).reshape(-1, 1) * Ds) + jnp.diag(1 / S**2)
    b = Ds.T @ ((o - c) / s**2) - p / S**2

    cfac = jax.scipy.linalg.cho_factor(N)
    delta = jax.scipy.linalg.cho_solve(cfac, b)
    return _src.at[1:].set(_src[1:] + delta)


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
