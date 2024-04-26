#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np

from .position_focal_plane import position_focal_plane

__all__ = (
    'zeta',
    'exposure',
    'estimate',
)


zeta = jax.jit(position_focal_plane)

_iterate_src = jax.vmap(zeta, (0, None, None, None), 0)
_iterate_exp = jax.vmap(zeta, (None, 0, None, 0), 0)
_iterate_full = jax.vmap(_iterate_src, (None, 0, None, 0), 0)


def exposure(src, exp, cal, epoch):
    ''' Calculate positions from the parameters

    Arguments:
        src: source parameters
        exp: exposure parameters
        cal: calibration parameters
        epoch: observation epoch

    Returns:
      The estimated measurements for the given parameters.

    Note:
      exp and epoch should have the same number of elements.
    '''
    return _iterate_full(src, exp, cal, epoch).ravel()


def estimate(src, exp, cal, noise=None):
    ''' Generate the focal plane position from the parameters

    Arguments:
        src: source parameters with source_id
          src[0]: source_id
          src[1]: position
          src[2]: proper motion
          src[3]: parallax
        exp: exposure parameters with exposure_id and observation time
          exp[0]: exposure_id
          exp[1]: observation epoch
          exp[2]: pointing direction
          exp[3]: optics scaling
        cal: calibration parameters
          cal[0]: offset
          cal[1]: linear scaling
          cal[2]: quadratic parameter
    Options:
        noise: A typical measurement noise for each source

    Returns:
      The estimated measurements for the given paramters.
      The measurements are disturbed if noise is not zero.
    '''
    obs = exposure(src[:, 1:], exp[:, 2:], cal, exp[:, 1])
    if noise is not None:
        sig = np.random.gamma(100.0, np.tile(noise, exp.shape[0]) / 100.0)
        obs = np.random.normal(obs, sig)
    else:
        sig = jnp.zeros(shape=obs.shape)
    sid = jnp.tile(src[:, 0], exp.shape[0])
    eid = jnp.repeat(exp[:, 0], src.shape[0])
    return jnp.stack([sid, eid, obs]).T
