#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
from .exposure import zeta, exposure

__all__ = (
  'estimate',
  'estimate_for_exposure',
  'estimate_for_source',
)

_iterate_src = jax.vmap(zeta, (0, None, None, None), 0)
_iterate_exp = jax.vmap(zeta, (None, 0, None, 0), 0)


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


def estimate_for_exposure(src, exp, cal):
    ''' Generate the focal plane position for a specific exposure

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

    Returns:
      The estimated measurements for the specific exposure.
    '''
    z = _iterate_src(src[:, 1:], exp[2:], cal, exp[1])
    return jnp.stack([src[:, 0], jnp.tile(exp[0], z.size), z]).T


def estimate_for_source(src, exp, cal):
    ''' Generate the focal plane position for a specific source

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

    Returns:
      The estimated measurements for the specific source.
    '''
    z = _iterate_exp(src[1:], exp[:, 2:], cal, exp[:, 1])
    return jnp.stack([jnp.tile(src[0], z.size), exp[:, 1], z]).T
