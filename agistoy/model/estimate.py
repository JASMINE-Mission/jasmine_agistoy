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


def estimate(src, exp, cal, noise=0.00):
    ''' Generate the focal plane position from the parameters

    Arguments:
        src: source parameters
        exp: exposure parameters
        cal: calibration parameters

    Returns:
      The estimated measurements for the given paramters.
      The measurements are disturbed if noise is not zero.
    '''
    obs = exposure(src[:, 1:], exp[:, 2:], cal, exp[:, 1])
    if noise > 0:
        obs = np.random.normal(obs, noise)
    sid = jnp.tile(src[:, 0], exp.shape[0])
    eid = jnp.repeat(exp[:, 0], src.shape[0])
    return jnp.stack([sid, eid, obs]).T


def estimate_for_exposure(src, exp, cal):
    ''' Generate the focal plane position for a specific exposure

    Arguments:
        src: source parameters (iteration key)
        exp: exposure parameters of a specific exposure
        cal: calibration parameters

    Returns:
      The estimated measurements for the specific exposure.
    '''
    z = _iterate_src(src[:, 1:], exp[2:], cal, exp[1])
    return jnp.stack([src[:, 0], jnp.tile(exp[0], z.size), z]).T


def estimate_for_source(src, exp, cal):
    ''' Generate the focal plane position for a specific source

    Arguments:
        src: source parameters of a specific source
        exp: exposure parameters (iteration key)
        cal: calibration parameters

    Returns:
      The estimated measurements for the specific source.
    '''
    z = _iterate_exp(src[1:], exp[:, 2:], cal, exp[:, 1])
    return jnp.stack([jnp.tile(src[0], z.size), exp[:, 1], z]).T
