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


def estimate(src, exp, cal, noise=None):
    ''' Generate the focal plane position from the parameters

    Arguments:
        src: source parameters with source_id
          src[:,0]: source_id
          src[:,1]: position
          src[:,2]: proper motion
          src[:,3]: parallax
        exp: exposure parameters with exposure_id and observation time
          exp[:,0]: observation epoch
          exp[:,1]: exposure_id
          exp[:,2]: calibration_id
          exp[:,3]: pointing direction
          exp[:,4]: optics scaling
        cal: calibration parameters
          cal[:,0]: calibration_id
          cal[:,1]: offset
          cal[:,2]: linear scaling
          cal[:,3]: quadratic parameter
    Options:
        noise: A typical measurement noise for each source

    Returns:
      The estimated measurements for the given paramters.
      The measurements are disturbed if noise is not zero.
    '''
    obs = exposure(src, exp, cal)
    if noise is None:
        sig = jnp.zeros(shape=obs.shape[0])
        val = obs[:, 2]
    else:
        if isinstance(noise, float):
            noise = np.tile(noise, src.shape[0])
        sig = np.random.gamma(100.0, np.tile(noise, exp.shape[0]) / 100.0)
        val = np.random.normal(obs[:, 2], sig)
    sid = obs[:, 0]
    eid = obs[:, 1]
    return jnp.stack([sid, eid, val, sig]).T
