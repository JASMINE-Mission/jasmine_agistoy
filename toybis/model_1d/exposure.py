#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
from .position_focal_plane import position_focal_plane

__all__ = (
    'zeta',
    'exposure',
)


zeta = jax.jit(position_focal_plane)

_iterate_src = jax.vmap(zeta, (0, None, None, None), 0)
_iterate_full = jax.vmap(_iterate_src, (None, 0, None, 0), 0)


def exposure(src, exp, cal):
    ''' Calculate positions from the parameters

    Arguments:
        src: source parameters with source_id
          src[:,0]: source_id
          src[:,1]: position
          src[:,2]: proper motion
          src[:,3]: parallax
        exp: exposure parameters with exposure_id and observation time
          exp[:,0]: exposure_id
          exp[:,1]: calibration_id
          exp[:,2]: observation epoch
          exp[:,3]: pointing direction
          exp[:,4]: optics scaling
        cal: calibration parameters
          cal[:,0]: calibration_id
          cal[:,1]: offset
          cal[:,2]: linear scaling
          cal[:,3]: quadratic parameter

    Returns:
      The estimated measurements for the given parameters.

    Note:
      exp and epoch should have the same number of elements.
    '''
    src = jnp.atleast_2d(src)
    exp = jnp.atleast_2d(exp)
    cal = jnp.atleast_2d(cal)

    cal_id = exp[:, 1]
    zarr = []
    for n, cid in enumerate(np.unique(cal_id)):
        t = exp[cal_id == cid, 2]
        s = src[:, 1:]
        e = exp[cal_id == cid, 3:]
        c = cal[n, 1:]
        v = _iterate_full(s, e, c, t).ravel()
        x = np.tile(src[:, 0], e.shape[0])
        y = np.repeat(exp[cal_id == cid, 0], src.shape[0])
        z = np.tile(cid, x.shape[0])
        zarr.append(jnp.stack([x, y, z, v]).T)
    return jnp.concatenate(zarr)
