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
_iterate_full = jax.vmap(_iterate_src, (None, 0, 0, 0), 0)


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

    cdict = {int(_[0]): _[1:] for _ in cal}
    cal_id = exp[:, 1].astype('int')

    t = exp[:, 2]
    s = src[:, 1:]
    e = exp[:, 3:]
    try:
      c = jnp.take(cal[:, 1:], cal_id, axis=0)
      assert jnp.isfinite(c)
    except:
      c = jnp.array([cdict[int(_)] for _ in cal_id])

    x = np.tile(src[:, 0], e.shape[0])
    y = np.repeat(exp[:, 0], src.shape[0])
    z = np.tile(cal_id, src.shape[0])

    v =_iterate_full(s, e, c, t).ravel()

    return jnp.stack([x, y, z, v]).T
