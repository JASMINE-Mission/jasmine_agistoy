#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
from .position_focal_plane import position_focal_plane

__all__ = (
    'zeta',
    'exposure',
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
