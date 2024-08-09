#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from .epoch_position import epoch_position

__all__ = (
    'position_ideal_plane',
)


def position_ideal_plane(s, e, t):
    ''' Position on the ideal focal plane

    Arguments:
      s: source parameters
      e: exposure parameters
          e[0]: telescope pointing
          e[1]: telescope scaling

    Returns:
      The position on the ideal focal plane at the observation epoch.
    '''
    alpha = epoch_position(s, t)
    eta = alpha - e[0]
    return jnp.tan(e[1] * eta)
