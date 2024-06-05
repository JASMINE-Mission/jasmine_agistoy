#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from .longitude import longitude

__all__ = (
    'epoch_position',
)


def epoch_position(s, t):
    ''' Celestial position of a source

    Arguments:
        s: source parameters
          s[0]: position
          s[1]: proper motion
          s[2]: parallax

    Returns:
      The celestial position of the source at the observation epoch.
    '''
    return s[0] + s[1] * t + s[2] * jnp.sin(longitude(t))
