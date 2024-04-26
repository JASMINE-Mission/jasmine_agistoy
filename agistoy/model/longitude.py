#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp

__all__ = (
    'longitude',
)


def longitude(t, T=3.0):
    ''' Longitude of the Earth

    Arguments:
        t: epoch of the observation

    Returns:
      The longitude of the Earth in radian.
    '''
    return 2 * jnp.pi * (t / T)
