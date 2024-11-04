#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Apply aberration to transform natural direction into proper direction.
'''
import jax.numpy as jnp
import numpy as np
import jax

from . import constants as c


__all__ = [
    'aberration',
]


def aberration(p_natural, velocity, solar_distance, lorenz_m1):
    ''' Apply aberration to transform directions

    Arguments:
        p_natural: `Array[*, 3]`
          natural direction to the source (unit vector).

        velocity: `Array[3]`
          observer's velocity with respect to the Solar System barycenter
          in units of c.

        distance_sun: `float`
          distance between the observer and the sun (au).

        lorenz_bm1: `float`
          reciprocal of the Lorenz factor.
          sqrt(1 - |β|²)

    Returns:
        p_proper: `Array[*, 3]`
          proper direction to the source (unit vector).
    '''
    velocity = velocity.reshape((1, 3))

    pdv = p_natural @ velocity.T
    w1 = 1.0 + pdv / (1.0 + lorenz_m1)
    w2 = c.ERFA_SRS / solar_distance

    p_proper = lorenz_m1 * p_natural \
        + w1 * velocity + w2 * (velocity + pdv * p_natural)

    norm = jnp.sqrt(jnp.sum(p_proper ** 2, axis=1, keepdims=True))

    return p_proper / norm
