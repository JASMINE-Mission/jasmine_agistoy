#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Apply gravitational deflection to transform coordinate direction into
    natural direction.
'''
import jax.numpy as jnp
import numpy as np
import jax

from . import constants as c


__all__ = [
    'deflection',
    'deflection_by_sun',
]


def deflection(mass, p_source, q_source, q_observer, distance,
               limiter=1e-8):
    ''' Apply gravitational deflection to transform directions

    Arguments:
        mass: `float`
          mass of the gravitating body in units of solar mass.

        p_source: `Array[*, 3]`
          coordinate direction toward sources (unit vector).

        q_source: `Array[*, 3]`
          direction from the body to sources (unit vector).

        q_observer: `Array[3]`
          direction from the body to the observer (unit vector).

        distance: `float`
          distance between the body and observer.

        limiter: `float`
          deflection is artificially reduced to avoid zero division.

    Returns:
        p_natural: `Array[*, 3]`
          natural direction from the observer to sources (unit vector).
    '''
    q_observer = q_observer.reshape((1, 3))

    qdqpe = (q_source * (q_source + q_observer)).sum(axis=1, keepdims=True)
    w = mass * c.ERFA_SRS / distance / jnp.clip(qdqpe, min=limiter)

    eq = jnp.cross(q_observer, q_source)
    peq = jnp.cross(p_source, eq)

    p_natural = p_source + w * peq

    norm = jnp.sqrt(jnp.sum(p_natural**2, axis=1, keepdims=True))

    return p_natural / norm


def deflection_by_sun(p_source, q_observer, distance):
    ''' Apply gravitational deflection by the Sun to transorm directions

    Arguments:
        p_source: `Array[*, 3]`

        q_observer: `Array[*, 3]`

        distance: `float`
    '''

    limiter = 1e-6 / np.clip(distance**2, min=1.0)

    # approximate p_source == q_source
    p_natural = deflection(
        1.0, p_source, p_source, q_observer, distance, limiter=limiter)

    return p_natural
