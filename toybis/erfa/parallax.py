#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Propagate the stellar positions considering parallaxes and proper motions
    to obtain instantaneous directions to the sources
'''
import jax.numpy as jnp
import astropy.constants as c
import astropy.units as u

from .constants import ERFA_DAS2R
from .transform import unitvector

__all__ = [
    'parallax',
]


def parallax(p_source, plx, p_observer):
    ''' Calculate the instananeous stellar positions

    Arguments:
        p_source: `Array[*, 3]`
          unit vector toward the source in the ICRS coordinates.

        plx: `Array[*]`
          parallx of the source in units of arcsecond.

        p_observer: `Array[3]`
          unit vector from the SSB to the observer.

    Returns:
        p_source: `Array[*, 3]`
          the source direction taking into account the parallax motion.

    Note:
        The radial velocities of the sources are ignored.
        The function fails if any source is located at the north pole.
    '''

    return unitvector(
        p_source - (plx.reshape((-1, 1)) \
                    * ERFA_DAS2R) * p_observer.reshape((1, 3)))[0]
