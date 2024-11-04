#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Propagate the stellar positions considering the proper motions
'''
import jax.numpy as jnp
import astropy.constants as c
import astropy.units as u

from .transform import spherical_to_cartesian

__all__ = [
    'propermotion',
]


def propermotion(spherical, pm, dt, p_observer):
    ''' Calculate the instananeous stellar positions

    Arguments:
        spherical: `Array[*, 2]`
          source spherical coordinates in the ICRS.

        pm: `Array[*, 2]`
          proper motions of the source.
          the first column contains d_ra, while the second column contains
          d_dec. both are in units of radians/yr. The RA proper motion is
          given in cos(Dec) d_ra / dt.

        dt: `float`
          the time difference from the catalog epoch in the TCB system
          in units of Juliand years.

        p_observer: `Array[3]`
          unit vector from the SSB to the observer.

    Returns:
        p_source: `Array[*, 3]`
          the source direction taking into account the parallax motion.

    Note:
        The radial velocities of the sources are ignored.
        The function may fail if any sources are located around the pole.
    '''
    p_observer = p_observer.reshape((1, 3))
    p_source = spherical_to_cartesian(spherical)

    dt_au = (1.0 * u.au / c.c).to_value('yr')
    dt_obs = dt + (p_source @ p_observer.T) * dt_au

    return spherical + pm * dt_obs
