#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as jnp


__all__ = [
    'spherical_to_cartesian',
    'cartesian_to_spherical',
]


def spherical_to_cartesian(theta, phi):
    ''' Convert spherical coordinates into cartesian coordinates

    Arguments:
        theta: `Array[*]`
          longitude angle in units of radian.

        phi: `Array[*]`
          latitude angle in units of radian.

    Returns:
        p: `Array[*, 3]`
          cartesian coordinates (unit vector).
          coordinates are stored in [x, y, z] order.
    '''

    cp = jnp.cos(phi)
    return jnp.array([
        jnp.cos(theta) * cp,
        jnp.sin(theta) * cp,
        jnp.sin(phi)
    ]).T


def cartesian_to_spherical(vector):
    ''' Convert cartesian coordinates into spherical coordinates

    Arguments:
        vector: `Array[*, 3]`
          cartesian coordinates.
          coordinates are stored in [x, y, z] order.

    Returns:
        theta: `Array[*]`
          longitude angle in units of radian.

        phi: `Array[*]`
          latitude angle in units of radian.
    '''

    z = vector[:, 2]
    d2 = vector[:, 1]**2 + vector[:, 0]**2

    theta = jnp.where(d2 == 0, 0.0, jnp.atan2(vector[:, 1], vector[:, 0]))
    phi = jnp.where(z == 0, 0.0, jnp.atan2(vector[:, 2], jnp.sqrt(d2)))

    return theta, phi
