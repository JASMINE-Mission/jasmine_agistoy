#!/usr/bin/env python
# -*- coding: utf-8 -*-
import jax.numpy as jnp


__all__ = [
    'spherical_to_cartesian',
    'cartesian_to_spherical',
]


def unitvector(p_vector):
    ''' Conert a vector into a unit vector and its modulus

    Arguments:
        p_vector: `Array[*, 3]`
          cartesian coordiantes.
          coordinates are stored in the [x, y, z] order.

    Returns:
        u_vector: `Array[*, 3]`
          cartesian coordinates (unit vector).
          coordinates are stored in the [x, y, z] order.

        modulus: `Array[*]`
          the length of the p_vector.
    '''

    modulus = jnp.sqrt(jnp.sum(p_vector**2, axis=1, keepdims=True))

    u_vector = jnp.where(
        modulus == 0, jnp.zeros_like(p_vector), p_vector / modulus)

    return u_vector, modulus


def spherical_to_cartesian(spherical):
    ''' Convert spherical coordinates into cartesian coordinates

    Arguments:
        spherical: `Array[*, 2]`
          spherical coordinates in units of radian.
          coordinates are stored in the [lon, lat] order.

    Returns:
        p: `Array[*, 3]`
          cartesian coordinates (unit vector).
          coordinates are stored in the [x, y, z] order.
    '''

    return jnp.array([
        jnp.cos(spherical[:, 0]) * jnp.cos(spherical[:, 1]),
        jnp.sin(spherical[:, 0]) * jnp.cos(spherical[:, 1]),
        jnp.sin(spherical[:, 1])
    ]).T


def cartesian_to_spherical(vector):
    ''' Convert cartesian coordinates into spherical coordinates

    Arguments:
        vector: `Array[*, 3]`
          cartesian coordinates.
          coordinates are stored in the [x, y, z] order.

    Returns:
        theta: `Array[*]`
          longitude angle in units of radian.

        phi: `Array[*]`
          latitude angle in units of radian.
    '''

    z = vector[:, 2]
    d2 = vector[:, 1]**2 + vector[:, 0]**2

    return jnp.array([
        jnp.where(d2 == 0, 0.0, jnp.atan2(vector[:, 1], vector[:, 0])),
        jnp.where(z == 0, 0.0, jnp.atan2(vector[:, 2], jnp.sqrt(d2)))
    ]).T
