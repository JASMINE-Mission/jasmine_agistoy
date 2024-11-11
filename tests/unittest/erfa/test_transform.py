#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.erfafuncs.transform import \
    unitvector, spherical_to_cartesian, cartesian_to_spherical


@fixture
def spherical():
    div3 = 1.0 / 3.0
    return jnp.pi * jnp.array([
        [+0.00, +0.00],
        [+div3, +0.00],
        [+0.25, +0.00],
        [+0.50, +0.00],
        [+1.00, +0.00],
        [+1.50, +0.00],
        [+2.00, +0.00],
        [+0.00, +0.25],
        [+0.00, +div3],
    ])


@fixture
def cartesian():
    sq2 = np.sqrt(2) / 2.0
    sq3 = np.sqrt(3) / 2.0
    return jnp.array([
        [+1.0, +0.0, +0.0],
        [+0.5, +sq3, +0.0],
        [+sq2, +sq2, +0.0],
        [+0.0, +1.0, +0.0],
        [-1.0, +0.0, +0.0],
        [+0.0, -1.0, +0.0],
        [+1.0, +0.0, +0.0],
        [+sq2, +0.0, +sq2],
        [+0.5, +0.0, +sq3],
    ])


def fix_longitude(lon):
    return np.mod(lon, 2 * np.pi)


def test_unitvector():
    p_vector = jnp.array([
        [1, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [0, 0, 0],
    ])

    u_vector, modulus = unitvector(p_vector)


    modulus0 = np.array([1, 2, 2, 2, 0]).reshape((-1, 1))
    assert approx(modulus - modulus0) == 0
    assert approx(u_vector[1] - np.array([1, 0, 0])) == 0
    assert approx(u_vector[4] - np.array([0, 0, 0])) == 0


def test_spherical_to_cartesian(spherical, cartesian):
    vec = spherical_to_cartesian(spherical)

    print(cartesian - vec)
    assert approx(cartesian - vec) == 0


def test_cartesian_to_spherical(cartesian, spherical):

    sph = cartesian_to_spherical(cartesian)

    delta_theta = fix_longitude(sph[:, 0] - spherical[:, 0])
    delta_phi = sph[:, 1] - spherical[:, 1]

    assert approx(delta_theta) == 0
    assert approx(delta_phi) == 0
