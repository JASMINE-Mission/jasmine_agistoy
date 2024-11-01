#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.erfa.deflection import deflection, deflection_by_sun


@fixture
def p_source():
    return jnp.array([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ])


@fixture
def q_observer():
    return jnp.array([1, 0, 0])


def check_unitary(array, axis=1):
    return approx(np.sum(array**2, axis=axis) - 1) == 0


def test_deflection_direction(p_source, q_observer):
    ''' test the direction of the gravitational deflection '''

    # assume 1e3 solar mass for test
    mass = 1.0e3
    distance = 1.0

    p_natural = deflection(
        mass, p_source, p_source, q_observer, distance, 1e-8)

    delta = p_natural - p_source
    assert check_unitary(p_natural)    # p should be unit vectors
    assert approx(delta[0]) == 0       # source at x-direction is not affected
    assert approx(delta[1]) != 0       # source at y-direction is affected
    assert approx(delta[2]) != 0       # source at y-direction is affected
    assert approx(delta[3]) != 0       # source at z-direction is affected
    assert approx(delta[4]) != 0       # source at z-direction is affected
    assert approx(delta[1, 2]) == 0    # y-coodinate is not affected
    assert approx(delta[2, 2]) == 0    # y-coodinate is not affected
    assert approx(delta[3, 1]) == 0    # z-coodinate is not affected
    assert approx(delta[4, 1]) == 0    # z-coodinate is not affected


def test_deflection_by_sun(p_source, q_observer):
    ''' test the function '''

    deflection_by_sun(p_source, q_observer, 1.0)
    deflection_by_sun(p_source, q_observer, 10.0)
