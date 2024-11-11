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

    for p in p_source:
        p_natural = deflection(mass, p, p, q_observer, distance, 1e-8)

        delta = p_natural - p
        assert check_unitary(p_natural)
        if p[0]!=0:
            assert approx(delta[0]) == 0    # source at x-direction is not affected
        elif p[1]!=0:
            assert approx(delta[1]) != 0    # y-direction is affected
            assert approx(delta[2]) == 0    # z-direction is not affected
        elif p[2]!=0:
            assert approx(delta[2]) != 0    # z-direction is not affected
            assert approx(delta[1]) == 0    # y-direction is affected



def test_deflection_by_sun(p_source, q_observer):
    ''' test the function '''

    assert np.all(deflection_by_sun(p_source[0],q_observer,1) == deflection_by_sun(p_source[0],q_observer,10))