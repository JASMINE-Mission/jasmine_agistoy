#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.erfa.aberration import aberration
from toybis.erfa.constants import ERFA_SRS

@fixture
def p_source():
    return jnp.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])


def norm(array, axis=1):
    return np.sqrt(np.sum(array**2, axis=axis))


def shift(s, v):
    return s - np.acos((v + np.cos(s)) / (1 - v * np.cos(s)))


def check_unitary(array, axis=1):
    return approx(np.sum(array**2, axis=axis) - 1) == 0


def test_aberration_direction(p_source):

    velocity = jnp.array([0, 0, 1e-8])
    solar_distance = 1.0
    lorenz_bm1 = np.sqrt(1.0 - (velocity ** 2).sum())

    p_ab = aberration(p_source, velocity, solar_distance, lorenz_bm1)

    delta = p_ab - p_source
    print(delta)
    assert check_unitary(p_ab)      # p should be a unit vector
    assert approx(delta[0]) != 0    # x-direction is affected
    assert approx(delta[1]) != 0    # y-direction is affected
    assert approx(delta[2]) == 0    # z-direction is not affected


def test_aberration_magnitude():

    source = jnp.array([[0, 0, 1]])
    velocity = jnp.array([
        [0.01, 0, 0],
        [0.05, 0, 0],
        [0.25, 0, 0],
        [0, 0.01, 0],
        [0, 0.05, 0],
        [0, 0.25, 0],
    ])
    solar_distance = 1.0
    lorenz_bm1 = np.sqrt(1.0 - (velocity ** 2).sum(axis=1))

    for v, l in zip(velocity, lorenz_bm1):
        p_ab = aberration(source, v, solar_distance, l)

        # calculate angular shifts along with x- and y-axes
        tx = np.atan2(p_ab[0, 0], p_ab[0, 2])
        ty = np.atan2(p_ab[0, 1], p_ab[0, 2])

        # calculate the expected angular shifts
        sx = shift(np.pi / 2.0, v[0] * (1.0 + ERFA_SRS))
        sy = shift(np.pi / 2.0, v[1] * (1.0 + ERFA_SRS))

        # evaluate the differences in units of μas
        delta_x = (tx - sx) / np.pi * (180 * 3600e6)
        delta_y = (ty - sy) / np.pi * (180 * 3600e6)

        assert check_unitary(p_ab)    # p should be a unit vector
        assert delta_x < 0.01         # delta_x smaller than 0.01 μas
        assert delta_y < 0.01         # delta_y smaller than 0.01 μas
