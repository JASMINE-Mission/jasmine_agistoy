#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.erfa_tools.aberration import aberration
from toybis.erfa_tools.constants import ERFA_SRS

@fixture
def p_source():
    return jnp.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ])


def norm(array, axis=1):
    return np.sqrt(np.sum(array**2, axis=axis))


def aberration_shift(s, v):
    # Estimate of the abberation magnitude Eq. (18) of Klioner (2003)
    # Klioner (2003), AJ, 125, 1580
    return s - np.acos((np.abs(v) + np.cos(s)) / (1 + np.abs(v) * np.cos(s)))


def check_unitary(array, axis=1):
    return approx(np.sum(array**2, axis=axis) - 1) == 0


def test_aberration_1d():
    p_source = jnp.array([1, 0, 0])
    velocity = jnp.array([0, 1e-8, 0])
    distance = 1.0
    lorentz_bm1 = np.sqrt(1.0 - (velocity**2).sum())

    p_ab = aberration(p_source, velocity, distance, lorentz_bm1)

    print(p_ab)


def test_aberration_2dx1d():
    p_source = jnp.array([[1, 0, 0], [1, 0, 0]])
    velocity = jnp.array([0, 1e-8, 0]).reshape((1, 3))
    distance = 1.0
    lorentz_bm1 = np.sqrt(1.0 - (velocity**2).sum())

    p_ab = aberration(p_source, velocity, distance, lorentz_bm1)

    print(p_ab)


def test_aberration_1dx2d():
    p_source = jnp.array([1, 0, 0]).reshape((1, 3))
    velocity = jnp.array([[0, 1e-8, 0], [0, 0, 1e-8]])
    distance = 1.0
    lorentz_bm1 = np.sqrt(1.0 - (velocity**2).sum())

    p_ab = aberration(p_source, velocity, distance, lorentz_bm1)

    print(p_ab)


def test_aberration_direction(p_source):
    ''' test berration direction '''
    velocity = jnp.array([0, 0, 1e-8]).reshape((1, 3))
    solar_distance = 1.0
    lorenz_bm1 = np.sqrt(1.0 - (velocity ** 2).sum())

    p_ab = aberration(p_source, velocity, solar_distance, lorenz_bm1)

    delta = p_ab - p_source
    assert check_unitary(p_ab)      # p should be unit vectors
    assert approx(delta[0]) != 0    # x-direction is affected
    assert approx(delta[1]) != 0    # x-direction is affected
    assert approx(delta[2]) != 0    # y-direction is affected
    assert approx(delta[3]) != 0    # y-direction is affected
    assert approx(delta[4]) == 0    # z-direction is not affected
    assert approx(delta[5]) == 0    # z-direction is not affected


def test_aberration_magnitude():
    ''' test magnitude of aberration '''
    source = jnp.array([[0, 0, 1]])
    velocity = jnp.array([
        [0.0001, 0, 0],    # 30 km/s
        [0.0010, 0, 0],
        [0.0100, 0, 0],
        [0, 0.0001, 0],    # 30 km/s
        [0, 0.0010, 0],
        [0, 0.0100, 0],
    ])
    solar_distance = 1.0
    lorenz_bm1 = np.sqrt(1.0 - (velocity ** 2).sum(axis=1))

    for v, l in zip(velocity, lorenz_bm1):
        p_ab = aberration(source, v, solar_distance, l)

        # calculate angular shifts along with x- and y-axes
        tx = np.atan2(p_ab[0, 0], p_ab[0, 2])
        ty = np.atan2(p_ab[0, 1], p_ab[0, 2])

        # calculate the expected angular shifts
        sx = aberration_shift(np.pi / 2.0, v[0] * (1.0 + ERFA_SRS))
        sy = aberration_shift(np.pi / 2.0, v[1] * (1.0 + ERFA_SRS))

        # evaluate the differences in units of μas
        delta_x = np.abs(tx - sx) / np.pi * (180 * 3600e6)
        delta_y = np.abs(ty - sy) / np.pi * (180 * 3600e6)

        print(tx, sx, delta_x)
        print(ty, sy, delta_y)
        assert check_unitary(p_ab)    # p should be unit vectors
        assert delta_x < 0.01         # delta_x smaller than 0.01 μas
        assert delta_y < 0.01         # delta_y smaller than 0.01 μas
