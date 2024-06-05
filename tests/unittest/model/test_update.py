#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import jax.numpy as jnp
import numpy as np

from toybis.model.estimate import estimate
from toybis.model.update_exposure import update_exposure
from toybis.model.update_source import update_source
from toybis.model.update_calibration import update_calibration


@fixture
def src():
    return jnp.array([
        [0, 0.0, 0.0, 0.0],
        [1, np.pi/4, 0.0, 0.0]
    ])

@fixture
def exp():
    return jnp.array([
        [0.0, 0, 0, 0.0, 1.0],
        [3.0, 1, 0, np.pi/4, 1.0],
        [6.0, 2, 0, 0.0, 1.0],
        [9.0, 3, 0, np.pi/4, 1.0],
    ])

@fixture
def cal():
    return jnp.array([
        [0, 0.0, 0.0, 0.0],
    ])

@fixture
def obs():
    return jnp.array([
        [0, 0,  0.0, 0.1],
        [1, 0,  1.0, 0.1],
        [0, 1, -1.0, 0.1],
        [1, 1,  0.0, 0.1],
        [0, 2,  0.0, 0.1],
        [1, 2,  1.0, 0.1],
        [0, 3, -1.0, 0.1],
        [1, 3,  0.0, 0.1],
    ])

@fixture
def ref():
    return jnp.array([
        [0, 0, 0, 0, 1e3, 1e3, 1e3],
        [1, 0, 0, 0, 1e3, 1e3, 1e3],
    ])


def test_update_exposure(obs, ref, src, exp, cal):
    ex = update_exposure(obs, ref, src, exp, cal)
    assert exp == approx(ex, abs=1e-4)


def test_update_source(obs, ref, src, exp, cal):
    sx = update_source(obs, ref, src, exp, cal)
    assert src == approx(sx, abs=1e-4)


def test_update_calibration(obs, ref, src, exp, cal):
    cx = update_calibration(obs, ref, src, exp, cal)
    assert cal == approx(cx, abs=1e-4)
