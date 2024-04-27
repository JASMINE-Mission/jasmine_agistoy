#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from agistoy.model.estimate import estimate


@fixture
def src():
    return np.array([
        [0, 0.0, 0.0, 0.0],
        [1, np.pi/4, 0.0, 0.0]
    ])

@fixture
def exp():
    return np.array([
        [0.0, 0, 0, 0.0, 1.0],
        [3.0, 1, 0, 0.0, 1.0],
        [6.0, 2, 0, 0.0, 1.0]
    ])

@fixture
def cal():
    return np.array([
        [0, 0.0, 0.0, 0.0]
    ])


def test_estimate(src, exp, cal):
    z = estimate(src, exp, cal)[:, 2]

    assert z == approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_estimate_for_source(src, exp, cal):
    z = estimate(src[0], exp, cal)[:, 2]

    assert z == approx([0.0, 0.0, 0.0])


def test_estimate_for_exposure(src, exp, cal):
    z = estimate(src, exp[0], cal)[:, 2]

    assert z == approx([0.0, 1.0])


def test_estimate_for_calibration(src, exp, cal):
    z = estimate(src, exp, cal[0])[:, 2]

    assert z == approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_estimate(src, exp, cal):
    z = estimate(src, exp, cal, noise=0.1)[:, 2]

    assert z == approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], abs=0.3)
