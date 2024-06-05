#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model.exposure import zeta, exposure


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


def test_zeta(src, exp, cal):
    t = exp[:, 0]
    s = src[:, 1:]
    e = exp[:, 3:]
    c = cal[:, 1:]

    z = zeta(s[0], e[0], c[0], t)
    assert z == approx([0.0, 0.0, 0.0])

    z = zeta(s[1], e[0], c[0], t)
    assert z == approx([1.0, 1.0, 1.0])


def test_exposure(src, exp, cal):
    z = exposure(src, exp, cal)[:, 2]

    assert z == approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
