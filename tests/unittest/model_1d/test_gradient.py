#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model_1d.gradient import dzds, dzde, dzdc


@fixture
def epoch():
    return np.array([0.0, 3.0, 6.0, 9.0])

@fixture
def src():
    return np.array([
        [0.0, 0.0, 0.0],
        [np.pi/4, 0.0, 0.0]
    ])

@fixture
def exp():
    return np.array([
        [0, 0.0, 1.0],
        [0, 0.0, 1.0],
        [0, 0.0, 1.0],
        [0, 0.0, 1.0],
    ])

@fixture
def cal():
    return np.array([
        [0.0, 0.0, 0.0]
    ])


def test_dzds(src, exp, cal, epoch):
     z = dzds(src[0], exp, cal[0], epoch)
     assert np.isfinite(z).all()


def test_dzde(src, exp, cal, epoch):
     z = dzde(src, exp[0], cal[0], epoch[0])
     assert np.isfinite(z).all()


def test_dzdc(src, exp, cal, epoch):
     z = dzdc(src, exp, cal[0], epoch)
     assert np.isfinite(z).all()
