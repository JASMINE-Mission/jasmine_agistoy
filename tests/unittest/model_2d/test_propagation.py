#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model_2d.propagation import _icrs2comrs,_comrs2fovrs

@fixture
def ra():
    return 0.0

@fixture
def dec():
    return 0.0

@fixture
def phi_c():
    return 0.0

@fixture
def lambda_c():
    return 0.0

@fixture
def rx_at():
    return 0.0

@fixture
def ry_at():
    return 0.0

@fixture
def angle_at():
    return np.pi/4


def test__icrs2comrs(ra, dec):
    phi_c,lambda_c = _icrs2comrs(ra,dec)

    assert phi_c == approx(0.0)

    assert lambda_c == approx(0.0)

def test__comrs2fovrs(phi_c,lambda_c, rx_at,ry_at,angle_at):
    eta,zeta = _comrs2fovrs(phi_c,lambda_c, rx_at,ry_at,angle_at)

    if rx_at == 0 and ry_at == 0:
        assert eta == approx(phi_c-angle_at)
        assert zeta == approx(lambda_c)