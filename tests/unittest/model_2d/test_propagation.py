#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model_2d.propagation import _icrs2comrs,_comrs2fovrs_fromquat,_comrs2fovrs, _fovrs2fprs, _comrs2fprs

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
def phi_c2():
    return np.pi/4

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

@fixture
def pt_ra():
    return 0.0

@fixture
def pt_dec():
    return 0.0

@fixture
def pt_rot():
    return 0

@fixture
def eta():
    return np.pi/4

@fixture
def zeta():
    return 0.0

@fixture
def F():
    return 1


def test_icrs2comrs(ra, dec):
    phi_c,lambda_c = _icrs2comrs(ra,dec)

    assert phi_c == approx(0.0)

    assert lambda_c == approx(0.0)

def test_comrs2fovrs_fromquat(phi_c,lambda_c, rx_at,ry_at,angle_at):
    eta,zeta = _comrs2fovrs_fromquat(phi_c,lambda_c, rx_at,ry_at,angle_at)

    if rx_at == 0 and ry_at == 0:
        assert eta.item() == approx(phi_c-angle_at)
        assert zeta.item() == approx(lambda_c)

def test_comrs2fovrs(phi_c,lambda_c, pt_ra,pt_dec,pt_rot):
    eta,zeta = _comrs2fovrs(phi_c,lambda_c, pt_ra,pt_dec,pt_rot)

    assert eta.item() == approx(0.)
    assert zeta.item() == approx(0.)

def test_fovrs2fprs(eta,zeta,F):
    xf,yf = _fovrs2fprs(eta,zeta,F)

    assert xf.item() == approx(1.)
    assert yf.item() == approx(0.)

def test_comrs2fprs(phi_c2,lambda_c, pt_ra,pt_dec,pt_rot,F):
    xf,yf = _comrs2fprs(phi_c2,lambda_c, pt_ra,pt_dec,pt_rot,F)
    print(xf,yf,xf.item(),yf.item())
    assert xf.item() == approx(1.)
    assert yf.item() == approx(0.)
