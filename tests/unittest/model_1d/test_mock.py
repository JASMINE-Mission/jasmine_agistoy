#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import fixture


from toybis.model_1d.mock import generate_mock_source
from toybis.model_1d.mock import generate_mock_source_pair
from toybis.model_1d.mock import generate_mock_exposure
from toybis.model_1d.mock import generate_mock_exposure_pair
from toybis.model_1d.mock import generate_mock_calibration
from toybis.model_1d.mock import generate_mock_calibration_pair
from toybis.model_1d.mock import generate_mock_simulation
from toybis.model_1d.exposure import exposure


def test_mock_source():
    for n in [1, 100, 1000]:
      assert generate_mock_source(n).shape == (n, 5)

def test_mock_source_pair():
    for n in [1, 100, 1000]:
      src, shat = generate_mock_source_pair(n)
      assert src.shape == shat.shape

def test_mock_exposure():
    for n in [1, 100, 1000]:
      assert generate_mock_exposure(n).shape == (n, 4)

def test_mock_exposure_pair():
    for n in [1, 100, 1000]:
      src, shat = generate_mock_exposure_pair(n)
      assert src.shape == shat.shape

def test_mock_calibration():
    for n in [1, 100, 1000]:
      assert generate_mock_calibration(n).shape == (n, 3)

def test_mock_calibration_pair():
    for n in [1, 100, 1000]:
      src, shat = generate_mock_calibration_pair(n)
      assert src.shape == shat.shape

def test_mock_simulation():
    t, h = generate_mock_simulation(10, 100, 1)

    assert t[0].shape == (10, 5)
    assert t[1].shape == (100, 4)
    assert t[2].shape == (1, 3)
    assert h[0].shape == (10, 5)
    assert h[1].shape == (100, 4)
    assert h[2].shape == (1, 3)


@fixture
def src(n_source=100):
    return generate_mock_source(n_source)

@fixture
def exp(n_exposure=100):
    return generate_mock_exposure(n_exposure)

@fixture
def cal(n_calib=10):
    return generate_mock_calibration(n_calib)


def test_with_exposure(src, exp, cal):
    return src, exp, cal
