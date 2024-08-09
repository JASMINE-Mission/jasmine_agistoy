#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import fixture


from toybis.model_1d.mock import generate_mock_source
from toybis.model_1d.mock import generate_mock_source_pair
from toybis.model_1d.mock import generate_mock_observation
from toybis.model_1d.mock import generate_mock_observation_pair
from toybis.model_1d.mock import generate_mock_simulation
from toybis.model_1d.mock import generate_mock_reference
from toybis.model_1d.exposure import exposure


@fixture
def src(n_source=100):
    return generate_mock_source(n_source)

@fixture
def obs(n_exposure=100, n_calib=2):
    return generate_mock_observation(n_exposure, n_calib)


def test_mock_source():
    for n in [1, 100, 1000]:
      assert generate_mock_source(n).shape == (n, 5)


def test_mock_source_pair():
    for n in [1, 100, 1000]:
      src, shat = generate_mock_source_pair(n)
      assert src.shape == shat.shape


def test_mock_exposure():
    for n in [1, 100, 1000]:
      exp, cal = generate_mock_observation(n, 1)
      assert exp.shape == (n, 5)
      assert cal.shape == (1, 4)


def test_mock_exposure_pair():
    for n in [1, 100, 1000]:
      obs, ohat = generate_mock_observation_pair(n, 1)
      assert obs[0].shape == ohat[0].shape
      assert obs[1].shape == ohat[1].shape


def test_mock_simulation():
    t, h = generate_mock_simulation(10, 100, 1)

    assert t[0].shape == (10, 5)
    assert t[1].shape == (100, 5)
    assert t[2].shape == (1, 4)
    assert h[0].shape == (10, 5)
    assert h[1].shape == (100, 5)
    assert h[2].shape == (1, 4)


def test_mock_reference(src):
    ref = generate_mock_reference(src)

    assert ref.shape[0] == src.shape[0]


def test_with_exposure(src, obs):
    exp, cal = obs
    assert exposure(src, exp, cal).shape == (10000, 4)
