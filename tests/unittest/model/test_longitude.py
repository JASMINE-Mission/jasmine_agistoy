#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from agistoy.model.longitude import longitude


@fixture
def epoch():
    return np.array([0.0, 3.0, 6.0])

@fixture
def phase():
    return 2 * np.pi * np.array([0.0, 1.0, 2.0])


def test_longitude(epoch, phase):
    assert longitude(epoch, T=3.0) == approx(phase)


def test_longitude_scaling():
    assert longitude(np.array([1.0]), T=1.0 * np.pi) == approx([2.0])
    assert longitude(np.array([2.0]), T=2.0 * np.pi) == approx([2.0])
    assert longitude(np.array([3.0]), T=3.0 * np.pi) == approx([2.0])
