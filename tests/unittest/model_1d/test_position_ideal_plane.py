#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model_1d.position_ideal_plane import position_ideal_plane


@fixture
def epoch():
    return np.array([0.0, 3.0, 6.0])


def test_ideal_plane_null(epoch):
    src = np.array([0.0, 0.0, 0.0])
    exp = np.array([0.0, 1.0])

    xi = position_ideal_plane(src, exp, epoch)

    assert xi == approx([0.0, 0.0, 0.0])


def test_ideal_plane_pan(epoch):
    src = np.array([1.0, 0.0, 0.0])
    exp = np.array([1.0, 1.0])

    xi = position_ideal_plane(src, exp, epoch)

    assert xi == approx([0.0, 0.0, 0.0])


def test_ideal_plane_scale(epoch):
    src = np.array([np.pi/4.0, 0.0, 0.0])
    exp = np.array([0.0, 1.0])

    xi = position_ideal_plane(src, exp, epoch)

    assert xi == approx([1.0, 1.0, 1.0])
