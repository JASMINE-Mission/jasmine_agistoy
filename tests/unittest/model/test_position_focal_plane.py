#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model.position_focal_plane import position_focal_plane


@fixture
def epoch():
    return np.array([0.0, 3.0, 6.0])


def test_ideal_plane_null(epoch):
    src = np.array([0.0, 0.0, 0.0])
    exp = np.array([0.0, 1.0])
    cal = np.array([0.0, 0.0, 0.0])

    xi = position_focal_plane(src, exp, cal, epoch)

    assert xi == approx([0.0, 0.0, 0.0])


def test_ideal_plane_c0(epoch):
    src = np.array([0.0, 0.0, 0.0])
    exp = np.array([0.0, 1.0])
    cal = np.array([0.1, 0.0, 0.0])

    xi = position_focal_plane(src, exp, cal, epoch)

    assert xi == approx([0.1, 0.1, 0.1])


def test_ideal_plane_c1(epoch):
    src = np.array([np.pi/4.0, 0.0, 0.0])
    exp = np.array([0.0, 1.0])
    cal = np.array([0.0, 0.1, 0.0])

    xi = position_focal_plane(src, exp, cal, epoch)

    assert xi == approx([1.1, 1.1, 1.1])


def test_ideal_plane_c2(epoch):
    src = np.array([np.pi/4.0, 0.0, 0.0])
    exp = np.array([0.0, 1.0])
    cal = np.array([0.0, 0.1, -0.2])

    xi = position_focal_plane(src, exp, cal, epoch)

    assert xi == approx([1.0, 1.0, 1.0])
