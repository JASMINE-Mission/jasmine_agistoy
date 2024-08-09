#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np

from toybis.model_1d.epoch_position import epoch_position


@fixture
def epoch():
    return np.array([0.0, 3.0, 6.0])


def test_epoch_position_null(epoch):
    src = np.array([0.0, 0.0, 0.0])
    x = epoch_position(src, epoch)

    assert x == approx([0.0, 0.0, 0.0])


def test_epoch_position_origin(epoch):
    src = np.array([1.0, 0.0, 0.0])
    x = epoch_position(src, epoch)

    assert x == approx([1.0, 1.0, 1.0])


def test_epoch_position_motion(epoch):
    src = np.array([0.0, 1.0, 0.0])
    x = epoch_position(src, epoch)

    assert x == approx([0.0, 3.0, 6.0])


def test_epoch_position_parallax(epoch):
    src = np.array([0.0, 0.0, 1.0])
    x = epoch_position(src, epoch)

    assert x == approx([0.0, 0.0, 0.0])
