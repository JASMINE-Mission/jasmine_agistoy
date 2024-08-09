#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import fixture
import numpy as np

from toybis.model_1d.mock import generate_mock_simulation
from toybis.model_1d.mock import generate_mock_simulation
from toybis.model_1d.mock import generate_mock_reference
from toybis.model_1d.estimate import estimate
from toybis.model_1d.update_source import update_source
from toybis.model_1d.update_exposure import update_exposure
from toybis.model_1d.update_calibration import update_calibration


@fixture
def sim(n_source=10, n_exposure=100, n_calib=3):
    return generate_mock_simulation(n_source, n_exposure, n_calib)


def test_integrate(sim):
    src, exp, cal = sim[0]
    shat, ehat, chat = sim[1]

    noise = 0.001 * 0.01 / np.sqrt(src[:, 5])
    obs = estimate(src, exp, cal, noise=noise)

    ref = generate_mock_reference(src, frac=0.2)

    cx = chat
    ex = ehat
    sx = shat

    z0 = estimate(sx, ex, cx)
    dz0 = np.sqrt(np.mean((obs[:, 3] - z0[:, 3]) ** 2))
    print(f'# O - C = {dz0:.3e}')

    for n in range(5):
        cx = update_calibration(obs, ref, sx, ex, cx)
        ex = update_exposure(obs, ref, sx, ex, cx)
        sx = update_source(obs, ref, sx, ex, cx)

        z = estimate(sx, ex, cx)
        dz = np.sqrt(np.mean((obs[:, 3] - z[:, 3]) ** 2))
        print(f'# O - C = {dz:.3e}')

    dc = np.sqrt(np.mean((cx - cal)[:, 1:] ** 2))
    Dc = np.sqrt(np.mean((chat - cal)[:, 1:] ** 2))
    de = np.sqrt(np.mean((ex - exp)[:, 3:] ** 2))
    De = np.sqrt(np.mean((ehat - exp)[:, 3:] ** 2))
    ds = np.sqrt(np.mean((sx - src)[:, 1:] ** 2))
    Ds = np.sqrt(np.mean((shat - src)[:, 1:] ** 2))

    for n in range(cal.shape[0]):
        print(f'cal[{n}]: {cx[n][1:]} / {cal[n][1:]}')

    print(f'scale ratio: {ex[:, 4] / exp[:, 4]}')

    print(f'# cal {Dc:.3e} => {dc:.3e}')
    print(f'# exp {De:.3e} => {de:.3e}')
    print(f'# src {Ds:.3e} => {ds:.3e}')

    assert dz < dz0
    assert False
