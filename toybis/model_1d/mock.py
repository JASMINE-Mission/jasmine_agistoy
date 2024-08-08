#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Generate mock observations for testing '''

import numpy as np
import jax.numpy as jnp
import jax


def generate_mock_source(n_source, seed=42):
    rng = np.random.default_rng(seed=seed)

    source_id = jnp.arange(n_source, dtype='int')

    x0 = rng.uniform(-1.0, 1.0, size=source_id.shape)
    mu = rng.normal(0, 0.05 / 3600, size=source_id.shape)
    plx = rng.gamma(3, 0.05 / 3600 / 3, size=source_id.shape)
    mag = rng.uniform(10, 15, size=source_id.shape)
    flx = 1e4 * 10 ** (-0.4 * mag)

    return jnp.stack([source_id, x0, mu, plx, flx]).T


def generate_mock_exposure(n_exposure, seed=42):
    rng = np.random.default_rng(seed=42)

    exposure_id = jnp.arange(n_exposure, dtype='int')

    ep = jnp.linespace(-5, 5, exposure_id.size)
    pt = rng.uniform(-2.0, 2.0, size=exposure_id.shape)
    st = rng.uniform(50.0, 0.15 / 50.0, size=exposure_id.shape)

    return jnp.stack([exposure_id, ep, pt, st]).T


def generate_mock_calibration(n_calibration, seed=12):
    rng = np.random.default_rng(seed=42)

    calib_id = jnp.arange(n_calibration, dtype='int')

    c0 = rng.normal(0.0, 0.1, size=calib_id.shape)
    c1 = rng.normal(0.0, 0.1, size=calib_id.shape)
    c2 = rng.normal(0.0, 0.1, size=calib_id.shape)

    return jnp.stack([c0, c1, c2]).T
