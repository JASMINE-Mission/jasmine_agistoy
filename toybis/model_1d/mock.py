#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Generate mock observations for testing '''

import numpy as np
import jax.numpy as jnp
import jax


def generate_mock_source(n_source, seed=42):
    ''' Generate a mock source table '''
    rng = np.random.default_rng(seed=seed)

    source_id = jnp.arange(n_source, dtype='int')

    x0 = rng.uniform(-1.0, 1.0, size=source_id.shape)
    mu = rng.normal(0, 0.05 / 3600, size=source_id.shape)
    plx = rng.gamma(3, 0.05 / 3600 / 3, size=source_id.shape)
    mag = rng.uniform(10, 15, size=source_id.shape)
    flx = 1e4 * 10 ** (-0.4 * mag)

    return jnp.stack([source_id, x0, mu, plx, flx]).T


def generate_mock_source_pair(n_source, seed=42):
    ''' Generate a mock source table and its initial guess '''
    rng = np.random.default_rng(seed=seed)
    src = generate_mock_source(n_source, seed=seed)

    shat = jnp.hstack([
        src[:, 0:1],
        src[:, 1:2] + rng.normal(0.0, 0.05, size=src[:, 1:2].shape),
        src[:, 2:3] * 0,
        src[:, 3:4] * 0,
        src[:, 4:5] * rng.normal(1.0, 0.05, size=src[:, 1:2].shape),
    ])

    return src, shat


def generate_mock_exposure(n_exposure, seed=42):
    ''' Generate a mock exposure table '''
    rng = np.random.default_rng(seed=42)

    exposure_id = jnp.arange(n_exposure, dtype='int')

    ep = jnp.linspace(-5, 5, exposure_id.size)
    pt = rng.uniform(-2.0, 2.0, size=exposure_id.shape)
    st = rng.gamma(50.0, 0.15 / 50.0, size=exposure_id.shape)

    return jnp.stack([exposure_id, ep, pt, st]).T


def generate_mock_exposure_pair(n_exposure, seed=42):
    ''' Generate a mock exposure table and its initial guess '''
    rng = np.random.default_rng(seed=seed)
    exp = generate_mock_exposure(n_exposure, seed=seed)

    ehat = jnp.hstack([
        exp[:, 0:2],
        exp[:, 2:3] + rng.normal(0.0, 0.01, size=exp[:, 2:3].shape),
        exp[:, 3:4] * 0 + 0.15,
    ])

    return exp, ehat


def generate_mock_calibration(n_calibration, seed=42):
    ''' Generate a mock calibration table '''
    rng = np.random.default_rng(seed=42)

    calib_id = jnp.arange(n_calibration, dtype='int')

    c0 = rng.normal(0.0, 0.1, size=calib_id.shape)
    c1 = rng.normal(0.0, 0.1, size=calib_id.shape)
    c2 = rng.normal(0.0, 0.1, size=calib_id.shape)

    return jnp.stack([c0, c1, c2]).T


def generate_mock_calibration_pair(n_calibration, seed=42):
    ''' Generate a mock calibration table and its initial guess '''
    rng = np.random.default_rng(seed=42)

    cal = generate_mock_calibration(n_calibration, seed=seed)

    chat = jnp.zeros(shape=cal.shape)

    return cal, chat


def generate_mock_simulation(n_src, n_exp, n_cal, seed=42):
    ''' Generate a mock simulation dataset '''
    src, shat = generate_mock_source_pair(n_src, seed)
    exp, ehat = generate_mock_exposure_pair(n_exp, seed)
    cal, chat = generate_mock_calibration_pair(n_cal, seed)

    return (src, exp, cal), (shat, ehat, chat)
