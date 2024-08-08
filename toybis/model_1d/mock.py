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


def generate_mock_observation(n_exposure, n_calib, seed=42):
    ''' Generate a mock exposure table '''
    rng = np.random.default_rng(seed=42)

    exposure_id = jnp.arange(n_exposure, dtype='int')
    calib_id = jnp.arange(n_calib, dtype='int')

    cx = rng.choice(calib_id, size=exposure_id.size)
    ep = jnp.linspace(-5, 5, exposure_id.size)
    pt = rng.uniform(-2.0, 2.0, size=exposure_id.shape)
    st = rng.gamma(50.0, 0.15 / 50.0, size=exposure_id.shape)

    exp = jnp.stack([exposure_id, cx, ep, pt, st]).T

    c0 = rng.normal(0.0, 0.1, size=calib_id.shape)
    c1 = rng.normal(0.0, 0.1, size=calib_id.shape)
    c2 = rng.normal(0.0, 0.1, size=calib_id.shape)

    cal = jnp.stack([calib_id, c0, c1, c2]).T

    return exp, cal


def generate_mock_observation_pair(n_exposure, n_calib, seed=42):
    ''' Generate a mock exposure table and its initial guess '''
    rng = np.random.default_rng(seed=seed)
    exp, cal = generate_mock_observation(n_exposure, n_calib, seed=seed)

    ehat = jnp.hstack([
        exp[:, 0:3],
        exp[:, 3:4] + rng.normal(0.0, 0.01, size=exp[:, 2:3].shape),
        exp[:, 4:5] * 0 + 0.15,
    ])

    chat = jnp.hstack([
        cal[:, 0:1],
        cal[:, 1:2] * 0,
        cal[:, 2:3] * 0,
        cal[:, 3:4] * 0,
    ])

    return (exp, cal), (ehat, chat)


def generate_mock_simulation(n_src, n_exp, n_cal, seed=42):
    ''' Generate a mock simulation dataset '''
    src, shat = generate_mock_source_pair(n_src, seed)
    obs, ohat = generate_mock_observation_pair(n_exp, n_cal, seed)

    return (src, *obs), (shat, *ohat)


def generate_mock_reference(src, frac=0.1, seed=42, weight=1e8):
    rng = np.random.default_rng(seed=seed)

    ref_flag = rng.binomial(1, frac, size=src.shape[0])

    x0 = src[:, 1]
    mu = src[:, 2]
    plx = src[:, 3]
    sig_x0  = np.random.gamma(1000, 0.0001 / 1000, size=src.shape[0])
    sig_mu  = np.random.gamma(1000, 0.0001 / 1000, size=src.shape[0])
    sig_plx = np.random.gamma(1000, 0.0001 / 1000, size=src.shape[0])

    ref = jnp.stack([
        src[:, 0],
        np.where(ref_flag, rng.normal(x0, sig_x0), np.random.normal(x0, 0.01)),
        np.where(ref_flag, rng.normal(mu, sig_mu), 0.0),
        np.where(ref_flag, rng.normal(plx, sig_plx), 0.0),
        np.where(ref_flag, sig_x0, weight),
        np.where(ref_flag, sig_mu, weight),
        np.where(ref_flag, sig_plx, weight),
    ]).T

    return ref
