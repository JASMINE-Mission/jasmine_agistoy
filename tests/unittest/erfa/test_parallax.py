#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.erfa.parallax import parallax


@fixture
def p_source(n_lon=8, n_lat=5):
    lon, lat = np.meshgrid(
      2 * np.pi * np.linspace(0, 1, n_lon + 1)[:-1],
      np.pi / 2 * np.linspace(-1, 1, n_lat + 2)[1:-1]
    )
    lon, lat = lon.flatten(), lat.flatten()
    return jnp.array([
        [np.cos(r) * np.cos(d), np.sin(r) * np.cos(d), np.sin(d)]
        for r, d in zip(lon, lat)])


@fixture
def p_observer(n_obs=16, L=1.0):
    theta = np.linspace(0, np.pi, n_obs)
    return jnp.array([[L * np.cos(t), L * np.sin(t), 0] for t in theta])


def test_parallax(p_source):
    ''' call the parallax function '''
    plx = jnp.zeros(p_source.shape[0])
    p_observer = jnp.array([1, 0, 0])

    q_source = parallax(p_source, plx, p_observer)

    assert approx(q_source - p_source) == 0


def test_parallax_amplitude(p_observer):
    ''' check the amplitude of parallaxes at the north pole '''
    p_source = jnp.array([[0, 0, 1]])
    plx = jnp.array([25.0e-4])

    for p_obs in p_observer:
        q_source = parallax(p_source, plx, p_obs)

        r = np.sqrt(q_source[:, 0]**2 + q_source[:, 1]**2)
        p = np.asin(r) / np.pi * 180. * 3600.
        assert approx(p - plx[0]) == 0.0
