#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp
import astropy.units as u
import astropy.constants as c

from toybis.erfa_tools.propermotion import propermotion
from toybis.erfa_tools.transform import spherical_to_cartesian


@fixture
def spherical(n_lon=8, n_lat=5):
    lon, lat = np.meshgrid(
      2 * np.pi * np.linspace(0, 1, n_lon + 1)[:-1],
      np.pi / 2 * np.linspace(-1, 1, n_lat + 2)[1:-1]
    )
    return jnp.array([lon.flatten(), lat.flatten()]).T


def separation(sph0, sph1):
    ''' calculate the separation angle of two directions '''
    d_lat = sph1[:, 1] - sph0[:, 1]
    d_lon = (sph1[:, 0] - sph0[:, 0]) * np.cos(d_lat)

    return np.sqrt(d_lon**2 + d_lat**2)


def test_propermotion(spherical):
    ''' call the propermotion function '''
    pm = jnp.zeros_like(spherical)
    dt = 1.0
    p_observer = jnp.array([1, 0, 0])

    q_source = propermotion(spherical, pm, dt, p_observer)

    assert approx(q_source - spherical) == 0


def test_propermotion_latitude(spherical):
    ''' apply proper motions along with latitude '''
    pm = jnp.array([
        jnp.zeros_like(spherical[:, 0]),
        jnp.ones_like(spherical[:, 0]) * 1e-5
    ]).T
    dt = 1.0
    p_observer = jnp.array([1, 0, 0])

    q_source = propermotion(spherical, pm, dt, p_observer)

    p_source = spherical_to_cartesian(spherical)

    d_proj = p_source @ p_observer.reshape((3, 1)) * u.au
    t_obs = (dt * u.yr + d_proj / c.c).to_value('yr').ravel()

    v_pm = separation(q_source, spherical) / t_obs.ravel()

    assert approx(v_pm) == 1e-5


def test_propermotion_longitude(spherical):
    ''' apply proper motions along with longitude '''
    pm = jnp.array([
        jnp.ones_like(spherical[:, 0]) * 1e-5,
        jnp.zeros_like(spherical[:, 0])
    ]).T
    dt = 1.0
    p_observer = jnp.array([1, 0, 0])

    q_source = propermotion(spherical, pm, dt, p_observer)

    p_source = spherical_to_cartesian(spherical)

    d_proj = p_source @ p_observer.reshape((3, 1)) * u.au
    t_obs = (dt * u.yr + d_proj / c.c).to_value('yr').ravel()

    v_pm = separation(q_source, spherical) / t_obs.ravel()

    assert approx(v_pm) == 1e-5
