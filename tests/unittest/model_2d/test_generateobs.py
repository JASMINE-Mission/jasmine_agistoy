#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.model_2d.generateobs import generate_mock_obs_simple


@fixture
def src():
    source_id = jnp.arange(2)
    alpha0 = np.linspace(-1,1,len(source_id))
    delta0 = np.linspace(-1,1,len(source_id))
    return np.stack([source_id, alpha0, delta0]).T


@fixture
def att():
    exposure_id = jnp.arange(3)
    ep = jnp.linspace(0,3, exposure_id.size)
    pt_ra = np.linspace(-1,1,len(exposure_id))
    pt_dec = np.linspace(-1,1,len(exposure_id))
    return np.stack([exposure_id, ep, pt_ra,pt_dec]).T

@fixture
def cal():
    return None

@fixture
def noise():
    return 0

def test_generate_mock_obs_simple(src,att,cal,noise):
    foo = lambda s,a,c,t: jnp.asarray((jnp.array(s[0]+a[0]),jnp.array(s[1]+a[1])))
    obs = generate_mock_obs_simple(src,att,cal,foo,noise)
    expcted_values = np.array([[ 0.,  0.,  0., -2.],
                            [ 0.,  0.,  1., -2.],
                            [ 0.,  1.,  0., -1.],
                            [ 0.,  1.,  1., -1.],
                            [ 0.,  2.,  0.,  0.],
                            [ 0.,  2.,  1.,  0.],
                            [ 1.,  0.,  0.,  0.],
                            [ 1.,  0.,  1.,  0.],
                            [ 1.,  1.,  0.,  1.],
                            [ 1.,  1.,  1.,  1.],
                            [ 1.,  2.,  0.,  2.],
                            [ 1.,  2.,  1.,  2.]])
    
    assert all(np.array([[int(b) == int(expcted_values[i,j]) for j,b in enumerate(a)] for i,a in enumerate(obs[0])]).flatten())


