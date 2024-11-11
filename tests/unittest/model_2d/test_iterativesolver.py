#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.model_2d.iterativesolver import _Jacobian_autodiff,update_source_inner,update_source,update_attitude_inner,update_attitude,bloc_iteration,iterate_attitude,iterate_source

@fixture
def argnum():
    return 0

@fixture
def axis():
    return 0

@fixture
def input():
    return jnp.stack((jnp.array([1., 2., 3.]),jnp.array([1., 2., 3.])))

@fixture
def src():
    return np.array([0, -1.])

@fixture
def src_all():
    return np.array([[0, -1.],
                     [1, -2.]])

@fixture
def att():
    return np.array([0,0.1,2])

@fixture
def att_all():
    return np.array([[0,0.1,2],
                      [1,0.2,2],
                      [2,0.3,4]])

@fixture
def cal():
    return None

@fixture
def obs():
    return np.array([[0,0,2],
                         [0,1,2],
                         [0,2,4]])

@fixture
def ephemeris():
    return np.array([1,0,0,0,0,0])

@fixture
def ephemeris_all():
    return np.array([[1,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [1,0,0,0,0,0]])

@fixture
def _min_nobs():
    return 1     

def test_Jacobian_autodiff(argnum,axis,input):
    def foo(x):
        return jnp.asarray(
            [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
    jac = _Jacobian_autodiff(foo,argnum,axis)
    ans = jac(input)
    true_ans = jnp.array([[ 1. ,      0.    ,   0.     ],
                            [ 0.   ,    0.  ,     5.     ],
                            [ 0.    ,  16.  ,    -2.     ],
                            [ 1.6209 ,  0. ,      0.84147]])
    assert all(np.array([[int(b) == int(true_ans[i,j]) for j,b in enumerate(a)] for i,a in enumerate(ans[0])]).flatten())
    assert all(np.array([[int(b) == int(true_ans[i,j]) for j,b in enumerate(a)] for i,a in enumerate(ans[1])]).flatten())

def test_update_source_inner(src,att_all,cal,ephemeris_all,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans = update_source_inner(src,att_all,cal,ephemeris_all,obs,None,None,None,foo,_min_nobs)

    assert ans.item() == approx(0.)

def test_update_source(src_all,att_all,ephemeris_all,cal,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans = update_source(src_all,att_all,cal,ephemeris_all,obs,None,None,None,foo,_min_nobs)

    assert ans[0].item() == approx(0.)
    assert ans[1].item() == approx(-2.)

def test_iterate_source(src_all,att_all,cal,ephemeris_all,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans = iterate_source(src_all,att_all,cal,obs,None,None,None,foo,1,_min_nobs)

    assert ans[0][-1].item() == approx(0.)
    assert ans[1][-1].item() == approx(-2.)


def test_update_attitude_inner(src_all,att,cal,ephemeris,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans = update_attitude_inner(src_all,att,cal,ephemeris,obs,foo,_min_nobs)

    assert ans.item() == approx(3.)

def test_update_attitude(src_all,att_all,cal,ephemeris_all,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans = update_attitude(src_all,att_all,cal,ephemeris_all,obs,foo,_min_nobs)

    assert ans[0].item() == approx(3.)
    assert ans[1].item() == approx(3.)
    assert ans[2].item() == approx(5.)

def test_iterate_attitude(src_all,att_all,cal,ephemeris_all,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans = iterate_attitude(src_all,att_all,cal,ephemeris_all,obs,foo,1,_min_nobs)

    assert ans[0][-1].item() == approx(3.)
    assert ans[1][-1].item() == approx(3.)
    assert ans[2][-1].item() == approx(5.)

def test_bloc_iteration(src_all,att_all,cal,ephemeris_all,obs,_min_nobs):
    foo = lambda s,a,c,t,e: s+a 
    ans1,ans2,ans3 = bloc_iteration(src_all,att_all,cal,ephemeris_all,obs,foo,1,1,1,None,None,None,_min_nobs,_min_nobs)

    assert ans1[0][-1].item() == approx(-1.)
    assert ans1[1][-1].item() == approx(-2.)
    assert ans2[0][-1].item() == approx(3.)
    assert ans2[1][-1].item() == approx(3.)
    assert ans2[2][-1].item() == approx(5.)