#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.model_2d.iterativesolver import _Jacobian_autodiff

@fixture
def foo(x):
  return jnp.asarray(
    [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])

@fixture
def argnum():
    return 0

@fixture
def axis():
    return 0

@fixture
def input():
    return jnp.stack((jnp.array([1., 2., 3.]),jnp.array([1., 2., 3.])))



def test_Jacobian_autodiff(foo,argnum,axis,input):
    jac = _Jacobian_autodiff(foo,argnum,axis)
    ans = jac(input)
    true_ans = jnp.array([[ 1. ,      0.    ,   0.     ],
                            [ 0.   ,    0.  ,     5.     ],
                            [ 0.    ,  16.  ,    -2.     ],
                            [ 1.6209 ,  0. ,      0.84147]])
    assert all(np.array([[int(b) == int(true_ans[i,j]) for j,b in enumerate(a)] for i,a in enumerate(ans[0])]).flatten())
    assert all(np.array([[int(b) == int(true_ans[i,j]) for j,b in enumerate(a)] for i,a in enumerate(ans[1])]).flatten())

