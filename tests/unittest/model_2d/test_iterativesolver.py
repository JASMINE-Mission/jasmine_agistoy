#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx, fixture
import numpy as np
import jax.numpy as jnp

from toybis.model_2d.iterativesolver import _Jacobian_autodiff

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
    assert ans[0] == true_ans
    assert ans[1] == true_ans

