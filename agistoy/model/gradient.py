#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
from .estimate import zeta

__all__ = (
  'dzds', 'dzde', 'dzdc',
)


dzds = jax.vmap(jax.grad(zeta, argnums=(0)), (None, 0, None, 0))
dzde = jax.vmap(jax.grad(zeta, argnums=(1)), (0, None, None, None))
_dzdc_src = jax.vmap(jax.grad(zeta, argnums=(2)), (0, None, None, None), 0)
_dzdc_exp = jax.vmap(_dzdc_src, (None, 0, None, 0), 0)
dzdc = lambda s, e, c, t: _dzdc_exp(s, e, c, t).reshape((-1, 3))
