#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' toybis: JASMINE parameter optimization toy model '''

import jax
jax.config.update('jax_enable_x64', True)


from .version import version as __version__

__all__ = (
  '__version__',
)
