#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .position_ideal_plane import position_ideal_plane

__all__ = (
    'position_focal_plane',
)


def position_focal_plane(s, e, c, t):
    ''' Position on the actual focal plane

    Arguments:
      s: source parameters
      e: exposure parameters
      c: calibration parameters
          c[0]: offset
          c[1]: linear slope
          c[2]: quadratic

    Returns:
      The position on the focal plane with distortion.
    '''
    eta = position_ideal_plane(s, e, t)
    return eta + c[0] + c[1] * eta + 0.5 * c[2] * eta**2
