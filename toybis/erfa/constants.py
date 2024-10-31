#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Define the constants used in the ERFA library
'''

__all__ = [
  'ERFA_DJ00',
  'ERFA_DJM0',
  'ERFA_DJM00',
  'ERFA_SRS',
]


# Reference epoch (J2000.0), Julian Date
ERFA_DJ00 = 2451545.0

# Julian Date of Modified Julian Date zero
ERFA_DJM0 = 2400000.5

# Reference epoch (J2000.0), Modified Julian Date
ERFA_DJM00 = ERFA_DJ00 - ERFA_DJM0

# Schwarzschild radius of the Sun (au)
ERFA_SRS = 1.97412574336e-8
