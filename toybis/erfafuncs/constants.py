#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Define the constants used in the ERFA library
'''
import astropy.units as u

__all__ = [
  'ERFA_DJ00',
  'ERFA_DJM0',
  'ERFA_DJM00',
  'ERFA_SRS',
  'ERFA_DAS2R',
]


# Reference epoch (J2000.0), Julian Date
ERFA_DJ00 = 2451545.0

# Julian Date of Modified Julian Date zero
ERFA_DJM0 = 2400000.5

# Reference epoch (J2000.0), Modified Julian Date
ERFA_DJM00 = ERFA_DJ00 - ERFA_DJM0

# Schwarzschild radius of the Sun (au)
ERFA_SRS = 1.97412574336e-8

# arcsecond to radian conversion factor
ERFA_DAS2R = (1.0 * u.arcsec).to_value('radian')

#speed of light in m/s
ERFA_CMPS = 299792458.0