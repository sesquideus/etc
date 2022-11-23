#!/usr/bin/env python

import argparse
import hmbp
import astropy.units as u

from main import ExposureTimeCalculator


etc = ExposureTimeCalculator(dark=0.01 * u.ph / u.s, readout=5.0 * u.ph, seeing=0.6 * u.arcsec)

desired_snr = 5
dit = 3600 * u.second
ndit = 1
filter_name = "Ks"
pwv = 2.5

print(f"Desired SNR {desired_snr:3d}, ndit = {ndit:4d}, dit = {dit}")
print(f"Limiting magnitude is {etc(desired_snr, dit, ndit, filter_name=filter_name, pwv=pwv):.3f}")
