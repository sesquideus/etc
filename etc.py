import numpy as np
import scipy as sp
import astropy
import hmbp
from astropy import units as u


class ExposureTimeCalculator():
    """ A very simple exposure time calculator """

    def __init__(self, *, dark: float=0, readout: float=0, seeing: float=1 * u.arcsec):
        self.dark = dark
        self.readout = readout
        self.seeing = seeing

        # Cache background to avoid repeated downloads
        self.skycalc = None

    def __call__(self, target_snr: float, dit: float, ndit: int=1, *, filter_name="K", airmass: float=1.0, pwv: float=0.5) -> float:
        """
            Find the maximum magnitude for which required SNR is reached
            The optimized function is complex so a simple bisection method is used
        """

        return sp.optimize.bisect(
            lambda m: self.calculate_snr(m, dit, ndit, filter_name=filter_name, airmass=airmass, pwv=pwv) - target_snr,
            0, 30, xtol=1e-9
        )

    def calculate_snr(self, magnitude: float, dit: float, ndit: int=1, *, filter_name="K", airmass: float=1.0, pwv: float=0.5) -> float:
        """ Calculate the SNR using cached skycalc data """
        if self.skycalc is None:
            self.skycalc = hmbp.in_skycalc_background(filter_name, airmass=airmass, pwv=pwv)

        obj_flux = hmbp.for_flux_in_filter(filter_name, magnitude, instrument="HAWKI", observatory="Paranal")
        bkg_flux = self.skycalc
        pixels = np.pi * (self.seeing / (0.106 * u.arcsec))**2
        area = np.pi * (4.1 * u.m)**2               # hardcoded collecting area of VLTI UT4

        signal      = obj_flux * dit * area
        background  = bkg_flux * pixels * dit * area
        dark        = pixels * self.dark * dit
        readout     = pixels * self.readout

        print(signal, background, dark, readout)

        return (signal / np.sqrt(signal + background + dark + readout)).value
