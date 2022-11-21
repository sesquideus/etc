import numpy as np
import scipy as sp
import astropy
import hmbp

from astropy import units as u


class ExposureTimeCalculator():
    """ Exposure time calculator """

    def __init__(self, *, dark: float=0, ron: float=0):
        self.dark = dark
        self.ron = ron

    def __call__(self, target_snr: float, exp_time: float, *, filter_name="K", airmass: float=1.0, pwv: float=0.5) -> float:
        """ Do the calculation. """
        return sp.optimize.bisect(lambda m: self.calculate_snr(m, exp_time, filter_name=filter_name, airmass=airmass, pwv=pwv) - target_snr, 50, -50)

    def calculate_snr(self, magnitude: float, exp_time: float, *, filter_name="K", airmass: float=1.0, pwv: float=0.5) -> float:
        obj = hmbp.for_flux_in_filter(filter_name, magnitude) * exp_time * u.second
        bkg = hmbp.in_skycalc_background(filter_name, airmass=airmass, pwv=pwv) * exp_time * u.second
        return obj / np.sqrt(np.square(obj) + np.square(self.noise_total(exp_time, obj, 1)))

    def noise_shot(self, photons: float) -> float:
        """ Shot noise, proportional to sqrt(photons) """
        return np.sqrt(photons)

    def noise_readout(self, npix: float) -> float:
        """ Readout noise, proportional to """
        return np.sqrt(npix) * self.ron

    def noise_dark(self, t: float) -> float:
        """ Dark noise, proportional to sqrt(time) """
        return np.sqrt(self.dark * t)

    def noise_total(self, exp_time: float, photons: float, readouts: int) -> float:
        return np.sqrt( \
            np.square(self.noise_shot(photons)) + \
            np.square(self.noise_readout(readouts)) + \
            np.square(self.noise_dark(exp_time)) \
        )



if __name__ == "__main__":
    print(ExposureTimeCalculator()(5, 3600))
