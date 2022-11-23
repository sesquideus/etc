import numpy as np
import scipy as sp
import astropy
import hmbp
from astropy import units as u


class ExposureTimeCalculator():
    """ A very simple exposure time calculator """

    def __init__(self, *, dark: float=0 * u.ph / u.s, readout: float=0 * u.ph, seeing: float=1 * u.arcsec):
        """
            Parameters
            ----------
            dark : astropy.Quantity
                dark current (photons / s)
            readout : astropy.Quantity
                readout noise (photons)
            seeing : astropy.Quantity
                seeing during the observation (arcsec)
        """
        try:
            _ = dark + 1 * u.ph / u.s
            _ = readout + 1 * u.ph
            _ = seeing + 1 * u.arcsec
        except u.core.UnitConversionError as exc:
            raise TypeError(f"Incompatible units: ") from exc

        self.dark = dark
        self.readout = readout
        self.seeing = seeing

        # Cache background flux to avoid repeated downloads
        self.skycalc = None

    def __call__(self, target_snr: float, dit: u.Quantity, ndit: int=1, *, filter_name="K", airmass: float=1.0, pwv: float=0.5) -> float:
        """
            Find the maximum magnitude for which required SNR is reached
            The optimized function is fairly complex: a simple bisection method is used

            Parameters
            ----------
            target_snr : float
                desired signal-to-noise ratio during the observation
            dit : astropy.Quantity
                detector integration time
            ndit : int
                number of detector integrations
            filter_name : str, optional
            airmass : float, optional
            pwv : float, optional
                pressure of water vapour, in mm

            Returns
            -------
            mag : u.Quantity
                limiting magnitude for the observation
        """

        return sp.optimize.bisect(
            lambda m: self.calculate_snr(m, dit, ndit, filter_name=filter_name, airmass=airmass, pwv=pwv) - target_snr,
            0, 30, xtol=1e-6
        )

    def calculate_snr(self, magnitude: u.Quantity, dit: u.Quantity, ndit: int=1, *, filter_name="K", airmass: float=1.0, pwv: float=0.5) -> float:
        """
            Calculate the SNR using cached skycalc data

            Telescope collecting area and pixel size are hardcoded for this exercise
            We also do not consider quantum efficiency of the sensor, so everything is processed as photon counts

            Parameters
            ----------
            magnitude : u.Quantity
            dit : u.Quantity
                detector integration time
            ndit : int
                number of detector integrations
            filter_name : str, optional
            airmass : float, optional
            pwv : float, optional
                pressure of water vapour, in mm

            Returns
            -------
            snr : float
                signal-to-noise ratio
        """
        if self.skycalc is None:
            self.skycalc = hmbp.in_skycalc_background(filter_name, airmass=airmass, pwv=pwv)

        obj_flux = hmbp.for_flux_in_filter(filter_name, magnitude, instrument="HAWKI", observatory="Paranal")
        bkg_flux = self.skycalc
        pixels = np.pi * (self.seeing / (0.106 * u.arcsec))**2      # hardcoded pixel size (probably could be fetched from somewhere)
        area = np.pi * (4.1 * u.m)**2                               # hardcoded collecting area of VLTI UT4

        signal      = self.signal(obj_flux, dit, area)
        background  = self.noise_background_squared(bkg_flux, pixels, dit, area)
        dark        = self.noise_dark_squared(pixels, dit, ndit)
        readout     = self.noise_readout_squared(pixels)

        return (signal / np.sqrt(signal + background + dark + readout)).value

    @staticmethod
    def signal(flux: u.Quantity, dit: u.Quantity, area: u.Quantity) -> u.Quantity:
        return flux * dit * area

    @staticmethod
    def noise_background_squared(background_flux: u.Quantity, pixels: float, dit: u.Quantity, area: u.Quantity) -> u.Quantity:
        return background_flux * pixels * dit * area

    def noise_dark_squared(self, pixels: float, dit: u.Quantity, ndit: int=1) -> u.Quantity:
        return self.dark * pixels * ndit * dit

    def noise_readout_squared(self, pixels):
        return self.readout * pixels

