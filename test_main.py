import pytest
from astropy import units as u

from .main import ExposureTimeCalculator


class TestInit():
    def test_dark_throws(self):
        with pytest.raises(TypeError):
            calc = ExposureTimeCalculator(dark=0.01)

    def test_readout_throws(self):
        with pytest.raises(TypeError):
            calc = ExposureTimeCalculator(readout=5 * u.m)

    def test_seeing_throws(self):
        with pytest.raises(TypeError):
            calc = ExposureTimeCalculator(readout=5 * u.s)


class TestSignal():
    calc = ExposureTimeCalculator(dark=0.01 * u.ph / u.s, readout=5.0 * u.ph, seeing=0.6 * u.arcsec)

    def test_signal_value(self):
        signal = self.calc.signal(1e8 * u.ph / (u.m**2) / u.s, 50 * u.s, 1 * u.m**2)
        assert signal.value == pytest.approx(5e9, rel=1e-6)

    def test_signal_units(self):
        signal = self.calc.signal(1e8 * u.ph / (u.m**2) / u.s, 50 * u.s, 1 * u.m**2)
        try:
            _ = signal + 1 * u.ph
        except u.core.UnitConversionError:
            assert False


class TestNoise():
    calc = ExposureTimeCalculator(dark=0.01 * u.ph / u.s, readout=5.0 * u.ph, seeing=0.6 * u.arcsec)

    def test_readout_value(self):
        readout = self.calc.noise_readout_squared(5)
        assert readout.value == pytest.approx(25, rel=1e-9)

    def test_readout_dimension(self):
        try:
            _ = self.calc.noise_readout_squared(50) + 1 * u.ph
        except u.core.UnitConversionError:
            assert False

    def test_dark_value(self):
        dark = self.calc.noise_dark_squared(50, 7 * u.s)
        assert dark.value == pytest.approx(3.5, rel=1e-6)

    def test_dark_dimension(self):
        try:
            _ = self.calc.noise_dark_squared(7, 3 * u.s) + 1 * u.ph
        except u.core.UnitConversionError:
            assert False
