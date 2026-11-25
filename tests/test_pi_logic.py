"""Checks the Pi driver logic with fake chips (runs on any computer)."""
import numpy as np
import pytest

from bridge_shm.pi.hx711 import HX711, counts_to_microstrain, robust_mean
from bridge_shm.pi.i2c_sensors import ADS1115, BME280, MPU6050
from bridge_shm.pi.audio import EventDetector


class FakeLG:
    """Pretends to be lgpio + an HX711 that outputs `value`."""
    def __init__(self, value):
        self.set(value)
    def set(self, value):
        v = value & 0xFFFFFF
        self.bits = [(v >> (23 - i)) & 1 for i in range(24)]
        self.edges = 0
    def gpio_claim_input(self, h, p): pass
    def gpio_claim_output(self, h, p, level): pass
    def gpio_write(self, h, p, level):
        if level == 1:
            self.edges += 1
    def gpio_read(self, h, p):
        if self.edges == 0 or self.edges >= 25:      # idle / ready
            self.edges = 0 if self.edges >= 25 else self.edges
            return 0
        return self.bits[self.edges - 1]


@pytest.mark.parametrize("value", [0, 1, 12345, -12345, 8388607, -8388608])
def test_hx711_decodes_24bit_twos_complement(value):
    lg = FakeLG(value)
    hx = HX711(lg, 0, 5, 6)
    lg.set(value)
    assert hx.read_raw() == value


def test_microstrain_conversion_roundtrip():
    eps = 1000e-6
    counts = eps * 2.0 / 4 / (0.5 / 128) * (1 << 23)
    assert counts_to_microstrain(counts, 2.0) == pytest.approx(1000, rel=1e-9)


def test_robust_mean_ignores_glitch():
    assert robust_mean([100, 101, 99, 100, 5_000_000]) == pytest.approx(100, abs=1)


def test_bme280_temperature_matches_datasheet_example():
    b = BME280.__new__(BME280)
    b.T1, b.T2, b.T3 = 27504, 26435, -1000
    b.H1, b.H2, b.H3, b.H4, b.H5, b.H6 = 75, 362, 0, 313, 50, 30
    t, h = b.compensate(519888, 30000)
    assert t == pytest.approx(25.08, abs=0.01)
    assert 0 <= h <= 100


def test_ads1115_config_word():
    assert ADS1115.config_word(0) == 0xC3E3          # AIN0, +/-4.096 V, single shot, 860 SPS
    assert ADS1115.config_word(1) == 0xD3E3


def test_mpu6050_fifo_byte_order():
    raw = np.array([16384, -16384, 0], dtype=">i2").tobytes()
    g = np.frombuffer(raw, dtype=">i2").reshape(-1, 3).astype(float) / MPU6050.LSB_PER_G
    assert g.tolist() == [[1.0, -1.0, 0.0]]


def test_crackle_detector_counts_bursts():
    fs, rng = 48000, np.random.default_rng(0)
    det = EventDetector(fs)
    det.set_noise(rng.normal(0, 0.001, fs), k=6)
    x = rng.normal(0, 0.001, 2 * fs)
    for t in (0.3, 0.9, 1.5):                       # three short "crackles"
        i = int(t * fs)
        n = np.arange(200)
        x[i:i + 200] += 0.05 * np.exp(-n / 40) * np.sin(2 * np.pi * 6000 * n / fs)
    for chunk in np.array_split(x, 50):             # arrives in blocks, like the sound card
        det.process(chunk)
    tab = det.table()
    assert len(tab) == 3
    assert np.allclose(tab.t_s, [0.3, 0.9, 1.5], atol=0.005)


def test_ads1115_differential_config_word():
    # A0-A1, +/-0.256 V, single shot, 64 SPS, comparator off
    assert ADS1115.config_word("A0-A1", 0.256, 64) == 0x8B63
