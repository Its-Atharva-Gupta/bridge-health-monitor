"""Checks the beam physics, feature extraction and two-MPU recording."""
import numpy as np
import pandas as pd
import pytest

from bridge_shm import beam, config as C
from bridge_shm.features import tap_features
from bridge_shm.pi.i2c_sensors import MPU6050, record_together


def test_beam_matches_textbook_formulas():
    EI, F, L = 5e5, 10.0, C.SPAN_MM
    x, w, slope, _ = beam.static(F, L / 2, (), EI)
    assert w[np.argmin(abs(x - L / 2))] == pytest.approx(F * L ** 3 / (48 * EI), rel=1e-4)
    # slope at the MPUs, and the sag-from-tilt factor the rig uses
    th = np.interp(C.MPU_X_MM, x, slope)
    assert th == pytest.approx(F * (L ** 2 - 4 * C.MPU_X_MM ** 2) / (16 * EI), rel=1e-3)
    assert w.max() / th == pytest.approx(C.SAG_PER_TILT_RAD, rel=1e-3)
    assert np.interp(L - C.MPU_X_MM, x, slope) == pytest.approx(-th, rel=1e-3)   # symmetric


def test_damage_on_left_tilts_left_mpu_more():
    zones = ((0.33 * C.SPAN_MM, 0.41 * C.SPAN_MM, 0.6),)   # between the left MPU and the centre
    x, _, s, _ = beam.static(1.0, C.LOAD_X_MM, zones)
    _, _, s0, _ = beam.static(1.0, C.LOAD_X_MM, ())
    left = np.interp(C.MPU_X_MM, x, s) / np.interp(C.MPU_X_MM, x, s0)
    right = np.interp(C.SPAN_MM - C.MPU_X_MM, x, s) / np.interp(C.SPAN_MM - C.MPU_X_MM, x, s0)
    assert left > right > 1.0


def test_modes_uniform_beam_and_damage_lowers_frequency():
    o1, o2, *_ = beam.modes((), sensor_mass_ratio=0.0)
    assert o2 / o1 == pytest.approx(4.0, rel=0.01)          # simply supported: f2 = 4 f1
    d1, d2, *_ = beam.modes(((95, 130, 0.5),))
    h1, h2, *_ = beam.modes(())
    assert d1 < h1 and d2 < h2


def ring(f1, f2, z, fs=1000, n=3000, t0=0.3, w2=0.5, seed=0):
    t = np.arange(n) / fs
    tt = np.clip(t - t0, 0, None)
    a = (np.exp(-z * 2 * np.pi * f1 * tt) * np.sin(2 * np.pi * f1 * tt)
         + w2 * np.exp(-0.02 * 2 * np.pi * f2 * tt) * np.sin(2 * np.pi * f2 * tt))
    a[t < t0] = 0
    return t, a + np.random.default_rng(seed).normal(0, 0.01, n)


@pytest.mark.parametrize("w2", [0.5, 2.0])   # mode 2 weaker, or TALLER than mode 1
def test_tap_features_find_both_modes_and_damping(w2):
    t, a = ring(48.0, 175.0, 0.03, w2=w2)
    _, b = ring(48.0, 175.0, 0.03, w2=-w2, seed=1)            # 2nd MPU: mode 2 opposite sign
    f = tap_features(pd.DataFrame(dict(t_s=t, accel_g=a, accel2_g=b)))
    assert f["freq_hz"] == pytest.approx(48.0, abs=0.3)
    assert f["freq2_hz"] == pytest.approx(175.0, abs=1.0)
    assert f["damping_ratio"] == pytest.approx(0.03, rel=0.1)  # quickly decaying ring is OK


def test_tap_features_one_mpu_only():
    t, a = ring(50.0, 180.0, 0.015)
    f = tap_features(pd.DataFrame(dict(t_s=t, accel_g=a)))
    assert f["freq_hz"] == pytest.approx(50.0, abs=0.3)


class FakeMPUBus:
    """Two MPU6050 FIFOs on one bus, each producing a known ramp."""
    def __init__(self):
        self.fifo = {0x68: bytearray(), 0x69: bytearray()}
        self.n = {0x68: 0, 0x69: 100}
    def write_byte_data(self, a, reg, v): pass
    def read_byte_data(self, a, reg):
        return 0x00 if reg == MPU6050.INT_STATUS else 0x68
    def read_i2c_block_data(self, a, reg, n):
        for _ in range(7):                                    # 7 new samples since last look
            v = self.n[a]; self.n[a] += 1
            self.fifo[a] += np.array([v, -v, 2 * v], ">i2").tobytes()
        c = len(self.fifo[a]); return [c >> 8, c & 0xFF]
    def i2c_rdwr(self, w, r):
        a = r.addr
        data, self.fifo[a] = self.fifo[a][:r.len], self.fifo[a][r.len:]
        for i, byte in enumerate(data):
            r.buf[i] = bytes([byte])


def test_record_together_two_mpus(monkeypatch):
    pytest.importorskip("smbus2")
    bus = FakeMPUBus()
    m1, m2 = MPU6050.__new__(MPU6050), MPU6050.__new__(MPU6050)
    m1.bus = m2.bus = bus
    m1.addr, m2.addr = 0x68, 0x69
    monkeypatch.setattr("time.sleep", lambda s: None)
    a, b = record_together([m1, m2], seconds=0.05, fs=1000)
    assert a.shape == b.shape == (50, 3)
    assert np.allclose(a[:, 0] * MPU6050.LSB_PER_G, np.arange(50))
    assert np.allclose(b[:, 2] * MPU6050.LSB_PER_G, 2 * (100 + np.arange(50)))
