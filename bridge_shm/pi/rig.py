"""The whole test rig as one object: reads every enabled sensor.

Sensor settings + calibration live in rig_config.json (created on first run).
Turn sensors on/off there with "enabled", e.g. keep the strain gauges off until
they are fitted. FakeRig behaves the same but uses the simulator, so the test
program can be tried on a laptop with no hardware.
"""
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from .. import config as C

DEFAULT_CONFIG = {
    "gpio_chip": "auto",
    "i2c_bus": 1,
    # HX711 #1: load cell under one support. Calibrated with a known weight at MIDSPAN,
    # so readings are directly "grams on the bridge" (no x2 needed).
    "load": {"enabled": True, "dout": 5, "sck": 6, "g_per_count": None},
    # HX711 #2 and #3: strain gauges (left / right of centre). Off until fitted.
    "strain_l": {"enabled": False, "dout": 17, "sck": 27, "gauge_factor": 2.0, "sign": 1},
    "strain_r": {"enabled": False, "dout": 23, "sck": 24, "gauge_factor": 2.0, "sign": 1},
    "mpu6050": {"enabled": True, "address": 0x68, "bridge_axis": "x", "vertical_axis": "z"},
    # temperature + humidity: use ONE of these
    "bme280": {"enabled": False, "address": 0x76},
    "dht": {"enabled": True},          # DHT11 or DHT22 on GPIO4 (kernel driver, see pi_setup.sh --dht)
    # sag sensor (slide pot or Hall sensor) on ADS1115 channel A0.
    # Without it, sag is estimated from the MPU6050 tilt.
    "sag": {"enabled": False, "address": 0x48, "channel": 0, "poly_mm_per_volt": None},
    "audio": {"enabled": False, "device": "USB", "samplerate": 48000,
              "threshold_x_noise": 6.0, "highpass_hz": 500.0, "keep_audio": False},
    "average_seconds": 1.0,
}

# beam theory: midspan sag / slope at the MPU position (point load at midspan)
SAG_PER_TILT_RAD = C.SPAN_MM ** 3 / (3 * (C.SPAN_MM ** 2 - 4 * C.MPU_X_MM ** 2))


def load_config(path):
    path = Path(path)
    cfg = deepcopy(DEFAULT_CONFIG)
    if path.exists():
        user = json.loads(path.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    else:
        save_config(path, cfg)
    return cfg


def save_config(path, cfg):
    Path(path).write_text(json.dumps(cfg, indent=2))


class Rig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hx = {}
        self.mpu = self.bme = self.ads = None
        if cfg.get("dht", {}).get("enabled") and cfg["bme280"]["enabled"]:
            raise RuntimeError("Enable either bme280 or dht in rig_config.json, not both.")
        if cfg.get("dht", {}).get("enabled"):
            from .i2c_sensors import DHTSysfs
            self.bme = DHTSysfs()          # same read() -> (temp, humidity) interface
        self._tilt0 = 0.0
        self._sag_v0 = 0.0
        need_gpio = any(cfg[k]["enabled"] for k in ("load", "strain_l", "strain_r"))
        if need_gpio:
            import lgpio
            from .hx711 import HX711, open_gpiochip
            self.lg = lgpio
            self.chip = open_gpiochip(lgpio, cfg["gpio_chip"])
            for k in ("load", "strain_l", "strain_r"):
                if cfg[k]["enabled"]:
                    self.hx[k] = HX711(lgpio, self.chip, cfg[k]["dout"], cfg[k]["sck"], name=k)
        if any(cfg[k]["enabled"] for k in ("mpu6050", "bme280", "sag")):
            from smbus2 import SMBus
            from .i2c_sensors import ADS1115, BME280, MPU6050
            self.bus = SMBus(cfg["i2c_bus"])
            if cfg["mpu6050"]["enabled"]:
                self.mpu = MPU6050(self.bus, cfg["mpu6050"]["address"])
            if cfg["bme280"]["enabled"]:
                self.bme = BME280(self.bus, cfg["bme280"]["address"])
            if cfg["sag"]["enabled"]:
                self.ads = ADS1115(self.bus, cfg["sag"]["address"])

    # ---------------------------------------------------------------- basics
    @property
    def has_sag_sensor(self):
        return self.ads is not None and self.cfg["sag"]["poly_mm_per_volt"] is not None

    def environment(self):
        return self.bme.read() if self.bme else (float("nan"), float("nan"))

    def _tilt_now(self, seconds):
        from .i2c_sensors import tilt_deg
        m = self.cfg["mpu6050"]
        return tilt_deg(self.mpu.average_accel(seconds), m["bridge_axis"], m["vertical_axis"])

    def _sag_volts(self, seconds):
        vals = []
        import time
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            vals.append(self.ads.read_volts(self.cfg["sag"]["channel"]))
        return float(np.median(vals))

    def zero(self, seconds=1.5):
        """Bridge empty: zero the load cell, gauges, tilt and sag."""
        for h in self.hx.values():
            h.tare(seconds)
        if self.mpu:
            self._tilt0 = self._tilt_now(seconds)
        if self.ads:
            self._sag_v0 = self._sag_volts(seconds)

    def read_step(self, seconds=None):
        """Average every sensor for `seconds` (load must be still)."""
        from .hx711 import counts_to_microstrain, robust_mean
        s = seconds or self.cfg["average_seconds"]
        row = {}
        if "load" in self.hx:
            h, g = self.hx["load"], self.cfg["load"]["g_per_count"]
            counts = robust_mean(h.read_many(seconds=s)) - h.offset
            row["load_g"] = counts * g if g else float("nan")
        else:
            row["load_g"] = float("nan")
        for side in ("l", "r"):
            k = f"strain_{side}"
            if k in self.hx:
                h, c = self.hx[k], self.cfg[k]
                counts = robust_mean(h.read_many(seconds=s)) - h.offset
                row[f"strain_{side}_ue"] = c["sign"] * counts_to_microstrain(counts, c["gauge_factor"])
            else:
                row[f"strain_{side}_ue"] = float("nan")
        row["tilt_deg"] = self._tilt_now(min(s, 0.5)) - self._tilt0 if self.mpu else float("nan")
        if self.has_sag_sensor:
            dv = self._sag_volts(min(s, 0.5)) - self._sag_v0
            row["sag_mm"] = float(np.polyval(self.cfg["sag"]["poly_mm_per_volt"], dv))
            row["sag_source"] = "sensor"
        else:
            row["sag_mm"] = abs(math.radians(row["tilt_deg"])) * SAG_PER_TILT_RAD
            row["sag_source"] = "tilt"
        return row

    def record_tap(self, seconds=3.0):
        if not self.mpu:
            raise RuntimeError("MPU6050 is disabled in rig_config.json")
        acc = self.mpu.record(seconds, C.TAP_FS_HZ)
        k = "xyz".index(self.cfg["mpu6050"]["vertical_axis"])
        return np.arange(len(acc)) / C.TAP_FS_HZ, acc[:, k]

    def crackle_listener(self):
        a = self.cfg["audio"]
        if not a["enabled"]:
            return None
        from .audio import CrackleListener
        return CrackleListener(a["device"], a["samplerate"], a["threshold_x_noise"],
                               a["highpass_hz"], a["keep_audio"])


class FakeRig:
    """Same interface, powered by the simulator. For practising on a laptop."""

    has_sag_sensor = True

    def __init__(self, cfg, stage="D0", seed=None):
        from ..simulate import simulate_cycle
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.stage = stage
        self._sim = simulate_cycle
        self._env = (self.rng.normal(27, 1), self.rng.uniform(45, 65))

    def environment(self):
        return self._env

    def zero(self, seconds=1.5):
        self.steps, self.tap, self.ae = self._sim(self.rng, self.stage, *self._env)
        self.i = 0

    def read_step(self, seconds=None):
        r = self.steps.iloc[self.i].to_dict()
        self.i += 1
        r.pop("phase")
        r["sag_source"] = "sensor"
        return r

    def record_tap(self, seconds=3.0):
        return self.tap.t_s.to_numpy(), self.tap.accel_g.to_numpy()

    def crackle_listener(self):
        ae = self.ae

        class _L:
            def start(self, quiet_seconds=1.0): pass
            def stop(self): return ae, None
        return _L()
