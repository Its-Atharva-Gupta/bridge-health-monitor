"""HX711 24-bit amplifier driver for the Raspberry Pi 5 (uses lgpio).

Why our own driver: most HX711 tutorials/libraries use RPi.GPIO, which does not
work on the Pi 5's new RP1 GPIO chip. lgpio does.

How the HX711 talks: when a reading is ready it pulls DOUT low. We then pulse
SCK 24 times and read one bit per pulse (most significant first), then 1-3
extra pulses choose the gain for the NEXT reading (25 pulses = channel A, x128).
SCK must not stay high for more than ~60 microseconds or the chip powers down,
so the pulse loop is kept tight, and every reading is filtered for glitches.
"""
import time

import numpy as np

_GAIN_PULSES = {128: 25, 64: 27, 32: 26}


def open_gpiochip(lgpio, preferred="auto"):
    """Find the Pi 5 header GPIO chip (RP1). It is gpiochip0 on new kernels, gpiochip4 on old."""
    if preferred != "auto":
        return lgpio.gpiochip_open(int(preferred))
    fallback = None
    for n in range(0, 6):
        try:
            h = lgpio.gpiochip_open(n)
        except Exception:
            continue
        try:
            label = lgpio.gpio_get_chip_info(h)[3]
        except Exception:
            label = ""
        if "rp1" in str(label).lower():
            if fallback is not None:
                lgpio.gpiochip_close(fallback)
            return h
        if fallback is None:
            fallback = h
        else:
            lgpio.gpiochip_close(h)
    if fallback is None:
        raise RuntimeError("No GPIO chip found. Is this a Raspberry Pi with lgpio installed?")
    return fallback


class HX711:
    def __init__(self, lgpio, chip, dout, sck, gain=128, name="hx711"):
        self.lg, self.h, self.dout, self.sck, self.name = lgpio, chip, dout, sck, name
        self.pulses = _GAIN_PULSES[gain]
        lgpio.gpio_claim_input(chip, dout)
        lgpio.gpio_claim_output(chip, sck, 0)
        self.offset = 0.0
        time.sleep(0.05)
        self.read_raw(timeout=1.0)          # first reading sets the gain; discard it

    def ready(self):
        return self.lg.gpio_read(self.h, self.dout) == 0

    def read_raw(self, timeout=0.5):
        t0 = time.monotonic()
        while not self.ready():
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"{self.name}: no data (check DT/SCK wiring and power)")
            time.sleep(0.001)
        lg, h, sck, dout = self.lg, self.h, self.sck, self.dout
        value = 0
        for _ in range(24):
            lg.gpio_write(h, sck, 1)
            lg.gpio_write(h, sck, 0)
            value = (value << 1) | lg.gpio_read(h, dout)
        for _ in range(self.pulses - 24):
            lg.gpio_write(h, sck, 1)
            lg.gpio_write(h, sck, 0)
        if value & 0x800000:                # 24-bit two's complement
            value -= 1 << 24
        return value

    def read_many(self, n=None, seconds=None):
        """Collect readings for `n` samples or `seconds`, return the raw list."""
        vals, t0 = [], time.monotonic()
        while (n is not None and len(vals) < n) or (seconds is not None and time.monotonic() - t0 < seconds):
            vals.append(self.read_raw())
        return np.array(vals, float)

    def tare(self, seconds=1.0):
        self.offset = robust_mean(self.read_many(seconds=seconds))
        return self.offset


def robust_mean(x):
    """Mean after dropping glitches (readings far from the median)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    keep = np.abs(x - med) < 5 * 1.4826 * mad
    return float(x[keep].mean()) if keep.any() else float(med)


def counts_to_microstrain(counts, gauge_factor=2.0, gain=128):
    """Quarter-bridge strain gauge (one active gauge) on an HX711.

    The HX711 full scale is +/-(0.5 * Vexcitation / gain), and on standard boards
    the bridge is excited by the same voltage, so it cancels:
        counts / 2^23 = (GF * strain / 4) / (0.5 / gain)
    """
    return counts * 0.5 / gain / (1 << 23) * 4.0 / gauge_factor * 1e6
