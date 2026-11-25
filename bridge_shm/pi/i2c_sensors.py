"""I2C sensors: MPU6050 (tilt + vibration), BME280 (temperature, humidity),
ADS1115 (analog voltages: sag sensor, flex sensors).

All share the same two wires: SDA = GPIO2 (pin 3), SCL = GPIO3 (pin 5).
Two MPU6050s: wire the SECOND one's AD0 pin to 3.3 V so its address becomes 0x69.
Check they are found with:  i2cdetect -y 1   (expect 68 and 69; 48 = ADS1115, 76 = BME280)
"""
import math
import struct
import time

import numpy as np


# --------------------------------------------------------------------- MPU6050
class MPU6050:
    PWR_MGMT_1, SMPLRT_DIV, CONFIG, ACCEL_CONFIG = 0x6B, 0x19, 0x1A, 0x1C
    FIFO_EN, USER_CTRL, FIFO_COUNT_H, FIFO_R_W = 0x23, 0x6A, 0x72, 0x74
    INT_STATUS, ACCEL_XOUT_H, WHO_AM_I = 0x3A, 0x3B, 0x75
    LSB_PER_G = 16384.0                  # +/-2 g range

    def __init__(self, bus, address=0x68):
        self.bus, self.addr = bus, address
        who = bus.read_byte_data(address, self.WHO_AM_I)
        if who not in (0x68, 0x70, 0x71, 0x72, 0x98):
            print(f"  note: MPU6050 WHO_AM_I = 0x{who:02x} (clone chip?) - continuing")
        bus.write_byte_data(address, self.PWR_MGMT_1, 0x80)       # reset
        time.sleep(0.1)
        bus.write_byte_data(address, self.PWR_MGMT_1, 0x01)       # wake, stable clock
        # filter off (accel bandwidth 260 Hz, so the ~180 Hz 2nd vibration mode isn't
        # muffled). Gyro rate is then 8 kHz, so divide by 8 for a 1 kHz sample rate.
        bus.write_byte_data(address, self.CONFIG, 0x00)
        bus.write_byte_data(address, self.SMPLRT_DIV, 0x07)
        bus.write_byte_data(address, self.ACCEL_CONFIG, 0x00)     # +/-2 g
        time.sleep(0.05)

    def read_accel_g(self):
        b = bytes(self.bus.read_i2c_block_data(self.addr, self.ACCEL_XOUT_H, 6))
        return np.array(struct.unpack(">hhh", b), float) / self.LSB_PER_G

    def average_accel(self, seconds=1.0):
        vals, t0 = [], time.monotonic()
        while time.monotonic() - t0 < seconds:
            vals.append(self.read_accel_g())
            time.sleep(0.002)
        return np.mean(vals, axis=0)

    # ---- FIFO recording (exact 1 kHz timing, done by the chip itself) ----
    def fifo_start(self):
        bus, a = self.bus, self.addr
        bus.write_byte_data(a, self.USER_CTRL, 0x04)              # reset FIFO
        bus.write_byte_data(a, self.FIFO_EN, 0x08)                # accel only
        bus.write_byte_data(a, self.USER_CTRL, 0x40)              # enable FIFO
        self._buf = bytearray()

    def fifo_drain(self):
        """Move whatever the chip has buffered into self._buf."""
        from smbus2 import i2c_msg
        bus, a = self.bus, self.addr
        hi, lo = bus.read_i2c_block_data(a, self.FIFO_COUNT_H, 2)
        count = (hi << 8) | lo
        if bus.read_byte_data(a, self.INT_STATUS) & 0x10:
            raise RuntimeError(f"MPU6050 0x{a:02x} FIFO overflowed: is the I2C bus set to 400 kHz? "
                               "(see pi_setup.sh)")
        count -= count % 6
        while count > 0:
            n = min(count, 240)
            w, r = i2c_msg.write(a, [self.FIFO_R_W]), i2c_msg.read(a, n)
            bus.i2c_rdwr(w, r)
            self._buf += bytes(r)
            count -= n

    def fifo_stop(self, n_samples):
        self.bus.write_byte_data(self.addr, self.USER_CTRL, 0x00)
        self.bus.write_byte_data(self.addr, self.FIFO_EN, 0x00)
        buf = bytes(self._buf[:n_samples * 6])
        return np.frombuffer(buf, dtype=">i2").reshape(-1, 3).astype(float) / self.LSB_PER_G

    def record(self, seconds, fs=1000):
        """Record accelerometer x,y,z at 1 kHz. Returns array (n, 3) in g."""
        return record_together([self], seconds, fs)[0]


def record_together(mpus, seconds, fs=1000):
    """Record several MPU6050s at once (started within a millisecond of each other)."""
    need = int(seconds * fs)
    for m in mpus:
        m.fifo_start()
    while any(len(m._buf) < need * 6 for m in mpus):
        for m in mpus:
            m.fifo_drain()
        time.sleep(0.005)
    return [m.fifo_stop(need) for m in mpus]


def tilt_deg(accel, bridge_axis="x", vertical_axis="z"):
    """Angle of the board in the vertical plane along the bridge."""
    i = "xyz".index(bridge_axis)
    k = "xyz".index(vertical_axis)
    return math.degrees(math.atan2(accel[i], accel[k]))


# --------------------------------------------------------------------- BME280
class BME280:
    """Temperature + humidity (Bosch datasheet compensation formulas)."""

    def __init__(self, bus, address=0x76):
        self.bus, self.addr = bus, address
        chip = bus.read_byte_data(address, 0xD0)
        if chip == 0x58:
            raise RuntimeError("This board is a BMP280, not a BME280: it has NO humidity sensor. "
                               "(Common mix-up when buying.)")
        if chip != 0x60:
            raise RuntimeError(f"Unexpected chip id 0x{chip:02x} at 0x{address:02x}")
        c = bytes(bus.read_i2c_block_data(address, 0x88, 24))
        self.T1, self.T2, self.T3 = struct.unpack("<Hhh", c[0:6])
        self.H1 = bus.read_byte_data(address, 0xA1)
        e = bytes(bus.read_i2c_block_data(address, 0xE1, 7))
        self.H2 = struct.unpack("<h", e[0:2])[0]
        self.H3 = e[2]
        h4 = (e[3] << 4) | (e[4] & 0x0F)
        h5 = (e[5] << 4) | (e[4] >> 4)
        self.H4 = h4 - 4096 if h4 & 0x800 else h4
        self.H5 = h5 - 4096 if h5 & 0x800 else h5
        self.H6 = struct.unpack("<b", e[6:7])[0]

    def compensate(self, adc_t, adc_h):
        var1 = (adc_t / 16384.0 - self.T1 / 1024.0) * self.T2
        var2 = ((adc_t / 131072.0 - self.T1 / 8192.0) ** 2) * self.T3
        t_fine = var1 + var2
        temp = t_fine / 5120.0
        h = t_fine - 76800.0
        h = (adc_h - (self.H4 * 64.0 + self.H5 / 16384.0 * h)) * (
            self.H2 / 65536.0 * (1.0 + self.H6 / 67108864.0 * h * (1.0 + self.H3 / 67108864.0 * h)))
        h = h * (1.0 - self.H1 * h / 524288.0)
        return temp, min(max(h, 0.0), 100.0)

    def read(self):
        """One forced measurement -> (temperature C, relative humidity %)."""
        self.bus.write_byte_data(self.addr, 0xF2, 0x01)           # humidity x1
        self.bus.write_byte_data(self.addr, 0xF4, 0x25)           # temp x1, pressure x1, forced
        time.sleep(0.015)
        d = self.bus.read_i2c_block_data(self.addr, 0xF7, 8)
        adc_t = (d[3] << 12) | (d[4] << 4) | (d[5] >> 4)
        adc_h = (d[6] << 8) | d[7]
        return self.compensate(adc_t, adc_h)


# --------------------------------------------------------------------- ADS1115
class ADS1115:
    """16-bit analog input.

    read_volts(ch)      single-ended, +/-4.096 V range (sag sensor, quick checks)
    read_diff(fsr=...)  DIFFERENTIAL A0-A1 (or A2-A3) on a small range, for bridge
                        circuits like strain gauges and load cells (signals of ~1 mV)
    """
    VOLTS_PER_LSB = 4.096 / 32768
    PGA = {6.144: 0, 4.096: 1, 2.048: 2, 1.024: 3, 0.512: 4, 0.256: 5}     # range (V) -> code
    RATES = {8: 0, 16: 1, 32: 2, 64: 3, 128: 4, 250: 5, 475: 6, 860: 7}     # samples/s -> code
    MUX = {"A0-A1": 0, "A0-A3": 1, "A1-A3": 2, "A2-A3": 3, "A0": 4, "A1": 5, "A2": 6, "A3": 7}

    def __init__(self, bus, address=0x48):
        self.bus, self.addr = bus, address

    @classmethod
    def config_word(cls, channel, fsr=4.096, rate=860):
        """channel: 0-3 (single-ended) or a MUX name like "A0-A1"."""
        mux = cls.MUX[channel] if isinstance(channel, str) else 4 + channel
        # start | input | range | single-shot | rate | comparator off
        return ((1 << 15) | (mux << 12) | (cls.PGA[fsr] << 9) | (1 << 8)
                | (cls.RATES[rate] << 5) | 0x03)

    def _convert(self, channel, fsr, rate):
        cfg = self.config_word(channel, fsr, rate)
        self.bus.write_i2c_block_data(self.addr, 0x01, [cfg >> 8, cfg & 0xFF])
        time.sleep(1.0 / rate)
        for _ in range(100):
            hi, lo = self.bus.read_i2c_block_data(self.addr, 0x01, 2)
            if hi & 0x80:                                         # conversion finished
                break
            time.sleep(0.001)
        hi, lo = self.bus.read_i2c_block_data(self.addr, 0x00, 2)
        return struct.unpack(">h", bytes([hi, lo]))[0] * fsr / 32768

    def read_volts(self, channel=0):
        return self._convert(channel, 4.096, 860)

    def read_diff(self, pair="A0-A1", fsr=0.256, rate=64):
        """Voltage of the first input minus the second, in volts."""
        return self._convert(pair, fsr, rate)


# --------------------------------------------------------------------- DHT11 / DHT22
class DHTSysfs:
    """DHT11/DHT22 read through the Raspberry Pi's built-in kernel driver.

    The DHT protocol needs microsecond timing that Python can't do reliably on a
    Pi 5, so the kernel does it. Enable once in /boot/firmware/config.txt:
        dtoverlay=dht11,gpiopin=4        (same line for a DHT22)
    then reboot. Individual reads sometimes fail (normal for DHTs), so we retry.
    """

    def __init__(self, base="/sys/bus/iio/devices"):
        from pathlib import Path
        for d in sorted(Path(base).glob("iio:device*")):
            name = (d / "name").read_text().strip() if (d / "name").exists() else ""
            if "dht" in name.lower():
                self.dev = d
                return
        raise RuntimeError("DHT sensor not found. Add 'dtoverlay=dht11,gpiopin=4' to "
                           "/boot/firmware/config.txt and reboot (or run: bash pi_setup.sh --dht).")

    def read(self, tries=8):
        last = None
        for _ in range(tries):
            try:
                t = int((self.dev / "in_temp_input").read_text()) / 1000.0
                h = int((self.dev / "in_humidityrelative_input").read_text()) / 1000.0
                return t, h
            except OSError as e:              # checksum/timeout: wait and retry
                last = e
                time.sleep(2.1)
        raise RuntimeError(f"DHT sensor did not answer after {tries} tries ({last}). Check wiring.")
