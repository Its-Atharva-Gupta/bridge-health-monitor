"""Read the load cell through an ADS1115 instead of the HX711, to find out which
one is causing jumpy readings.

  python -m bridge_shm.pi.diag_ads

Wiring (3-wire load cell + two equal resistors, e.g. 1 kOhm each):
  load cell end wire 1  -> 3.3 V          load cell end wire 2 -> GND
  load cell middle wire -> ADS1115 A0
  resistor from 3.3 V to A1, resistor from A1 to GND   (A1 = reference half-way point)
  ADS1115: VDD -> 3.3 V, GND -> GND, SDA/SCL -> same as the MPUs, ADDR -> GND (0x48)
(4-wire load cell: red E+ -> 3.3 V, black E- -> GND, green/white signal wires -> A0 and A1;
 no resistors needed.)

Step 1 checks the wiring (A0 and A1 should both sit near half the supply voltage).
Step 2 shows the tiny difference A0-A1 live. Press the load cell's free end and let go.
"""
import numpy as np

from .i2c_sensors import ADS1115


def main():
    from smbus2 import SMBus
    ads = ADS1115(SMBus(1), 0x48)
    print(__doc__)

    print("STEP 1: wiring check (nothing on the load cell)")
    supply = 3.3
    for ch, what in ((0, "A0 = load cell middle wire"), (1, "A1 = resistor half-way point")):
        v = float(np.median([ads.read_volts(ch) for _ in range(20)]))
        ok = abs(v - supply / 2) < 0.25 * supply
        print(f"  {what:32s} {v:6.3f} V   {'OK (about half of 3.3 V)' if ok else 'PROBLEM'}")
        if not ok:
            print("    -> Expected about 1.65 V. Near 0 V or 3.3 V means a wrong or broken wire: "
                  "re-check which load cell wire is the middle one (multimeter on Ohms).")

    print("\nSTEP 2: live load cell signal (A0 - A1), 16 readings per line. Ctrl+C to stop.")
    print("  Leave it alone for ~10 lines, then press the free end, then let go.\n")
    print("   average (microvolts)   spread (microvolts)")
    zero = None
    try:
        while True:
            vals = np.array([ads.read_diff("A0-A1", 0.256, 64) for _ in range(16)]) * 1e6
            avg = float(np.median(vals))
            zero = avg if zero is None else zero
            spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))
            bar = "#" * min(60, int(abs(avg - zero) / 20))
            print(f"   {avg - zero:+10.1f}           {spread:8.1f}      {bar}")
    except KeyboardInterrupt:
        print("""
How to read this:
  * Steady line (spread ~10-40 uV) that jumps clearly when you press and comes back
    when you let go -> the LOAD CELL and its wiring are fine: the HX711 board (or its
    wires / power) is the problem. Try another HX711.
  * Jumpy here too, or creeping up/down on its own -> the problem is the load cell,
    its wiring or its mounting (loose wires, missing resistors, something touching).
  * No change when you press -> wrong wires, or the load cell can't bend (check the
    mounting: one end screwed down, the other end free).""")


if __name__ == "__main__":
    main()
