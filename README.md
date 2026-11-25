# Bridge health monitoring: software

Pipeline: **sensors → test cycle files → features → anomaly model → health score / plots**

| File | What it does |
|---|---|
| `bridge_shm/config.py` | Bridge geometry, test protocol (load steps, sample rates), feature list |
| `bridge_shm/storage.py` | How a test cycle is saved (plain CSV + JSON, same for real and simulated data) |
| `bridge_shm/beam.py` | Beam physics: sag, tilt and bending moment under load, and the first two vibration modes, with weakened "damage zones" |
| `bridge_shm/simulate.py` | Fake test cycles from that physics + noise + weather + an off-centre load cup + damage stages (D0-D3) |
| `bridge_shm/features.py` | One cycle → one row: stiffness, softening, hysteresis, leftover sag, strain, left/right tilt ratio, two vibration frequencies, damping, crackles |
| `bridge_shm/model.py` | Detectors trained on healthy data only: Mahalanobis (main), Isolation Forest, sag threshold |
| `bridge_shm/evaluate.py` | False-alarm and detection rates, model comparison, poster plots |
| `bridge_shm/sensor_study.py` | Which sensors matter: re-runs the analysis with sensors left out |

## Run it

```bash
source .venv/bin/activate
python -m bridge_shm.simulate --out data/sim      # 72 fake cycles (36 healthy, 12 per damage stage)
python -m bridge_shm.evaluate data/sim            # results + plots in data/sim/results/
python -m bridge_shm.sensor_study                 # which sensors matter (simulated, 5 datasets)
python -m bridge_shm.sensor_study --data data/real    # ...the same on your real data
```

**Simulated results are for testing the code only, never for the poster.** The damage
effect sizes in `simulate.py` are guesses; your real experiment decides the real numbers.

## On the Raspberry Pi 5

Copy this whole `software` folder to the Pi, then **on the Pi**:

```bash
bash pi_setup.sh --dht                             # once (drop --dht if using a BME280); reboot if it asks
source .venv/bin/activate
python -m bridge_shm.pi.calibrate check            # live readings from every sensor
python -m bridge_shm.pi.calibrate load             # zero + known weight at the centre
python -m bridge_shm.pi.calibrate sag              # only if you have a sag sensor
python -m bridge_shm.pi.calibrate strain           # once the gauges are fitted
python -m bridge_shm.pi.run_test --plank P1 --stage D0 --cycles 5
```

Load cell readings jumpy? `python -m bridge_shm.pi.diag_ads` reads the load cell through an
ADS1115 instead of the HX711 (wiring is in the file's header) to find which one is at fault.

Turn sensors on/off in `rig_config.json` (created on first run; `"enabled": true/false`).
Without a sag sensor, sag is estimated from the tilt of the two MPU6050s. Without strain
gauges (or with one MPU), the model simply leaves those features out.

### Bridge setup the code assumes
- The plank **rests** on both supports (not nailed): pillars on the left, the load cell's
  free end (with a small flat pad) on the right. Small guide blocks stop it sliding off.
- Load cup at the exact centre, hanging freely; nothing under the plank touches the base.
- MPU6050 #1 at 1/4 span, #2 at 3/4 span, both mounted the same way round, firmly glued.
- Tap the plank at 1/3 span (90 mm from the left support): off-centre, so both
  vibration modes ring. The two MPUs must agree on the frequency or the test asks for a redo.

Practise without hardware (works on the laptop too):
`python -m bridge_shm.pi.run_test --fake --stage D0 --cycles 2`

### Wiring (Pi 5 header)

| Sensor pin | Pi GPIO | Physical pin |
|---|---|---|
| HX711 #1 (load cell) DT / SCK | GPIO5 / GPIO6 | 29 / 31 |
| HX711 #2 (left gauge) DT / SCK | GPIO17 / GPIO27 | 11 / 13 |
| HX711 #3 (right gauge) DT / SCK | GPIO23 / GPIO24 | 16 / 18 |
| I2C SDA (both MPU6050s, BME280, ADS1115) | GPIO2 | 3 |
| I2C SCL (all of them) | GPIO3 | 5 |
| **MPU6050 #2 AD0** (sets its address to 0x69) | 3.3 V | 1 or 17 |
| DHT11 / DHT22 data | GPIO4 | 7 |
| All sensor VCC | 3.3 V | 1 or 17 |
| All sensor GND | GND | 6, 9, 14, 20, 25, 30, 34, 39 |

Check the I2C sensors with `i2cdetect -y 1`: you should see **68** and **69** (the two MPUs).

**Power every sensor board from 3.3 V, never 5 V.** The Pi's pins are 3.3 V only; an
HX711 powered from 5 V sends 5 V signals back and can damage the Pi.

## Tests

`python -m pytest tests` checks the driver logic (HX711 decoding, BME280 maths against
the datasheet example, ADS1115 settings, crackle detection, recording two MPUs at once)
with fake chips, plus the beam physics against textbook formulas and the vibration analysis.

## Still to build
- Live demo: dashboard, voice, tone, LEDs
