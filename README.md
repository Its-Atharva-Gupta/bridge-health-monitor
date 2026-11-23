# Bridge health monitoring: software

Pipeline: **sensors → test cycle files → features → anomaly model → health score / plots**

| File | What it does |
|---|---|
| `bridge_shm/config.py` | Bridge geometry, test protocol (load steps, sample rates), feature list |
| `bridge_shm/storage.py` | How a test cycle is saved (plain CSV + JSON, same for real and simulated data) |
| `bridge_shm/simulate.py` | Fake test cycles from beam physics + noise + weather + damage stages (D0-D3) |
| `bridge_shm/features.py` | One cycle → one row: stiffness, softening, hysteresis, leftover sag, strain, frequency, damping, crackles |
| `bridge_shm/model.py` | Detectors trained on healthy data only: Mahalanobis (main), Isolation Forest, sag threshold |
| `bridge_shm/evaluate.py` | False-alarm and detection rates, model comparison, poster plots |

## Run it

```bash
source .venv/bin/activate
python -m bridge_shm.simulate --out data/sim      # 72 fake cycles (36 healthy, 12 per damage stage)
python -m bridge_shm.evaluate data/sim            # results + plots in data/sim/results/
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

Turn sensors on/off in `rig_config.json` (created on first run; `"enabled": true/false`).
Without a sag sensor, sag is estimated from the MPU6050 tilt. Without strain gauges,
the model simply leaves the strain features out.

Practise without hardware (works on the laptop too):
`python -m bridge_shm.pi.run_test --fake --stage D0 --cycles 2`

### Wiring (Pi 5 header)

| Sensor pin | Pi GPIO | Physical pin |
|---|---|---|
| HX711 #1 (load cell) DT / SCK | GPIO5 / GPIO6 | 29 / 31 |
| HX711 #2 (left gauge) DT / SCK | GPIO17 / GPIO27 | 11 / 13 |
| HX711 #3 (right gauge) DT / SCK | GPIO23 / GPIO24 | 16 / 18 |
| I2C SDA (MPU6050, BME280, ADS1115) | GPIO2 | 3 |
| I2C SCL (all three) | GPIO3 | 5 |
| DHT11 / DHT22 data | GPIO4 | 7 |
| All sensor VCC | 3.3 V | 1 or 17 |
| All sensor GND | GND | 6, 9, 14, 20, 25, 30, 34, 39 |

**Power every sensor board from 3.3 V, never 5 V.** The Pi's pins are 3.3 V only; an
HX711 powered from 5 V sends 5 V signals back and can damage the Pi.

## Tests

`python -m pytest tests` checks the driver logic (HX711 decoding, BME280 maths against
the datasheet example, ADS1115 settings, crackle detection) with fake chips.

## Still to build
- Live demo: dashboard, voice, tone, LEDs
