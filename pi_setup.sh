#!/usr/bin/env bash
# One-time Raspberry Pi 5 setup. Run ON THE PI from the software folder:
#   bash pi_setup.sh          (BME280)
#   bash pi_setup.sh --dht    (DHT11 / DHT22 instead)
set -e
echo "== installing system packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-lgpio i2c-tools libportaudio2 alsa-utils

echo "== enabling I2C at 400 kHz (needed for 1 kHz vibration recording) =="
sudo raspi-config nonint do_i2c 0
CFG=/boot/firmware/config.txt
if ! grep -q "i2c_arm_baudrate" "$CFG"; then
  echo "dtparam=i2c_arm_baudrate=400000" | sudo tee -a "$CFG"
  NEED_REBOOT=1
fi

if [ "$1" = "--dht" ] && ! grep -q "dtoverlay=dht11" "$CFG"; then
  echo "== enabling the DHT11/DHT22 kernel driver on GPIO4 =="
  echo "dtoverlay=dht11,gpiopin=4" | sudo tee -a "$CFG"
  NEED_REBOOT=1
fi

echo "== Python environment =="
# --system-site-packages lets the venv use the apt-installed lgpio
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install numpy scipy pandas scikit-learn smbus2 sounddevice

echo "== USB sound card: turn off automatic gain, fixed mic level =="
for c in $(grep -il usb /proc/asound/card*/id 2>/dev/null | sed 's#.*/card\([0-9]*\)/id#\1#'); do
  amixer -c "$c" sset 'Auto Gain Control' off 2>/dev/null || true
  amixer -c "$c" sset Mic 50% cap 2>/dev/null || true
  echo "  configured sound card $c"
done

echo "== I2C devices found (expect 48 = ADS1115, 68 = MPU6050, 76 = BME280) =="
i2cdetect -y 1 || true

if [ -n "$NEED_REBOOT" ]; then
  echo; echo "I2C speed changed: please reboot once (sudo reboot)."
fi
echo "Done."
