"""Piezo crackle (acoustic emission) detector via the USB sound card.

How it works:
  1. Listen to 1 s of quiet to measure the background noise level.
  2. High-pass filter the signal (removes slow thumps from handling the weights).
  3. A "crackle" = the signal jumping above  threshold_x_noise * background.
     Samples above threshold that are less than 10 ms apart belong to the same crackle.
  4. For each crackle record: time, peak amplitude, energy.
"""
import queue
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def find_input_device(sd, name_hint="USB"):
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name_hint.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"No input device matching '{name_hint}'. Is the USB sound card plugged in? "
                       f"Devices: {[d['name'] for d in sd.query_devices()]}")


class EventDetector:
    """Pure signal processing (no hardware) so it can be tested on any computer."""

    def __init__(self, fs, highpass_hz=500.0, hold_s=0.010):
        self.fs = fs
        self.sos = butter(4, highpass_hz, btype="high", fs=fs, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.hold = int(hold_s * fs)
        self.threshold = None
        self.n = 0                       # samples processed so far
        self.events = []                 # dicts: start, last, peak, energy

    def filt(self, x):
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y

    def set_noise(self, x, k):
        y = self.filt(x)
        self.threshold = k * (np.sqrt(np.mean(y ** 2)) + 1e-9)
        self.n = 0

    def process(self, x):
        y = self.filt(x)
        idx = np.flatnonzero(np.abs(y) > self.threshold)
        for i in idx:
            g = self.n + i
            v = float(y[i])
            ev = self.events[-1] if self.events else None
            if ev is not None and g - ev["last"] <= self.hold:
                ev["last"] = g
                ev["peak"] = max(ev["peak"], abs(v))
                ev["energy"] += v * v / self.fs
            else:
                self.events.append(dict(start=g, last=g, peak=abs(v), energy=v * v / self.fs))
        self.n += x.size

    def table(self):
        import pandas as pd
        return pd.DataFrame([dict(t_s=e["start"] / self.fs, amplitude=e["peak"], energy=e["energy"])
                             for e in self.events], columns=["t_s", "amplitude", "energy"])


class CrackleListener:
    def __init__(self, device="USB", samplerate=48000, threshold_x_noise=6.0, highpass_hz=500.0,
                 keep_audio=False):
        import sounddevice as sd
        self.sd = sd
        self.device = find_input_device(sd, device) if isinstance(device, str) else device
        self.fs = samplerate
        self.k = threshold_x_noise
        self.det = EventDetector(samplerate, highpass_hz)
        self.q = queue.Queue()
        self.keep_audio = keep_audio
        self.audio = []
        self.stream = None

    def _callback(self, indata, frames, t, status):
        self.q.put(indata[:, 0].copy())

    def _worker(self):
        while self.running or not self.q.empty():
            try:
                x = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.keep_audio:
                self.audio.append(x)
            self.det.process(x.astype(float))

    def start(self, quiet_seconds=1.0):
        """Measure background noise (keep the bridge still!), then start listening."""
        quiet = self.sd.rec(int(quiet_seconds * self.fs), samplerate=self.fs, channels=1,
                            device=self.device, dtype="float32", blocking=True)[:, 0]
        self.det.set_noise(quiet.astype(float), self.k)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.stream = self.sd.InputStream(samplerate=self.fs, channels=1, device=self.device,
                                          dtype="float32", blocksize=2048, callback=self._callback)
        self.stream.start()
        self.t0 = time.monotonic()

    def stop(self):
        self.stream.stop(); self.stream.close()
        self.running = False
        self.thread.join()
        audio = np.concatenate(self.audio) if self.audio else None
        return self.det.table(), audio
