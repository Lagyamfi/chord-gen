"""Live system-audio capture through a loopback device.

macOS setup (once):
  1. brew install blackhole-2ch   (then restart Core Audio or log out/in)
  2. Audio MIDI Setup -> "+" -> Create Multi-Output Device; tick your speakers
     (or headphones) AND BlackHole 2ch, speakers first, drift correction on
     BlackHole.
  3. System Settings -> Sound -> Output: choose the Multi-Output Device. You
     still hear everything, and BlackHole gets a copy we can record.
Linux: use a PulseAudio/PipeWire "Monitor of ..." source. Windows: WASAPI loopback.
"""

from __future__ import annotations

import queue

import numpy as np

DEFAULT_HINT = "BlackHole"


def _sd():
    try:
        import sounddevice
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("live mode needs the 'live' extra: uv sync --extra live") from e
    return sounddevice


def list_input_devices() -> list[dict]:
    sd = _sd()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"], "samplerate": d["default_samplerate"]}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def find_input_device(hint: str = DEFAULT_HINT) -> dict | None:
    hint = hint.lower()
    return next((d for d in list_input_devices() if hint in d["name"].lower()), None)


class LoopbackRecorder:
    """Records a device to mono float32 in the background; `read()` drains what
    has arrived since the last call. Use as a context manager."""

    def __init__(self, device: dict):
        self.device = device
        self.samplerate = int(device["samplerate"])
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):  # runs on the audio thread
        self._queue.put(indata.mean(axis=1).astype(np.float32).copy())

    def __enter__(self) -> LoopbackRecorder:
        sd = _sd()
        self._stream = sd.InputStream(
            device=self.device["index"],
            channels=min(2, self.device["channels"]),
            samplerate=self.samplerate,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def read(self) -> np.ndarray:
        blocks = []
        while True:
            try:
                blocks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
