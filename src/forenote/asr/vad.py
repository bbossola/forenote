"""Streaming Silero VAD on 512-sample frames (audio spec 5.2)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np

FRAME = 512  # 32 ms at 16 kHz
CONTEXT = 64
SAMPLE_RATE = 16000


def default_model_path() -> str:
    """The Silero VAD model file that faster-whisper includes."""
    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("The faster_whisper package is not installed. It includes the VAD model.")
    path = os.path.join(spec.submodule_search_locations[0], "assets", "silero_vad_v6.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The VAD model file is missing: {path}")
    return path


class StreamingVad:
    """Give a speech probability for each frame. Keep the model state between frames.

    One instance serves one audio stream. It is not thread-safe.
    """

    def __init__(self, model_path: str | None = None):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            model_path or default_model_path(), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME,):
            raise ValueError(f"a frame must have {FRAME} samples, not {frame.shape}")
        if frame.dtype != np.int16 and not np.issubdtype(frame.dtype, np.floating):
            raise ValueError(f"a frame must be int16 or floating point, not {frame.dtype}")
        x = frame.astype(np.float32)
        if frame.dtype == np.int16:
            x /= 32768.0
        x = x.reshape(1, FRAME)
        out, self._h, self._c = self._session.run(
            None, {"input": np.concatenate([self._context, x], axis=1), "h": self._h, "c": self._c}
        )
        self._context = x[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])
