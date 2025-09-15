"""Voice Activity Detection utilities wrapping Silero VAD."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import importlib.util
import numpy as np
import torch


@dataclass
class _SileroRuntime:
    model: torch.jit.ScriptModule
    get_speech_timestamps: callable
    device: torch.device


class VADProcessor:
    """Detect speech segments in audio using the Silero VAD model."""

    def __init__(
        self,
        threshold: float = 0.65,
        min_speech_ms: int = 200,
        min_silence_ms: int = 250,
        speech_pad_ms: int = 35,
        max_speech_duration_s: Optional[float] = 22.0,
        use_cuda: bool = False,
        model_dir: Optional[str] = None,
    ):
        self.threshold = float(threshold)
        self.min_speech_ms = int(min_speech_ms)
        self.min_silence_ms = int(min_silence_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.max_speech_duration_s = (
            float(max_speech_duration_s) if max_speech_duration_s is not None else None
        )
        self.use_cuda = bool(use_cuda)

        rt = self._load_model(model_dir)
        self.model: torch.jit.ScriptModule = rt.model
        self.get_speech_timestamps = rt.get_speech_timestamps  # type: ignore[attr-defined]
        self.device: torch.device = rt.device

        self.model.eval()

    # ---------------- internal: loading ----------------

    def _load_model(self, model_dir: Optional[str]) -> _SileroRuntime:
        """
        Load Silero VAD model + utils either from a local dir (model.jit, utils_vad.py)
        or via torch.hub. We keep and expose an explicit device handle.
        """
        device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")

        # Try local
        if model_dir:
            mdir = Path(model_dir)
            model_path = mdir / "model.jit"
            utils_path = mdir / "utils_vad.py"
            if model_path.is_file() and utils_path.is_file():
                try:
                    model = torch.jit.load(str(model_path), map_location="cpu")
                except Exception as e:
                    raise RuntimeError(f"Failed to load Silero VAD from {model_path}: {e}")

                spec = importlib.util.spec_from_file_location("silero_utils", str(utils_path))
                if not spec or not spec.loader:
                    raise RuntimeError("Could not load utils_vad.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                get_speech_timestamps = getattr(module, "get_speech_timestamps", None)
                if get_speech_timestamps is None:
                    raise RuntimeError("utils_vad.py does not provide get_speech_timestamps")

                try:
                    model.to(device)
                except Exception:
                    # Fallback to CPU if device move failed for any reason
                    device = torch.device("cpu")
                    model.to(device)

                return _SileroRuntime(model=model, get_speech_timestamps=get_speech_timestamps, device=device)

        # Fallback: torch.hub
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,  # требуется на новых torch
            )
            # utils: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
            get_speech_timestamps = utils[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD via torch.hub: {e}")

        try:
            model.to(device)
        except Exception:
            device = torch.device("cpu")
            model.to(device)

        return _SileroRuntime(model=model, get_speech_timestamps=get_speech_timestamps, device=device)

    # ---------------- public: inference ----------------

    @torch.inference_mode()
    def process(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Return (start, end) speech segments in seconds."""
        segs = self._silero_vad_segments(audio, sr)
        # get_speech_timestamps(return_seconds=True) already returns floats (or ints) in seconds
        return [(float(s["start"]), float(s["end"])) for s in segs]

    # ---------------- internal: inference ----------------

    def _silero_vad_segments(self, audio: np.ndarray, sr: int):
        """
        Run Silero VAD and return raw segment dicts as provided by utils.get_speech_timestamps.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Ensure float32 and finite
        audio = audio.astype(np.float32, copy=False)
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        wav_tensor = torch.from_numpy(audio).to(self.device)

        return self.get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=int(sr),                     # ВАЖНО: частота дискретизации
            threshold=float(self.threshold),
            min_speech_duration_ms=int(self.min_speech_ms),
            min_silence_duration_ms=int(self.min_silence_ms),
            speech_pad_ms=int(self.speech_pad_ms),
            max_speech_duration_s=self.max_speech_duration_s,
            return_seconds=True,                       # Вернём секунды, как ожидает остальной код
        )
