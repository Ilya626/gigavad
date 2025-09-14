import numpy as np
import sys
from pathlib import Path
import types

# Provide stub modules before importing the real code
stub_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules.setdefault("torch", stub_torch)
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

sys.path.append(str(Path(__file__).resolve().parents[1]))

import inference_gigaam


class DummyVADProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def process(self, audio, sr):
        return [(0.0, 1.0), (3.0, 4.0), (10.0, 11.0)]


def test_min_bin_speech_accumulates(monkeypatch):
    monkeypatch.setattr(inference_gigaam, "VADProcessor", DummyVADProcessor)
    sr = 1000
    audio = np.zeros(sr * 12, dtype=np.float32)
    chunks, _ = inference_gigaam.slice_with_silero_vad(
        sr,
        audio,
        target_speech_sec=2.0,
        max_silence_within_sec=1.2,
        min_bin_speech_sec=3.0,
    )
    assert len(chunks) == 1
    assert chunks[0] == (0, 11000)


def test_merge_close_segments(monkeypatch):
    monkeypatch.setattr(inference_gigaam, "VADProcessor", DummyVADProcessor)
    sr = 1000
    audio = np.zeros(sr * 12, dtype=np.float32)
    _, segs = inference_gigaam.slice_with_silero_vad(
        sr,
        audio,
        merge_close_segs=True,
        min_gap_sec=2.5,
    )
    assert segs == [(0.0, 4.0), (10.0, 11.0)]
