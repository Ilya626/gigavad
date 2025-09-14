import numpy as np
import sys
import pytest
from pathlib import Path
import types

# Provide stub modules before importing the real code
stub_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
sys.modules.setdefault("torch", stub_torch)
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

sys.path.append(str(Path(__file__).resolve().parents[1]))

import inference_gigaam


class DummyVADProcessor:
    def process(self, audio, sr):
        return [(0.0, 1.0), (3.0, 4.0), (10.0, 11.0)]


def test_min_bin_speech_enforced(monkeypatch):
    monkeypatch.setattr(inference_gigaam, "VADProcessor", DummyVADProcessor)
    sr = 1000
    audio = np.zeros(sr * 12, dtype=np.float32)
    cfg = inference_gigaam.VadConfig(
        chunk=inference_gigaam.ChunkingParams(target_speech_sec=2.0),
        pack=inference_gigaam.PackingParams(
            max_silence_within_sec=1.2,
            min_bin_speech_sec=3.0,
        ),
    )
    vad = DummyVADProcessor()
    with pytest.raises(ValueError):
        inference_gigaam.slice_with_silero_vad(sr, audio, vad, cfg.chunk, cfg.pack)


def test_merge_close_segments(monkeypatch):
    monkeypatch.setattr(inference_gigaam, "VADProcessor", DummyVADProcessor)
    sr = 1000
    audio = np.zeros(sr * 12, dtype=np.float32)
    cfg = inference_gigaam.VadConfig(
        chunk=inference_gigaam.ChunkingParams(
            merge_close_segs=True,
            min_gap_sec=2.5,
        )
    )
    vad = DummyVADProcessor()
    _, segs = inference_gigaam.slice_with_silero_vad(sr, audio, vad, cfg.chunk, cfg.pack)
    assert segs == [(0.0, 4.0), (10.0, 11.0)]
