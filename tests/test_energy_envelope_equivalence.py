import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from chunking_module import ChunkingProcessor


def reference_energy_envelope(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    pad = (frame_len - (n - frame_len) % hop_len) % hop_len
    x = np.pad(audio.astype(np.float32, copy=False), (0, pad))
    if len(x) < frame_len:
        return np.zeros(0, dtype=np.float32)
    frames = 1 + (len(x) - frame_len) // hop_len
    out = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        s = i * hop_len
        e = s + frame_len
        w = x[s:e]
        out[i] = float(np.sqrt(np.mean(w * w) + 1e-12))
    return out


def test_energy_envelope_matches_reference():
    rng = np.random.default_rng(0)
    proc = ChunkingProcessor()
    for n in [0, 1, 10, 100, 1234]:
        audio = rng.standard_normal(n).astype(np.float32)
        for frame_len, hop_len in [(5, 1), (20, 10), (40, 20)]:
            env_new = proc._compute_energy_envelope(audio, frame_len, hop_len)
            env_ref = reference_energy_envelope(audio, frame_len, hop_len)
            assert np.allclose(env_new, env_ref, rtol=1e-7, atol=1e-9)
