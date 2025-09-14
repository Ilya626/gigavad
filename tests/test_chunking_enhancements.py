import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from chunking_module import ChunkingProcessor


def test_find_cut_near_local_min():
    proc = ChunkingProcessor()
    env = np.array([3.0, 2.0, 1.0, 2.0, 3.0], dtype=np.float32)
    audio = np.zeros(10, dtype=np.float32)
    cut = proc._find_cut_near(
        target_samp=2,
        sr=1,
        env=env,
        frame_len=1,
        hop_len=1,
        search_sec=1.0,
        thr=0.5,
        audio=audio,
    )
    assert cut == 2


def test_find_cut_near_applies_fade_when_no_minimum():
    proc = ChunkingProcessor()
    env = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sr = 1000
    audio = np.ones(sr * 2, dtype=np.float32)
    cut = proc._find_cut_near(
        target_samp=sr,
        sr=sr,
        env=env,
        frame_len=1,
        hop_len=1,
        search_sec=1.0,
        thr=0.5,
        audio=audio,
    )
    assert cut == sr
    fade_len = int(round(0.005 * sr))
    assert np.all(audio[sr - fade_len:sr] < 1.0)
    # At least some samples after the cut should be attenuated
    assert np.any(audio[sr:sr + fade_len] < 1.0)


def test_slice_recalc_noise_floor_runs():
    proc = ChunkingProcessor()
    sr = 1000
    audio = np.zeros(sr * 5, dtype=np.float32)
    chunks_ref = proc.slice_into_chunks(audio, sr)
    chunks_recalc = proc.slice_into_chunks(audio, sr, recalc_sec=1.0)
    assert chunks_ref == chunks_recalc
