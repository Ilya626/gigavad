"""Utilities for splitting audio into overlapping energy-based chunks."""

import numpy as np
from typing import List, Tuple


class ChunkingProcessor:
    """Split audio into chunks using a simple energy envelope heuristic."""

    def __init__(self, chunk_sec=22.0, overlap_sec=1.0,
                 search_silence_sec=0.6, silence_abs=0.0,
                 silence_peak_ratio=0.002, frame_ms=20.0):
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self.search_silence_sec = search_silence_sec
        self.silence_abs = silence_abs
        self.silence_peak_ratio = silence_peak_ratio
        self.frame_ms = frame_ms

    def process(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """Return chunk boundaries for the given audio and sample rate."""
        return self.slice_into_chunks(audio, sr)

    def _compute_energy_envelope(self, audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
        """Compute short-time RMS energy envelope for mono audio.

        The previous implementation used a Python loop to iterate over
        frames.  Here we rely on ``numpy.lib.stride_tricks.sliding_window_view``
        to construct all frames at once and compute the per-frame RMS value in
        a vectorized fashion.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        n = len(audio)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        pad = (frame_len - (n - frame_len) % hop_len) % hop_len
        x = np.pad(audio.astype(np.float32, copy=False), (0, pad))
        if len(x) < frame_len:
            return np.zeros(0, dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len]
        out = np.sqrt(np.mean(windows * windows, axis=1) + 1e-12)
        return out.astype(np.float32, copy=False)

    def _find_cut_near(self, target_samp: int, sr: int, env: np.ndarray,
                      frame_len: int, hop_len: int, search_sec: float, thr: float) -> int:
        """Find a near-silence cut point close to ``target_samp``."""
        target_frame = max(0, min(len(env) - 1, target_samp // hop_len))
        radius = max(1, int(round(search_sec * sr / hop_len)))
        lo = max(0, target_frame - radius)
        hi = min(len(env) - 1, target_frame + radius)
        window = env[lo:hi + 1]
        if window.size == 0:
            return target_samp
        idx_rel = int(np.argmin(window))
        if window[idx_rel] <= thr:
            best_frame = lo + idx_rel
            return best_frame * hop_len
        return target_samp

    def slice_into_chunks(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """Return list of ``(start, end)`` sample indices for each chunk."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        n = len(audio)
        if n == 0:
            return []

        frame_len = max(1, int(round(sr * self.frame_ms / 1000.0)))
        hop_len = frame_len // 2
        env = self._compute_energy_envelope(audio, frame_len, hop_len)
        peak = float(np.max(np.abs(audio))) if n else 0.0
        thr = max(1e-7, float(self.silence_abs) if self.silence_abs > 0 else peak * float(self.silence_peak_ratio))

        chunk_len = int(round(self.chunk_sec * sr))
        overlap = int(round(self.overlap_sec * sr))
        min_len = int(round(3.0 * sr))  # минимальная длина чанка 3 секунды
        search_sec = float(self.search_silence_sec)

        chunks: List[Tuple[int, int]] = []
        start = 0
        while start < n:
            raw_end = min(n, start + chunk_len)
            cut = self._find_cut_near(raw_end, sr, env, frame_len, hop_len, search_sec, thr)
            end = min(n, max(start + min_len, cut))
            if end <= start:
                end = min(n, start + chunk_len)
            chunks.append((start, end))
            if end >= n:
                break
            start = max(0, end - overlap)

        return chunks
