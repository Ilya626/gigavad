"""Utilities for splitting audio into overlapping energy-based chunks."""

import numpy as np
from typing import List, Tuple, Optional


class ChunkingProcessor:
    """Split audio into chunks using a simple energy envelope heuristic.

    Parameters
    ----------
    chunk_sec:
        Target chunk length in seconds.
    overlap_sec:
        Overlap between consecutive chunks in seconds.
    min_chunk_sec:
        Minimum allowed chunk length in seconds.
    search_silence_sec:
        Radius around a boundary to search for a lower-energy cut.
    silence_abs:
        Absolute RMS threshold for silence.
    silence_peak_ratio:
        When ``silence_abs`` is zero, use ``global_peak * silence_peak_ratio``.
    frame_ms:
        Frame size for energy envelope in milliseconds.
    adaptive:
        If ``True`` (default), estimate a noise floor from the energy envelope
        and include it when computing the silence threshold.
    """

    def __init__(self, chunk_sec=22.0, overlap_sec=1.0,
                 min_chunk_sec: float = 3.0,
                 search_silence_sec=0.6, silence_abs=0.0,
                 silence_peak_ratio=0.002, frame_ms=20.0,
                 adaptive: bool = True):
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self.min_chunk_sec = min_chunk_sec
        self.search_silence_sec = search_silence_sec
        self.silence_abs = silence_abs
        self.silence_peak_ratio = silence_peak_ratio
        self.frame_ms = frame_ms
        self.adaptive = adaptive

    def process(self, audio: np.ndarray, sr: int, recalc_sec: Optional[float] = None) -> List[Tuple[int, int]]:
        """Return chunk boundaries for the given audio and sample rate."""
        return self.slice_into_chunks(audio, sr, recalc_sec=recalc_sec)

    def _compute_energy_envelope(self, audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
        """Compute short-time RMS energy envelope for mono audio.

        ``numpy.lib.stride_tricks.sliding_window_view`` constructs the full
        matrix of frames in memory which can be costly for long signals.  This
        implementation instead uses a cumulative sum trick to compute the
        sliding average of squared samples and downsamples by ``hop_len`` to
        obtain the per-frame RMS energy without allocating the intermediate
        window matrix.
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

        sq = x * x
        frames = 1 + (len(x) - frame_len) // hop_len
        out = np.empty(frames, dtype=np.float32)
        for i in range(frames):
            s = i * hop_len
            e = s + frame_len
            out[i] = float(np.sqrt(np.mean(sq[s:e]) + 1e-12))
        return out

    def _find_cut_near(self, target_samp: int, sr: int, env: np.ndarray,
                      frame_len: int, hop_len: int, search_sec: float, thr: float,
                      audio: Optional[np.ndarray] = None) -> int:
        """Find a near-silence cut point close to ``target_samp``.

        If no frame in the search window falls below ``thr``, a local minimum
        of the energy envelope is selected instead.  When even a local minimum
        cannot be identified (e.g. the window is strictly monotonic), an
        in-place short fade-in/fade-out is applied around ``target_samp`` in
        the ``audio`` array, if provided, to smooth the eventual hard cut.
        """
        target_frame = max(0, min(len(env) - 1, target_samp // hop_len))
        radius = max(1, int(round(search_sec * sr / hop_len)))
        lo = max(0, target_frame - radius)
        hi = min(len(env) - 1, target_frame + radius)
        window = env[lo:hi + 1]
        if window.size == 0:
            return target_samp

        below = np.where(window <= thr)[0]
        if below.size:
            idx_rel = below[np.argmin(window[below])]
            best_frame = lo + idx_rel
            return best_frame * hop_len

        if window.size >= 3:
            local_mins = np.where((window[1:-1] < window[:-2]) & (window[1:-1] <= window[2:]))[0] + 1
            if local_mins.size:
                idx_rel = local_mins[np.argmin(window[local_mins])]
                best_frame = lo + idx_rel
                return best_frame * hop_len

        if audio is not None:
            fade_samps = max(1, int(round(0.005 * sr)))
            s0 = max(0, target_samp - fade_samps)
            e0 = min(len(audio), target_samp + fade_samps)
            if target_samp > s0:
                fade = np.linspace(0.0, 1.0, target_samp - s0, endpoint=False)
                audio[s0:target_samp] *= fade
            if e0 > target_samp:
                fade = np.linspace(1.0, 0.0, e0 - target_samp, endpoint=False)
                audio[target_samp:e0] *= fade

        return target_samp

    def slice_into_chunks(self, audio: np.ndarray, sr: int,
                          recalc_sec: Optional[float] = None) -> List[Tuple[int, int]]:
        """Return list of ``(start, end)`` sample indices for each chunk.

        When ``adaptive`` is ``True``, an estimate of the noise floor from the
        lower quantile of the energy envelope is included when computing the
        silence threshold.  If ``recalc_sec`` is provided, the noise floor is
        re-estimated every ``recalc_sec`` seconds from the local envelope to
        track slowly varying background levels.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        n = len(audio)
        if n == 0:
            return []

        frame_len = max(1, int(round(sr * self.frame_ms / 1000.0)))
        hop_len = frame_len // 2
        env = self._compute_energy_envelope(audio, frame_len, hop_len)
        peak = float(np.max(np.abs(audio))) if n else 0.0
        noise_floor = float(np.quantile(env, 0.1)) if env.size else 0.0

        chunk_len = int(round(self.chunk_sec * sr))
        overlap = int(round(self.overlap_sec * sr))
        min_len = int(round(self.min_chunk_sec * sr))
        search_sec = float(self.search_silence_sec)

        recalc_frames = None
        if recalc_sec is not None and recalc_sec > 0:
            recalc_frames = max(1, int(round(recalc_sec * sr / hop_len)))

        chunks: List[Tuple[int, int]] = []
        start = 0
        while start < n:
            if recalc_frames is not None and env.size:
                start_frame = start // hop_len
                seg = env[start_frame:start_frame + recalc_frames]
                if seg.size:
                    noise_floor = float(np.quantile(seg, 0.1))

            if self.adaptive:
                thr = max(1e-7, float(self.silence_abs), peak * float(self.silence_peak_ratio), noise_floor)
            else:
                thr = max(1e-7, float(self.silence_abs) if self.silence_abs > 0 else peak * float(self.silence_peak_ratio))

            raw_end = min(n, start + chunk_len)
            cut = self._find_cut_near(raw_end, sr, env, frame_len, hop_len, search_sec, thr, audio)
            end = min(n, max(start + min_len, cut))
            if end <= start:
                end = min(n, start + chunk_len)
            chunks.append((start, end))
            if end >= n:
                break
            start = max(0, end - overlap)

        return chunks
