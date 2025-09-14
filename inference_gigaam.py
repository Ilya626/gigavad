#!/usr/bin/env python
from __future__ import annotations

"""
Chunked transcription for Salute GigaAM (no pyannote needed).

- Cuts long audio into short chunks (by energy/silence), with overlap.
- Feeds chunks sequentially to model.transcribe.
- Deduplicates boundary repeats.
- Works on single files, dirs, or JSON/JSONL manifests.
- Outputs a JSON { "input_path": "full text" }.
- Optional: write JSONL with segments [start,end,text].

Example:
  python inference_gigaam_chunked.py Ilya_1_h.wav out/transcript.json \
    --model v2_rnnt --lang ru --chunk_sec 22 --overlap_sec 1.5 \
    --silence_peak_ratio 0.002 --write_segments out/segments.jsonl --debug
    # optionally add: --vad_silero --vad_pad_ms 200 --silero_model_dir /path/to/silero-vad
"""

import argparse
import json
import math
import os
import sys
import gc
import time
import tempfile
import subprocess
import shutil
import io
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from vad_module import VADProcessor
from chunking_module import ChunkingProcessor

try:
    import torch
except Exception as e:
    raise SystemExit(f"PyTorch is required: {e}")

try:
    import gigaam  # type: ignore
except Exception as e:
    gigaam = None  # type: ignore
    _IMPORT_ERR = e

# Optional RUPunct integration for inline punctuation of segments
_HAVE_TRANSFORMERS = False
try:
    from transformers import pipeline as _hf_pipeline, AutoTokenizer as _AutoTokenizer  # type: ignore
    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False

def _build_rupunct_pipeline(model_id: str = "RUPunct/RUPunct_big"):
    """Create a HuggingFace pipeline for Russian punctuation restoration."""
    if not _HAVE_TRANSFORMERS:
        raise RuntimeError("transformers not available for RUPunct")
    tk = _AutoTokenizer.from_pretrained(model_id, strip_accents=False, add_prefix_space=True)
    clf = _hf_pipeline("ner", model=model_id, tokenizer=tk, aggregation_strategy="first")
    return clf

def _rupunct_process_token(token: str, label: str) -> str:
    """Apply punctuation/casing transformation to a single token."""
    # Mirror mapping from rupunct_apply.py
    mapping = {
        "LOWER_O": "",
        "LOWER_PERIOD": ".",
        "LOWER_COMMA": ",",
        "LOWER_QUESTION": "?",
        "LOWER_TIRE": " —",
        "LOWER_DVOETOCHIE": ":",
        "LOWER_VOSKL": "!",
        "LOWER_PERIODCOMMA": ";",
        "LOWER_DEFIS": "-",
        "LOWER_MNOGOTOCHIE": "...",
        "LOWER_QUESTIONVOSKL": "?!",
    }
    upper_map = {
        "UPPER_O": (True, ""),
        "UPPER_PERIOD": (True, "."),
        "UPPER_COMMA": (True, ","),
        "UPPER_QUESTION": (True, "?"),
        "UPPER_TIRE": (True, " —"),
        "UPPER_DVOETOCHIE": (True, ":"),
        "UPPER_VOSKL": (True, "!"),
        "UPPER_PERIODCOMMA": (True, ";"),
        "UPPER_DEFIS": (True, "-"),
        "UPPER_MNOGOTOCHIE": (True, "..."),
        "UPPER_QUESTIONVOSKL": (True, "?!"),
    }
    upper_total_map = {
        "UPPER_TOTAL_O": (str.upper, ""),
        "UPPER_TOTAL_PERIOD": (str.upper, "."),
        "UPPER_TOTAL_COMMA": (str.upper, ","),
        "UPPER_TOTAL_QUESTION": (str.upper, "?"),
        "UPPER_TOTAL_TIRE": (str.upper, " —"),
        "UPPER_TOTAL_DVOETOCHIE": (str.upper, ":"),
        "UPPER_TOTAL_VOSKL": (str.upper, "!"),
        "UPPER_TOTAL_PERIODCOMMA": (str.upper, ";"),
        "UPPER_TOTAL_DEFIS": (str.upper, "-"),
        "UPPER_TOTAL_MNOGOTOCHIE": (str.upper, "..."),
        "UPPER_TOTAL_QUESTIONVOSKL": (str.upper, "?!"),
    }
    if label in mapping:
        return token + mapping[label]
    if label in upper_map:
        cap, p = upper_map[label]
        return token.capitalize() + p
    if label in upper_total_map:
        fn, p = upper_total_map[label]
        return fn(token) + p
    return token

def _rupunct_fix_spaces(text: str) -> str:
    """Normalize whitespace around punctuation marks."""
    import re as _re
    rules = [
        (_re.compile(r"\s+([,.;:!?—])"), r"\1"),
        (_re.compile(r"\s+—\s*"), " — "),
        (_re.compile(r"\s{2,}"), " "),
    ]
    out = text.strip()
    for rx, rep in rules:
        out = rx.sub(rep, out)
    return out.strip()

def _rupunct_text(clf, text: str) -> str:
    """Run RUPunct on ``text`` using ``clf`` and return punctuated string."""
    if not text or not text.strip():
        return text
    preds = clf(text)
    parts = []
    for it in preds:
        token = (it.get("word") or "").strip()
        label = it.get("entity_group") or "LOWER_O"
        parts.append(_rupunct_process_token(token, label))
    return _rupunct_fix_spaces(" ".join(parts))


# ------------------------- Utilities & Env -------------------------

def _safe_transcribe(model, source, lang: Optional[str] = None):
    """Safely call model.transcribe with language if supported.

    - Detects if the callable accepts a 'language' or 'lang' kwarg.
    - Falls back to calling without language if not supported.
    - 'source' can be a path string or a file-like buffer depending on the model.
    """
    fn = getattr(model, "transcribe", None)
    if fn is None:
        raise AttributeError("Model has no 'transcribe' method")
    kwargs = {}
    if lang:
        try:
            import inspect
            sig = inspect.signature(fn)
            params = sig.parameters
            if "language" in params:
                kwargs["language"] = lang
            elif "lang" in params:
                kwargs["lang"] = lang
        except Exception:
            # If we can't introspect, try without passing language
            kwargs = {}
    return fn(source, **kwargs)

def configure_local_caches() -> Path:
    """Configure cache directories for torch and temporary files."""
    repo_root = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    tmp = repo_root / ".tmp"
    for var in ("TMPDIR", "TMP", "TEMP"):
        os.environ.setdefault(var, str(tmp))
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return repo_root


def require_cuda():
    """Raise an error if CUDA is not available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU inference is disabled.")


def vram_report(tag: str) -> None:
    """Print a brief GPU memory usage report tagged with ``tag``."""
    try:
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        free, _ = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        gb = 1024 ** 3
        print(f"[VRAM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G")
    except Exception:
        pass


def acquire_gpu_lock(lock_path: Path, timeout_s: int = 120) -> tuple[bool, int]:
    """Attempt to create a file lock to coordinate GPU usage."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(pid))
            print(f"[LOCK] Acquired GPU lock at {lock_path}")
            return True, pid
        except FileExistsError:
            try:
                existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
                try:
                    os.kill(existing_pid, 0)
                except OSError:
                    lock_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            time.sleep(1)
        except Exception:
            try:
                existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
                try:
                    os.kill(existing_pid, 0)
                except OSError:
                    lock_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            time.sleep(1)
    return False, pid


def release_gpu_lock(lock_path: Path, owner_pid: int) -> None:
    """Release a file lock created with :func:`acquire_gpu_lock`."""
    try:
        if lock_path.exists():
            try:
                content = lock_path.read_text(encoding="utf-8").strip()
            except Exception:
                content = ""
            if content == str(owner_pid):
                lock_path.unlink(missing_ok=True)
                print(f"[LOCK] Released GPU lock at {lock_path}")
    except Exception:
        pass


def _format_ts(sec: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _is_dense(text: str, dur: float, min_wps: float = 1.0, min_cps: float = 3.0) -> bool:
    """Heuristic to skip segments with too little text for their duration."""
    wps = len(text.split()) / max(dur, 1e-9)
    cps = len(text) / max(dur, 1e-9)
    return (wps >= min_wps) or (cps >= min_cps)


# ------------------------- Audio I/O helpers -------------------------

def _get_duration_sec(path: Path) -> Optional[float]:
    """Return audio duration in seconds using ``soundfile`` metadata."""
    try:
        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


def _preconvert_if_needed(path: Path, repo_root: Path, force: bool = False) -> Path:
    """Ensure mono 16k PCM16 WAV using ffmpeg; convert MP3/FLAC/etc. or mismatched WAV."""
    ext = path.suffix.lower()
    needs = force or ext in {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".flac"}
    if not needs and ext == ".wav":
        try:
            info = sf.info(str(path))
            subtype = getattr(info, "subtype", "") or ""
            if info.channels != 1 or info.samplerate != 16000 or "PCM_16" not in subtype:
                needs = True
        except Exception:
            needs = True

    if not needs:
        return path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[preconvert] ffmpeg not found, proceeding without conversion")
        # fallback: try to write as-is; model may still cope
        return path

    out_dir = repo_root / ".tmp" / "gigaam_pre"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_16k_mono.wav"

    cmd = [
        ffmpeg, "-y", "-i", str(path),
        "-ac", "1", "-ar", "16000", "-vn", "-acodec", "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception as e:
        print(f"[preconvert] Conversion skipped for {path}: {e}")
        return path


# ------------------------- Chunking logic (no VAD deps) -------------------------

@dataclass
class SileroParams:
    threshold: float = 0.65
    min_speech_ms: int = 200
    min_silence_ms: int = 250
    speech_pad_ms: int = 35
    use_cuda: bool = False
    model_dir: Optional[str] = None


@dataclass
class ChunkingParams:
    target_speech_sec: float = 22.0
    cut_search_sec: float = 2.0
    frame_ms: float = 20.0
    silence_abs: float = 0.0
    silence_peak_ratio: float = 0.002
    adaptive: bool = True
    pad_context_ms: int = 0
    min_gap_sec: float = 0.3
    merge_close_segs: bool = False


@dataclass
class PackingParams:
    max_overshoot_sec: float = 0.0
    max_silence_within_sec: float = 1.2
    min_bin_speech_sec: float = 0.0


@dataclass
class VadConfig:
    silero: SileroParams = field(default_factory=SileroParams)
    chunk: ChunkingParams = field(default_factory=ChunkingParams)
    pack: PackingParams = field(default_factory=PackingParams)


def slice_with_silero_vad(
    sr: int,
    audio: np.ndarray,
    vad_processor: VADProcessor,
    chunk_cfg: ChunkingParams,
    pack_cfg: PackingParams,
) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    """Compute chunks via Silero VAD and pack them into ≈target_speech_sec bins."""

    segs = vad_processor.process(audio, sr)

    cp_split = ChunkingProcessor(
        chunk_sec=chunk_cfg.target_speech_sec,
        overlap_sec=0.0,
        search_silence_sec=chunk_cfg.cut_search_sec,
        silence_abs=chunk_cfg.silence_abs,
        silence_peak_ratio=chunk_cfg.silence_peak_ratio,
        frame_ms=chunk_cfg.frame_ms,
        adaptive=chunk_cfg.adaptive,
    )

    refined: list[tuple[float, float]] = []
    for s, e in segs:
        if e - s <= chunk_cfg.target_speech_sec:
            refined.append((s, e))
            continue
        seg_audio = audio[int(round(s * sr)): int(round(e * sr))]
        sub_chunks = cp_split.process(seg_audio, sr)
        for ss, ee in sub_chunks:
            start_t = s + ss / sr
            end_t = s + ee / sr
            refined.append((start_t, min(end_t, e)))
    segs = refined

    if chunk_cfg.pad_context_ms > 0:
        pad = chunk_cfg.pad_context_ms / 1000.0
        total_dur = len(audio) / sr
        padded: list[tuple[float, float]] = []
        for s, e in segs:
            s = max(0.0, s - pad)
            e = min(total_dur, e + pad)
            padded.append((s, e))
        segs = padded

    if chunk_cfg.merge_close_segs and segs:
        merged: list[tuple[float, float]] = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = merged[-1]
            if s - pe < chunk_cfg.min_gap_sec:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        segs = merged

    bins: list[list[tuple[float, float]]] = []
    cur_bin: list[tuple[float, float]] = []
    cur_speech = 0.0
    last_end = None
    for s, e in segs:
        seg_duration = e - s
        if cur_bin:
            gap = s - (last_end if last_end is not None else s)
            if gap > pack_cfg.max_silence_within_sec or (
                (cur_speech + seg_duration) > (chunk_cfg.target_speech_sec + pack_cfg.max_overshoot_sec)
            ):
                bins.append(cur_bin)
                cur_bin = []
                cur_speech = 0.0
        cur_bin.append((s, e))
        cur_speech += seg_duration
        last_end = e
    if cur_bin:
        bins.append(cur_bin)

    if pack_cfg.min_bin_speech_sec > 0:
        for b in bins[:-1]:
            speech = sum(e - s for s, e in b)
            if speech < pack_cfg.min_bin_speech_sec:
                raise ValueError(
                    f"Bin speech {speech:.2f}s shorter than {pack_cfg.min_bin_speech_sec}s"
                )

    out: list[tuple[int, int]] = []
    for bin_segs in bins:
        start = bin_segs[0][0]
        end = bin_segs[-1][1]
        ss = max(0, int(round(start * sr)))
        ee = max(ss + 1, int(round(end * sr)))
        out.append((ss, ee))
    return out, segs


def _dedup_suffix_prefix(
    prev_tail: str,
    new_text: str,
    min_overlap: int = 16,
    similarity_threshold: float = 0.9,
) -> str:
    """Remove repeated prefix of ``new_text`` if it duplicates suffix of ``prev_tail``.

    The comparison is performed on word tokens after Unicode normalization
    (``NFKC``) and punctuation preprocessing so that visually similar text or
    text with different punctuation/spacing still deduplicates correctly.
    ``min_overlap`` denotes the minimal number of tokens that must overlap in
    order to trigger deduplication. ``similarity_threshold`` specifies the
    minimal ratio from :class:`difflib.SequenceMatcher` to treat two token
    sequences as overlapping.
    """

    import re
    import unicodedata
    from difflib import SequenceMatcher

    def _normalize_tokens(text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text)
        # Replace punctuation with spaces before tokenization
        text = "".join(
            " " if unicodedata.category(ch).startswith("P") else ch for ch in text
        )
        return text.lower().split()

    a_tokens = _normalize_tokens(prev_tail)
    b_tokens = _normalize_tokens(new_text)
    max_k = min(len(a_tokens), len(b_tokens), 200)  # safety limit
    for k in range(max_k, min_overlap - 1, -1):
        a_seq = a_tokens[-k:]
        b_seq = b_tokens[:k]
        ratio = SequenceMatcher(None, a_seq, b_seq).ratio()
        if ratio >= similarity_threshold:
            # Determine character index in original new_text to cut from
            norm_new = unicodedata.normalize("NFKC", new_text)
            word_matches = list(re.finditer(r"\w+", norm_new))
            if k > len(word_matches):
                return ""
            cut_norm = word_matches[k - 1].end()
            # Skip any trailing punctuation or spaces
            while cut_norm < len(norm_new) and not norm_new[cut_norm].isalnum():
                cut_norm += 1
            # Map normalized index back to original text index
            idx_map: list[int] = []
            for i, ch in enumerate(new_text):
                norm_ch = unicodedata.normalize("NFKC", ch)
                idx_map.extend([i] * len(norm_ch))
            if cut_norm >= len(idx_map):
                return ""
            cut_orig = idx_map[cut_norm]
            while cut_orig < len(new_text) and not new_text[cut_orig].isalnum():
                cut_orig += 1
            return new_text[cut_orig:]
    return new_text


# ------------------------- Main transcribe loop -------------------------

def transcribe_file_sequential(
    model,
    path: Path,
    repo_root: Path,
    lang: Optional[str],
    chunk_sec: float,
    overlap_sec: float,
    search_silence_sec: float,
    vad_cfg: VadConfig | None,
    dedup_tail_chars: int,
    min_dedup_overlap: int,
    debug: bool,
    use_vad_silero: bool = False,
    use_tempfile: bool = False,
    min_wps: float = 1.0,
    min_cps: float = 3.0,
) -> tuple[str, list[dict], list[str]]:
    """Transcribe ``path`` sequentially and return full text and segment info."""
    # Hardcode Russian language per project requirements
    lang = "ru"
    src = _preconvert_if_needed(path, repo_root, force=False)
    audio, sr = sf.read(str(src))
    if sr != 16000 or audio.ndim != 1:
        print(f"[audio] expected 16000Hz mono, got sr={sr}, ndim={audio.ndim}")
        return "", [], []
    n = len(audio)
    if n == 0:
        return "", [], []

    vad_cfg = vad_cfg or VadConfig()

    # Prefer VAD-based chunking when requested
    chunks = None
    speech_secs: list[tuple[float, float]] | None = None
    if use_vad_silero:
        try:
            vad_processor = VADProcessor(
                threshold=vad_cfg.silero.threshold,
                min_speech_ms=vad_cfg.silero.min_speech_ms,
                min_silence_ms=vad_cfg.silero.min_silence_ms,
                speech_pad_ms=vad_cfg.silero.speech_pad_ms,
                max_speech_duration_s=vad_cfg.chunk.target_speech_sec,
                use_cuda=vad_cfg.silero.use_cuda,
                model_dir=vad_cfg.silero.model_dir,
            )
            chunks, speech_secs = slice_with_silero_vad(
                sr,
                audio,
                vad_processor,
                vad_cfg.chunk,
                vad_cfg.pack,
            )
        except Exception as e:
            print(f"[VAD][Silero] failed ({e}); falling back to energy-based slicing.")

    if chunks is None:
        # If VAD was requested but failed, honor target_speech_sec as time target
        time_target = float(vad_cfg.chunk.target_speech_sec) if use_vad_silero else float(chunk_sec)
        cp = ChunkingProcessor(
            chunk_sec=time_target,
            overlap_sec=overlap_sec,
            search_silence_sec=search_silence_sec,
            silence_abs=vad_cfg.chunk.silence_abs,
            silence_peak_ratio=vad_cfg.chunk.silence_peak_ratio,
            frame_ms=vad_cfg.chunk.frame_ms,
            adaptive=vad_cfg.chunk.adaptive,
        )
        chunks = cp.process(audio, sr)
    if debug:
        print(
            f"[CHUNKS] {path.name}: {len(chunks)} chunks; sr={sr}; "
            f"chunk≈{time_target}s ovlp={overlap_sec}s search={search_silence_sec}s "
            f"thr(abs={vad_cfg.chunk.silence_abs}, peak_ratio={vad_cfg.chunk.silence_peak_ratio})"
        )

    tmpdir = repo_root / ".tmp" / "gigaam_chunks_seq"
    if use_tempfile:
        tmpdir.mkdir(parents=True, exist_ok=True)
    full_text_parts: List[str] = []
    segments: List[dict] = []

    for i, (s0, s1) in enumerate(chunks):
        seg_audio = audio[s0:s1]
        if use_tempfile:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", prefix=f"{path.stem}_chunk_", dir=tmpdir, delete=False
            ) as tmpf:
                sf.write(tmpf, seg_audio, sr, format="WAV")
                wav_path = Path(tmpf.name)
            try:
                try:
                    with torch.inference_mode():
                        out = _safe_transcribe(model, str(wav_path), lang)
                except TypeError:
                    # Some backends may still raise TypeError on buffer/path specifics; try without language
                    with torch.inference_mode():
                        out = _safe_transcribe(model, str(wav_path), None)
            except Exception as e:
                print(
                    f"[ERROR] transcribe chunk {i+1}/{len(chunks)} {path.name}: {e}"
                )
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass
                segments.append({"start": s0 / sr, "end": s1 / sr, "text": ""})
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            buf = io.BytesIO()
            sf.write(buf, seg_audio, sr, format="WAV")
            buf.seek(0)
            try:
                try:
                    with torch.inference_mode():
                        out = _safe_transcribe(model, buf, lang)
                except TypeError:
                    # Fallback: model does not accept file-like objects.
                    tmpdir.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        prefix=f"{path.stem}_chunk_",
                        dir=tmpdir,
                        delete=False,
                    ) as tmpf:
                        sf.write(tmpf, seg_audio, sr, format="WAV")
                        tmp_path = Path(tmpf.name)
                    try:
                        with torch.inference_mode():
                            out = _safe_transcribe(model, str(tmp_path), lang)
                    finally:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception as e:
                print(
                    f"[ERROR] transcribe chunk {i+1}/{len(chunks)} {path.name}: {e}"
                )
                segments.append({"start": s0 / sr, "end": s1 / sr, "text": ""})
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

        # extract text
        if isinstance(out, dict):
            text = out.get("transcription") or out.get("text") or out.get("transcript") or ""
        elif isinstance(out, list):
            # join list of strings/dicts
            parts = []
            for item in out:
                if isinstance(item, dict):
                    parts.append(str(item.get("transcription") or item.get("text") or item.get("transcript") or ""))
                else:
                    parts.append(str(item))
            text = " ".join([t for t in parts if t]).strip()
        else:
            text = out if isinstance(out, str) else str(out)

        if debug:
            print(f"[CHUNK {i+1:03d}/{len(chunks)}] {path.name} "
                  f"{_format_ts(s0/sr)}вЂ“{_format_ts(s1/sr)} txt_len={len(text)}")

        # deduplicate overlap
        tail = "".join(full_text_parts)[-dedup_tail_chars:] if full_text_parts else ""
        text = _dedup_suffix_prefix(tail, text, min_overlap=min_dedup_overlap)
        dur = (s1 - s0) / sr
        if text and _is_dense(text, dur, min_wps=min_wps, min_cps=min_cps):
            full_text_parts.append(text)
            segments.append({
                "start": s0 / sr,
                "end": s1 / sr,
                "text": text,
            })
        else:
            segments.append({"start": s0 / sr, "end": s1 / sr, "text": ""})

        # free VRAM between chunks
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    full_text = " ".join([t for t in full_text_parts if t]).strip()

    comparison_lines: list[str] = []
    if speech_secs:
        for s, e in speech_secs:
            comparison_lines.append(f"{s:.1f}-{e:.1f} Голос")
    return full_text, segments, comparison_lines


# ------------------------- CLI -------------------------



# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    """Build argument parser and return parsed arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Path to audio file/dir or JSONL/JSON manifest")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument("--model", default="v2_rnnt", help="Model type: v2_rnnt, v2_ctc, rnnt, ctc, etc.")
    parser.add_argument("--lang", type=str, default="", help="Force language code, e.g. 'ru' or 'en'. Empty=auto")
    parser.add_argument(
        "--output_format",
        choices=["json", "txt"],
        default="json",
        help="Output format: json (default) or txt",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="transcript_results.txt",
        help="Path for text report when using --output_format txt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON config file with argument values",
    )

    chunk_group = parser.add_argument_group("Chunking")
    chunk_group.add_argument("--chunk_sec", type=float, default=22.0, help="Target chunk length (seconds)")
    chunk_group.add_argument("--overlap_sec", type=float, default=1.0, help="Overlap between chunks (seconds)")
    chunk_group.add_argument(
        "--search_silence_sec",
        type=float,
        default=0.6,
        help="Search window near boundary to cut at silence (seconds)",
    )
    chunk_group.add_argument("--frame_ms", type=float, default=20.0, help="Frame size for energy envelope (ms)")
    chunk_group.add_argument(
        "--silence_threshold",
        type=float,
        default=0.0,
        help="Absolute RMS threshold; 0=use peak ratio",
    )
    chunk_group.add_argument(
        "--silence_peak_ratio",
        type=float,
        default=0.002,
        help="Threshold = global_peak * ratio when absolute=0",
    )
    chunk_group.add_argument(
        "--no_adaptive",
        action="store_true",
        help="Disable adaptive noise-floor thresholding",
    )

    vad_group = parser.add_argument_group("VAD")
    vad_group.add_argument(
        "--target_speech_sec",
        type=float,
        default=22.0,
        help="Target speech seconds per bin (no word cuts)",
    )
    vad_group.add_argument(
        "--vad_max_overshoot",
        type=float,
        default=1.0,
        help="Allowable overshoot beyond target speech per bin (sec)",
    )
    vad_group.add_argument(
        "--vad_max_silence_within",
        type=float,
        default=1.2,
        help="Max silence gap allowed inside a bin (sec)",
    )
    vad_group.add_argument(
        "--vad_min_bin_speech",
        type=float,
        default=10.0,
        help=(
            "Minimum speech per bin before it can be closed; gaps/overshoot are"
            " ignored until this is satisfied (sec)"
        ),
    )
    vad_group.add_argument(
        "--vad_min_gap_sec",
        type=float,
        default=0.3,
        help=(
            "Minimum pause considered a boundary (sec)."
            " Gaps smaller than this merge when --vad_merge_segs is used"
        ),
    )
    vad_group.add_argument(
        "--vad_cut_search_sec",
        type=float,
        default=2.0,
        help="Search window around target for internal pause cut (sec)",
    )
    vad_group.add_argument(
        "--vad_pad_ms",
        type=int,
        default=0,
        help="Padding added to each VAD segment before binning (ms)",
    )
    vad_group.add_argument(
        "--vad_merge_segs",
        action="store_true",
        help="Merge adjacent VAD segments separated by less than --vad_min_gap_sec",
    )

    silero_group = parser.add_argument_group("Silero VAD")
    silero_group.add_argument(
        "--vad_silero",
        action="store_true",
        help="Use Silero VAD (torch.hub) for chunking into ~22s speech bins",
    )
    silero_group.add_argument(
        "--silero_threshold", type=float, default=0.65, help="Silero VAD threshold"
    )
    silero_group.add_argument(
        "--silero_min_speech_ms",
        type=int,
        default=200,
        help="Minimum speech duration in ms",
    )
    silero_group.add_argument(
        "--silero_min_silence_ms",
        type=int,
        default=250,
        help="Minimum silence to separate speech in ms",
    )
    silero_group.add_argument(
        "--silero_speech_pad_ms",
        type=int,
        default=35,
        help="Padding around detected speech in ms",
    )
    silero_group.add_argument(
        "--silero_cuda",
        action="store_true",
        help="Run Silero VAD on CUDA if available",
    )
    silero_group.add_argument(
        "--silero_model_dir",
        type=str,
        default="",
        help="Directory with local Silero VAD model",
    )

    dedup_group = parser.add_argument_group("Deduplication")
    dedup_group.add_argument(
        "--dedup_tail_chars",
        type=int,
        default=80,
        help="How many trailing chars from previous text to compare",
    )
    dedup_group.add_argument(
        "--min_dedup_overlap",
        type=int,
        default=16,
        help="Min token overlap to consider as duplication",
    )

    filt_group = parser.add_argument_group("Filtering")
    filt_group.add_argument(
        "--min_wps",
        type=float,
        default=1.0,
        help="Minimum words-per-second to keep a segment",
    )
    filt_group.add_argument(
        "--min_cps",
        type=float,
        default=3.0,
        help="Minimum characters-per-second to keep a segment",
    )

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--batch_size", type=int, default=16, help="Files per loop batch (I/O grouping)"
    )
    misc_group.add_argument(
        "--sample", type=int, default=0, help="Sample N files from manifest (0=all)"
    )
    misc_group.add_argument(
        "--write_segments",
        type=str,
        default="",
        help="Optional JSONL file to write segments with timestamps",
    )
    misc_group.add_argument(
        "--write_segments_punct",
        type=str,
        default="",
        help="Optional JSONL to write segments with punctuation (text_punct)",
    )
    misc_group.add_argument(
        "--punct_ru",
        action="store_true",
        help="Apply RUPunct to segment texts and write text_punct",
    )
    misc_group.add_argument(
        "--no_lock", action="store_true", help="Do not acquire GPU lock"
    )
    misc_group.add_argument(
        "--use_tempfile",
        action="store_true",
        help="Use legacy temp files for model input",
    )
    misc_group.add_argument("--debug", action="store_true", help="Debug prints")

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, val)
    return args


def main():
    """Command-line interface for chunked GigaAM transcription."""
    args = parse_args()


    repo_root = configure_local_caches()
    require_cuda()

    if gigaam is None:
        raise SystemExit(
            "GigaAM is not installed.\n"
            "Use Python 3.11/3.12 venv and then: pip install gigaam  (or git+https://github.com/salute-developers/GigaAM).\n"
            f"Underlying import error: {_IMPORT_ERR}"
        )

    # GPU lock
    lock_path = repo_root / ".tmp" / "gpu.lock"
    pid = os.getpid()
    lock_acquired = False
    if not args.no_lock:
        lock_acquired, pid = acquire_gpu_lock(lock_path)
        if not lock_acquired:
            raise RuntimeError(f"Could not acquire GPU lock at {lock_path}; another process is running")

    # Load model
    model = gigaam.load_model(args.model)
    if hasattr(model, "eval"):
        model.eval()
    try:
        if hasattr(model, "to"):
            model = model.to("cuda")
        elif hasattr(model, "cuda"):
            model = model.cuda()
    except Exception:
        pass

    # Collect input files
    input_path = Path(args.input)
    if input_path.suffix.lower() in {".jsonl", ".json"} and input_path.is_file():
        audio_files: list[Path] = []
        if input_path.suffix.lower() == ".jsonl":
            with open(input_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    p = obj.get("audio_filepath") or obj.get("audio")
                    if p:
                        pp = Path(p)
                        if not pp.is_absolute():
                            pp = (input_path.parent / pp).resolve()
                        audio_files.append(pp)
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                audio_files = [Path(k) for k in obj.keys()]
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        p = it.get("audio_filepath") or it.get("audio")
                        if p:
                            pp = Path(p)
                            if not pp.is_absolute():
                                pp = (input_path.parent / pp).resolve()
                            audio_files.append(pp)
    else:
        audio_files = [input_path] if input_path.is_file() else sorted(
            p for p in input_path.glob("**/*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
        )

    # Optional sampling
    if args.sample and args.sample > 0 and len(audio_files) > args.sample:
        import random
        random.seed(42)
        random.shuffle(audio_files)
        audio_files = audio_files[: args.sample]

    # Transcribe
    results: dict[str, str] = {}
    all_reports: dict[str, list[str]] = {}
    seg_writer = None
    seg_writer_punct = None
    rupunct = None
    if args.write_segments:
        Path(args.write_segments).parent.mkdir(parents=True, exist_ok=True)
        seg_writer = open(args.write_segments, "w", encoding="utf-8")
    if args.write_segments_punct:
        Path(args.write_segments_punct).parent.mkdir(parents=True, exist_ok=True)
        seg_writer_punct = open(args.write_segments_punct, "w", encoding="utf-8")
    if args.punct_ru and (args.write_segments_punct or args.write_segments):
        try:
            rupunct = _build_rupunct_pipeline()
            print("[RUPunct] model loaded for inline punctuation")
        except Exception as e:
            print(f"[RUPunct][WARN] cannot load model: {e}")
            rupunct = None

    vad_cfg = VadConfig(
        silero=SileroParams(
            threshold=float(args.silero_threshold),
            min_speech_ms=int(args.silero_min_speech_ms),
            min_silence_ms=int(args.silero_min_silence_ms),
            speech_pad_ms=int(args.silero_speech_pad_ms),
            use_cuda=bool(args.silero_cuda),
            model_dir=args.silero_model_dir or None,
        ),
        chunk=ChunkingParams(
            target_speech_sec=float(args.target_speech_sec),
            cut_search_sec=float(args.vad_cut_search_sec),
            frame_ms=float(args.frame_ms),
            silence_abs=float(args.silence_threshold),
            silence_peak_ratio=float(args.silence_peak_ratio),
            adaptive=not args.no_adaptive,
            pad_context_ms=int(args.vad_pad_ms),
            min_gap_sec=float(args.vad_min_gap_sec),
            merge_close_segs=bool(args.vad_merge_segs),
        ),
        pack=PackingParams(
            max_overshoot_sec=float(args.vad_max_overshoot),
            max_silence_within_sec=float(args.vad_max_silence_within),
            min_bin_speech_sec=float(args.vad_min_bin_speech),
        ),
    )

    bs = max(1, int(args.batch_size))
    try:
        for i in range(0, len(audio_files), bs):
            batch_paths = audio_files[i: i + bs]
            print(f"Transcribing batch {i // bs + 1} [{len(batch_paths)} files] ...")
            for p in batch_paths:
                full_text, segments, comparison_lines = transcribe_file_sequential(
                    model, p, repo_root,
                    args.lang or None,
                    args.chunk_sec, args.overlap_sec,
                    args.search_silence_sec,
                    vad_cfg,
                    int(args.dedup_tail_chars), int(args.min_dedup_overlap),
                    bool(args.debug),
                    bool(args.vad_silero),
                    bool(args.use_tempfile),
                    float(args.min_wps), float(args.min_cps),
                )
                results[str(p)] = full_text
                all_reports[str(p)] = comparison_lines
                if (seg_writer or seg_writer_punct) and segments:
                    for s in segments:
                        base_obj = {
                            "audio": str(p),
                            "start": s["start"],
                            "end": s["end"],
                            "start_str": _format_ts(s["start"]),
                            "end_str": _format_ts(s["end"]),
                            "text": s["text"],
                        }
                        if seg_writer:
                            seg_writer.write(json.dumps(base_obj, ensure_ascii=False) + "\n")
                        if seg_writer_punct:
                            obj = dict(base_obj)
                            if rupunct is not None:
                                try:
                                    obj["text_punct"] = _rupunct_text(rupunct, s["text"]) or s["text"]
                                except Exception:
                                    obj["text_punct"] = s["text"]
                            else:
                                obj["text_punct"] = s["text"]
                            seg_writer_punct.write(json.dumps(obj, ensure_ascii=False) + "\n")

            vram_report(f"batch_{i // bs + 1}")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main JSON output
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        # Save text report if requested
        if args.output_format == "txt":
            report_path = Path(args.output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as f:
                for lines in all_reports.values():
                    for line in lines:
                        f.write(f"{line}\n")
    finally:
        if seg_writer:
            seg_writer.close()
        if seg_writer_punct:
            seg_writer_punct.close()
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        vram_report("final")
        # release lock
        if not args.no_lock:
            release_gpu_lock(lock_path, pid)


if __name__ == "__main__":
    main()
