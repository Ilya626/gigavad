#!/usr/bin/env python
from __future__ import annotations

"""
VAD-only chunked transcription for Salute GigaAM.

— Только VAD (Silero). 
— Можно паковать несколько VAD-сегментов в бины (~TARGET_SPEECH_SEC).
— Работает по директориям с аудиофайлами, создаёт segments.jsonl для каждого файла и финальный dialog.jsonl.
— Диалог собирается без пунктуации, объединяет последовательные реплики одного файла, если пауза <= MAX_DIALOG_GAP_SEC.
— Пунктуация применяется после объединения реплик.
"""

# ================================ ГЛОБАЛЬНЫЕ НАСТРОЙКИ ================================
# ВХОД/ВЫХОД
INPUT: str = "records"                       # Путь к папке с аудиофайлами
OUTPUT: str = "out/transcript.json"          # Куда писать основной JSON результат
OUTPUT_FORMAT: str = "json"                  # "json" или "txt"
OUTPUT_REPORT: str = "out/transcript_results.txt"  # Текстовый отчёт, если OUTPUT_FORMAT="txt"
WRITE_SEGMENTS: str = "out"                  # Папка для {имя_файла}_segments.jsonl

# МОДЕЛЬ/ЯЗЫК
MODEL_NAME: str = "v2_rnnt"                  # Модель GigaAM
LANG: str = "ru"                             # Жёстко "ru" для проекта

# VAD (Silero)
SILERO_THRESHOLD: float = 0.35                # Порог VAD
SILERO_MIN_SPEECH_MS: int = 100              # Мин. длительность речи, мс
SILERO_MIN_SILENCE_MS: int = 200             # Мин. пауза между речью, мс
SILERO_SPEECH_PAD_MS: int = 100              # Паддинг вокруг речи, мс
SILERO_CUDA: bool = True                     # Гонять Silero на CUDA (если есть)
SILERO_MODEL_DIR: str = ""                   # Кастомная директория модели (опц.)

# Постобработка VAD
PAD_CONTEXT_MS: int = 50                     # Паддинг к каждому VAD сегменту, мс
MERGE_CLOSE_SEGS: bool = True                # Сливать сегменты, если пауза < MIN_GAP_SEC
MIN_GAP_SEC: float = 0.1                     # Порог для слияния соседних сегментов, сек

# УПАКОВКА В БИНЫ (БЕЗ РЕЗКИ СЕГМЕНТОВ!)
VAD_PACK_BINS: bool = True                   # Включить упаковку в ~TARGET_SPEECH_SEC
TARGET_SPEECH_SEC: float = 15                # Целевая "речевая" длина бина, сек
VAD_MAX_OVERSHOOT: float = 2.0               # Допустимый переразмер бина, сек
VAD_MAX_SILENCE_WITHIN: float = 0.75         # Макс. пауза внутри одного бина, сек
MAX_BIN_DUR_SEC: float = 20.0                # Максимальная общая длина бина, сек
MAX_DIALOG_GAP_SEC: float = 3.0              # Макс. пауза между репликами одного файла для объединения, сек

# ДЕДУП/ФИЛЬТРАЦИЯ ТЕКСТА
KEEP_ALL: bool = True                        # True: ничего не фильтровать
DEDUP_TAIL_CHARS: int = 80                   # Если KEEP_ALL=False: хвост для дедупа, симв.
MIN_DEDUP_OVERLAP: int = 16                  # Мин. перекрытие токенов для дедупа
MIN_WPS: float = 1.0                         # Мин. слов/сек (для фильтра), если KEEP_ALL=False
MIN_CPS: float = 3.0                         # Мин. символов/сек (для фильтра), если KEEP_ALL=False

# ПРОЧЕЕ
PUNCT_RU: bool = True                        # Пунктуация RUPunct
BATCH_SIZE: int = 16                         # Батч по файлам (I/O-группировка)
SAMPLE: int = 0                              # Взять N файлов из манифеста (0 = все)
USE_TEMPFILE: bool = False                   # Писать временные WAV вместо буфера
PARALLEL_PROCESSES: int = 0                  # Отключено: всё в одном потоке
DEBUG: bool = True                           # Отладочные принты
# =====================================================================================

import json, os, sys, gc, io, tempfile
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata

import numpy as np
import soundfile as sf

try:
    import torch
except Exception as e:
    raise SystemExit(f"PyTorch is required: {e}")

try:
    import gigaam  # type: ignore
except Exception as e:
    gigaam = None  # type: ignore
    _IMPORT_ERR = e

# Optional RUPunct
_HAVE_RUPUNCT = False
try:
    import rupunct_apply as rp  # build_punct_pipeline / punctuate_text
    _HAVE_RUPUNCT = True
except Exception as _RUPUNCT_IMPORT_ERR:
    rp = None
    _HAVE_RUPUNCT = False

# Наш VAD
from vad_module import VADProcessor

# Для улучшенного ресэмплинга
try:
    from scipy.signal import resample
except ImportError:
    resample = None

# ------------------------- Утилиты & окружение -------------------------

def configure_local_caches() -> Path:
    repo_root = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
    tmp = repo_root / ".tmp"
    for var in ("TMPDIR", "TMP", "TEMP"):
        os.environ.setdefault(var, str(tmp))
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return repo_root

def _format_ts(sec: float) -> str:
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

def _read_wav_16k_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(str(path), always_2d=False)
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}")
        return np.array([]), target_sr
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float32, copy=False)
    if data.ndim == 2:
        data = data.mean(axis=1, dtype=np.float32)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1).mean(axis=1).astype(np.float32)
    if sr != target_sr:
        if resample is not None:
            data = resample(data, int(len(data) * target_sr / sr))
        else:
            n_in = data.shape[0]
            ratio = float(target_sr) / float(sr)
            n_out = max(1, int(round(n_in * ratio)))
            x_in = np.linspace(0.0, n_in - 1.0, num=n_in, endpoint=True, dtype=np.float64)
            x_out = np.linspace(0.0, n_in - 1.0, num=n_out, endpoint=True, dtype=np.float64)
            data = np.interp(x_out, x_in, data.astype(np.float64, copy=False)).astype(np.float32)
        sr = target_sr
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return data, sr

def _safe_transcribe(model, source, lang: Optional[str] = None):
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
            kwargs = {}
    return fn(source, **kwargs)

def _is_dense(text: str, dur: float, min_wps: float = 1.0, min_cps: float = 3.0) -> bool:
    if not text:
        return False
    wps = len(text.split()) / max(dur, 1e-9)
    cps = len(text) / max(dur, 1e-9)
    return (wps >= min_wps) or (cps >= min_cps)


# ------------------------- VAD → чанки (без резки!) -------------------------

@dataclass
class SileroParams:
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    use_cuda: bool
    model_dir: Optional[str]

@dataclass
class VadParams:
    target_speech_sec: float
    max_overshoot_sec: float
    max_silence_within_sec: float
    pad_context_ms: int
    min_gap_sec: float
    merge_close_segs: bool
    pack_bins: bool
    max_bin_dur_sec: float

@dataclass
class VadConfig:
    silero: SileroParams
    params: VadParams

def _merge_close(segs: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for s, e in segs[1:]:
        if s - out[-1][1] < min_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]

def slice_with_silero_vad(
    sr: int,
    audio: np.ndarray,
    vad_processor: VADProcessor,
    cfg: VadParams,
) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    """
    Только VAD. Никаких разрезаний по энергии.
    Если pack_bins=False -> по одному чанку на каждый VAD-сегмент.
    Если pack_bins=True  -> собираем несколько VAD-сегментов в бины ~target (не делим сегменты).
    """
    segs = list(vad_processor.process(audio, sr))  # [(s,e)] в секундах
    total_dur = len(audio) / sr

    max_chunk_sec = cfg.target_speech_sec + cfg.max_overshoot_sec  # 17 сек

    # Сначала добавляем паддинг
    if cfg.pad_context_ms > 0 and segs:
        pad = cfg.pad_context_ms / 1000.0
        segs = [(max(0.0, s - pad), min(total_dur, e + pad)) for s, e in segs]

    # Затем слияние близких
    if cfg.merge_close_segs:
        segs = _merge_close(segs, cfg.min_gap_sec)

    # Теперь разбиваем длинные сегменты после слияния
    new_segs = []
    for s, e in segs:
        seg_d = e - s
        if seg_d > cfg.max_bin_dur_sec:
            start = s
            while start < e:
                end = min(start + cfg.target_speech_sec, e)  # ~15 сек или остаток
                new_segs.append((start, end))
                start = end
        else:
            new_segs.append((s, e))
    segs = new_segs

    speech_secs = list(segs)  # для отчётов

    if not cfg.pack_bins:
        chunks = []
        for s, e in segs:
            ss = max(0, int(round(s * sr)))
            ee = max(ss + 1, int(round(e * sr)))
            chunks.append((ss, ee))
        return chunks, speech_secs

    bins: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    cur_speech = 0.0
    last_end = None
    cur_start = None

    for s, e in segs:
        seg_d = e - s

        if not cur and seg_d > max_chunk_sec:
            bins.append([(s, e)])
            last_end = e
            continue

        if cur:
            gap = s - (last_end if last_end is not None else s)
            potential_end = e
            potential_dur = potential_end - cur_start if cur_start is not None else seg_d
            need_close = (gap > cfg.max_silence_within_sec) or (
                (cur_speech + seg_d) > max_chunk_sec
            ) or (potential_dur > cfg.max_bin_dur_sec)
            if need_close:
                bins.append(cur)
                cur = []
                cur_speech = 0.0
                cur_start = None

        if not cur:
            cur_start = s

        cur.append((s, e))
        cur_speech += seg_d
        last_end = e

    if cur:
        bins.append(cur)

    chunks: list[tuple[int, int]] = []
    for b in bins:
        start = b[0][0]
        end = b[-1][1]
        ss = max(0, int(round(start * sr)))
        ee = max(ss + 1, int(round(end * sr)))
        chunks.append((ss, ee))

    return chunks, speech_secs


# ------------------------- Основной цикл транскриба -------------------------

def transcribe_file_sequential(
    model,
    path: Path,
    repo_root: Path,
    lang: Optional[str],
    vad_cfg: VadConfig,
    vad_processor: Optional[VADProcessor] = None,
    *,
    dedup_tail_chars: int,
    min_dedup_overlap: int,
    debug: bool,
    use_tempfile: bool = False,
    min_wps: float = 1.0,
    min_cps: float = 3.0,
    keep_all: bool = True,
) -> tuple[str, list[dict], list[str]]:
    # Язык жёстко
    lang = LANG or "ru"

    # WAV → mono 16k
    audio, sr = _read_wav_16k_mono(path, target_sr=16000)
    if len(audio) == 0:
        return "", [], []

    # VAD → чанки
    active_vad = vad_processor
    if active_vad is None:
        active_vad = VADProcessor(
            threshold=vad_cfg.silero.threshold,
            min_speech_ms=vad_cfg.silero.min_speech_ms,
            min_silence_ms=vad_cfg.silero.min_silence_ms,
            speech_pad_ms=vad_cfg.silero.speech_pad_ms,
            max_speech_duration_s=vad_cfg.params.target_speech_sec + vad_cfg.params.max_overshoot_sec,
            max_bin_dur_sec=vad_cfg.params.max_bin_dur_sec,
            use_cuda=(vad_cfg.silero.use_cuda and torch.cuda.is_available()),
            model_dir=vad_cfg.silero.model_dir,
        )
    chunks, speech_secs = slice_with_silero_vad(sr, audio, active_vad, vad_cfg.params)

    if debug:
        print(f"[CHUNKS] {path.name}: {len(chunks)} chunks; VAD only; pack={vad_cfg.params.pack_bins}")
        for i, (s, e) in enumerate(speech_secs):
            print(f"[VAD SEG {i+1}] {s:.2f}-{e:.2f} sec")

    # Финальная проверка и разбиение чанков
    max_bin_dur_samples = int(vad_cfg.params.max_bin_dur_sec * sr)
    new_chunks = []
    new_segments = []
    for s0, s1 in chunks:
        dur_samples = s1 - s0
        if dur_samples > max_bin_dur_samples:
            start = s0
            while start < s1:
                end = min(start + max_bin_dur_samples, s1)
                new_chunks.append((start, end))
                new_segments.append({"start": start / sr, "end": end / sr, "text": ""})
                start = end
        else:
            new_chunks.append((s0, s1))
            new_segments.append({"start": s0 / sr, "end": s1 / sr, "text": ""})
    chunks = new_chunks
    segments = new_segments

    if debug:
        for i, (s0, s1) in enumerate(chunks):
            print(f"[FINAL CHUNK {i+1}] {path.name} {_format_ts(s0/sr)}–{_format_ts(s1/sr)} dur={(s1-s0)/sr:.2f} sec")

    # Транскриб
    tmpdir = repo_root / ".tmp" / "gigaam_vad_only"
    tmpdir.mkdir(parents=True, exist_ok=True)

    full_text_parts: List[str] = []
    for i, ((s0, s1), seg) in enumerate(zip(chunks, segments)):
        seg_audio = audio[s0:s1]

        # Подготовка источника: всегда используем временный файл для совместимости с gigaam
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix=f"{path.stem}_chunk_", dir=tmpdir, delete=False) as tmpf:
            sf.write(tmpf, seg_audio, sr, format="WAV")
            wav_path = Path(tmpf.name)
        try:
            with torch.inference_mode():
                out = _safe_transcribe(model, str(wav_path), lang)
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Текст из ответа
        if isinstance(out, dict):
            text = out.get("transcription") or out.get("text") or out.get("transcript") or ""
        elif isinstance(out, list):
            text = " ".join([str(it.get("transcription") or it.get("text") or it.get("transcript") or it)
                             if isinstance(it, dict) else str(it) for it in out]).strip()
        else:
            text = out if isinstance(out, str) else str(out)

        if debug:
            print(f"[CHUNK {i+1:03d}/{len(chunks)}] {path.name} {_format_ts(s0/sr)}–{_format_ts(s1/sr)} len={len(text)}")

        if keep_all:
            full_text_parts.append(text)
            seg["text"] = text
        else:
            # дедуп/плотность (не связаны с энергетикой)
            tail = "".join(full_text_parts)[-dedup_tail_chars:] if full_text_parts else ""
            def _norm_tokens(t: str) -> list[str]:
                t = unicodedata.normalize("NFKC", t)
                t = re.sub(r'[^\w\s]', '', t)  # Удаляем пунктуацию
                return t.lower().split()
            a_tokens = _norm_tokens(tail)
            b_tokens = _norm_tokens(text)
            keep_text = text
            max_k = min(len(a_tokens), len(b_tokens), 200)
            dedup_found = False
            for k in range(max_k, min_dedup_overlap - 1, -1):
                if SequenceMatcher(None, a_tokens[-k:], b_tokens[:k]).ratio() >= 0.9:
                    words = re.findall(r"\w+\b", text, flags=re.UNICODE)
                    if k < len(words):
                        idx = 0; cut_words = 0
                        while idx < len(text) and cut_words < k:
                            m = re.match(r"\w+\b", text[idx:], flags=re.UNICODE)
                            if m:
                                cut_words += 1; idx += m.end()
                            else:
                                idx += 1
                        keep_text = text[idx:].lstrip()
                    else:
                        keep_text = ""
                    dedup_found = True
                    break
            if debug and dedup_found:
                print(f"[DEDUP] Cut overlap of {k} tokens")
            dur = (s1 - s0) / sr
            accept = bool(keep_text) and _is_dense(keep_text, dur, min_wps=min_wps, min_cps=min_cps)
            if accept:
                full_text_parts.append(keep_text)
                seg["text"] = keep_text

    full_text = " ".join([t for t in full_text_parts if t]).strip()

    if debug:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    comparison_lines: List[str] = []
    if speech_secs:
        for s, e in speech_secs:
            comparison_lines.append(f"{s:.1f}-{e:.1f} Голос")

    return full_text, segments, comparison_lines


# ------------------------- Функция для обработки одного файла -------------------------

def process_single_file(
    path: Path,
    repo_root: Path,
    vad_cfg: VadConfig,
    vad_processor: Optional[VADProcessor],
    model,
    lang: Optional[str],
    dedup_tail_chars: int,
    min_dedup_overlap: int,
    debug: bool,
    use_tempfile: bool,
    min_wps: float,
    min_cps: float,
    keep_all: bool,
):
    p = Path(path)

    local_model = model
    if local_model is None:
        if gigaam is None:
            raise RuntimeError(
                "GigaAM is not installed.\n"
                "pip install gigaam (или git+https://github.com/salute-developers/GigaAM)"
            )
        local_model = gigaam.load_model(MODEL_NAME)
        if hasattr(local_model, "eval"):
            local_model.eval()
        try:
            if torch.cuda.is_available():
                if hasattr(local_model, "to"):
                    local_model = local_model.to("cuda")
                elif hasattr(local_model, "cuda"):
                    local_model = local_model.cuda()
        except Exception:
            pass

    full_text, segments, _ = transcribe_file_sequential(
        model=local_model,
        path=p,
        repo_root=repo_root,
        lang=lang,
        vad_cfg=vad_cfg,
        vad_processor=vad_processor,
        dedup_tail_chars=dedup_tail_chars,
        min_dedup_overlap=min_dedup_overlap,
        debug=debug,
        use_tempfile=use_tempfile,
        min_wps=min_wps,
        min_cps=min_cps,
        keep_all=keep_all,
    )

    return str(p), full_text, segments


# ------------------------- MAIN -------------------------

def main():
    repo_root = configure_local_caches()

    if gigaam is None:
        raise SystemExit(
            "GigaAM is not installed.\n"
            "pip install gigaam  (или git+https://github.com/salute-developers/GigaAM)\n"
            f"Import error: {_IMPORT_ERR}"
        )

    input_path = Path(INPUT).resolve()
    if not input_path.is_dir():
        raise SystemExit(f"[ERROR] Input must be a directory, got: {INPUT}")

    # VAD конфиг (общий для всех)
    vad_cfg = VadConfig(
        silero=SileroParams(
            threshold=SILERO_THRESHOLD,
            min_speech_ms=SILERO_MIN_SPEECH_MS,
            min_silence_ms=SILERO_MIN_SILENCE_MS,
            speech_pad_ms=SILERO_SPEECH_PAD_MS,
            use_cuda=SILERO_CUDA,
            model_dir=SILERO_MODEL_DIR or None,
        ),
        params=VadParams(
            target_speech_sec=TARGET_SPEECH_SEC,
            max_overshoot_sec=VAD_MAX_OVERSHOOT,
            max_silence_within_sec=VAD_MAX_SILENCE_WITHIN,
            pad_context_ms=PAD_CONTEXT_MS,
            min_gap_sec=MIN_GAP_SEC,
            merge_close_segs=MERGE_CLOSE_SEGS,
            pack_bins=VAD_PACK_BINS,
            max_bin_dur_sec=MAX_BIN_DUR_SEC,
        ),
    )

    vad_processor = VADProcessor(
        threshold=vad_cfg.silero.threshold,
        min_speech_ms=vad_cfg.silero.min_speech_ms,
        min_silence_ms=vad_cfg.silero.min_silence_ms,
        speech_pad_ms=vad_cfg.silero.speech_pad_ms,
        max_speech_duration_s=vad_cfg.params.target_speech_sec + vad_cfg.params.max_overshoot_sec,
        max_bin_dur_sec=vad_cfg.params.max_bin_dur_sec,
        use_cuda=(vad_cfg.silero.use_cuda and torch.cuda.is_available()),
        model_dir=vad_cfg.silero.model_dir,
    )

    model = gigaam.load_model(MODEL_NAME)
    if hasattr(model, "eval"):
        model.eval()
    try:
        if torch.cuda.is_available():
            if hasattr(model, "to"):
                model = model.to("cuda")
            elif hasattr(model, "cuda"):
                model = model.cuda()
    except Exception:
        pass

    out_base = Path(OUTPUT).parent
    report_base = Path(OUTPUT_REPORT).parent
    segments_base = Path(WRITE_SEGMENTS).resolve()

    # Обходим все подпапки рекурсивно
    for root, dirs, files in os.walk(input_path):
        root_path = Path(root)
        audio_files = [
            root_path / f
            for f in files
            if Path(f).suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
        ]

        if not audio_files:
            continue

        if SAMPLE and SAMPLE > 0 and len(audio_files) > SAMPLE:
            import random
            random.seed(42)
            random.shuffle(audio_files)
            audio_files = audio_files[:SAMPLE]

        results: dict[str, str] = {}
        all_dialog_segments: List[dict] = []

        relative = root_path.relative_to(input_path)
        local_segments_dir = segments_base / relative
        local_segments_dir.mkdir(parents=True, exist_ok=True)

        # Последовательная обработка файлов в текущей подпапке
        for p in audio_files:
            file_path, full_text, segments = process_single_file(
                path=p,
                repo_root=repo_root,
                vad_cfg=vad_cfg,
                vad_processor=vad_processor,
                model=model,
                lang=LANG,
                dedup_tail_chars=int(DEDUP_TAIL_CHARS),
                min_dedup_overlap=int(MIN_DEDUP_OVERLAP),
                debug=bool(DEBUG),
                use_tempfile=bool(USE_TEMPFILE),
                min_wps=float(MIN_WPS),
                min_cps=float(MIN_CPS),
                keep_all=bool(KEEP_ALL),
            )

            p = Path(file_path)
            results[p.name] = full_text

            if local_segments_dir:
                seg_file = local_segments_dir / f"{p.stem}_segments.jsonl"
                with open(seg_file, "w", encoding="utf-8") as seg_writer:
                    for s in segments:
                        obj = {
                            "audio": p.name,
                            "start": _format_ts(s["start"]),
                            "end": _format_ts(s["end"]),
                            "duration": s["end"] - s["start"],
                            "text": s["text"],
                        }
                        seg_writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        all_dialog_segments.append({
                            "audio": p.name,
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                        })

        # Проверяем, есть ли сегменты для диалога в текущей подпапке
        if not all_dialog_segments:
            if DEBUG:
                print(f"[DEBUG] No segments found for dialog in {root}")
            continue

        # Объединяем последовательные реплики одного файла
        sorted_segments = sorted(all_dialog_segments, key=lambda x: x["start"])
        merged_segments = []
        current_seg = None
        for seg in sorted_segments:
            if not seg["text"]:
                continue
            if current_seg is None:
                current_seg = seg.copy()
                continue
            if (current_seg["audio"] == seg["audio"]) and (seg["start"] - current_seg["end"] <= MAX_DIALOG_GAP_SEC):
                current_seg["text"] = current_seg["text"] + " " + seg["text"]
                current_seg["end"] = seg["end"]
            else:
                merged_segments.append(current_seg)
                current_seg = seg.copy()
        if current_seg is not None:
            merged_segments.append(current_seg)

        # Применяем пунктуацию к объединённым репликам
        if PUNCT_RU and _HAVE_RUPUNCT:
            try:
                rupunct = rp.build_punct_pipeline()
                if DEBUG:
                    print(f"[RUPunct] Model loaded for final dialog punctuation in {root}")
                for seg in merged_segments:
                    try:
                        seg["text"] = rp.punctuate_text(rupunct, seg["text"]) or seg["text"]
                    except Exception as e:
                        if DEBUG:
                            print(f"[RUPunct][WARN] Failed to punctuate segment in {root}: {e}")
            except Exception as e:
                if DEBUG:
                    print(f"[RUPunct][WARN] Cannot load RUPunct model for {root}: {e}")

        # Формируем финальный текст для results
        results = {}
        for seg in merged_segments:
            audio = seg["audio"]
            if audio not in results:
                results[audio] = []
            if seg["text"]:  # Пропускаем пустые тексты
                results[audio].append(seg["text"])
        results = {k: " ".join(v).strip() for k, v in results.items()}

        # Записываем выходные файлы для текущей подпапки
        local_out_path = out_base / relative / Path(OUTPUT).name
        local_out_path.parent.mkdir(parents=True, exist_ok=True)
        with local_out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if OUTPUT_FORMAT == "txt":
            local_report_path = report_base / relative / Path(OUTPUT_REPORT).name
            local_report_path.parent.mkdir(parents=True, exist_ok=True)
            with local_report_path.open("w", encoding="utf-8") as f:
                for p, txt in results.items():
                    f.write(f"{p}\n{txt}\n\n")

        dialog_file = local_segments_dir / f"dialog_{root_path.name}.jsonl"
        dialog_txt_file = local_segments_dir / f"dialog_{root_path.name}.txt"
        dialog_clean_txt_file = local_segments_dir / f"dialog_{root_path.name}_clean.txt"
        with dialog_file.open("w", encoding="utf-8") as jsonl_out, dialog_txt_file.open(
            "w", encoding="utf-8"
        ) as txt_out, dialog_clean_txt_file.open("w", encoding="utf-8") as clean_txt_out:
            for seg in merged_segments:
                obj = {
                    "audio": seg["audio"],
                    "start": _format_ts(seg["start"]),
                    "end": _format_ts(seg["end"]),
                    "text": seg["text"],
                }
                jsonl_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

                audio_field = seg.get("audio") or ""
                audio_name = os.path.basename(audio_field)
                speaker = Path(audio_name).stem or audio_name or "audio"

                text_value = seg.get("text")
                if text_value is None:
                    text_value = ""
                elif not isinstance(text_value, str):
                    text_value = str(text_value)

                serialized_text = json.dumps(text_value, ensure_ascii=False)
                txt_out.write(f"{speaker}: {serialized_text}\n")

                clean_text = " ".join(text_value.split())
                if clean_text:
                    clean_txt_out.write(f"{speaker}: {clean_text}\n")
                else:
                    clean_txt_out.write(f"{speaker}:\n")

    try:
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
