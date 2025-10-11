#!/usr/bin/env python
"""Простой двухпроходный резюмер для dialog_*.txt."""

from __future__ import annotations

# =============================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ===============================
INPUT_DIALOG: str = "out\\2025-05-25\\dialog_2025-05-25.jsonl"              # Финальный текст диалога (dialog_*.txt)
CHEAT_SHEET_FILE: str = "out\\19_09\\Cheat_sheet.txt"                     # Готовый cheat sheet (TXT), "" — не отправлять
OUTPUT_TEXT: str = "out\\2025-05-25\\summary_2025-05-25.txt"                         # Путь для финального текста ("" — пропустить)

MODEL_PROVIDER: str = "openrouter"               # "openrouter" или "gemini"
MODEL_NAME: str = "tngtech/deepseek-r1t2-chimera:free"  # Имя модели выбранного провайдера
CHUNK_COUNT: int = 3                           # Сколько окон по ~20%
CHUNK_OVERLAP: int = 1                         # Перекрытие окон в сегментах
TOTAL_PASSES: int = 1                          # Первый прогон + уточнение
MAX_TOKENS: int = 6000                         # Лимит токенов на ответ
TEMPERATURE: float = 0.3                       # Температура модели
REQUEST_TIMEOUT: int = 60                     # Таймаут HTTP-запроса, сек
DRY_RUN: bool = False                          # True — не звонить в OpenRouter
DEBUG: bool = True                             # Печатать служебные сообщения

CONFIG_FILE: str = "summarize_config.ini"      # Файл с ключом OpenRouter
CONFIG_SECTION: str = "openrouter"            # Секция в конфиге OpenRouter
CONFIG_KEY: str = "api_key"                   # Ключ с API-ключом OpenRouter
GEMINI_CONFIG_SECTION: str = "gemini"         # Секция в конфиге Gemini
GEMINI_CONFIG_KEY: str = "api_key"            # Ключ с API-ключом Gemini
GEMINI_THINKING_MODE: str = "dynamic"         # default, dynamic, off, fixed/manual
GEMINI_THINKING_BUDGET: Optional[int] = None   # Токены для режима fixed или переопределения default
GEMINI_INCLUDE_THOUGHTS: bool = False          # True — запросить summary мыслей (не включается в ответ)

SYSTEM_PROMPT: str = (
    "Ты — аналитик настольных RPG-сессий. Собирай структурированные резюме,"
    " держи факты и индексы и не выдумывай события, которых нет в источнике. Ты имеешь читшит, а так же информацию из прошлых частей резюме. Опиши новую часть, включив её в резюме. Если её нет - прост пиши резюме.\n"
    "Сделай мне резюме по событиям по следующей структуре. Ты обязан указывать все факты, детали и нюансы.\n"
    "Блок Сцены/События (по римским разделам): краткие сцены, контекст, участники, ставки. Каждая сцена фиксирует: кто участвовал, что произошло, к чему ведёт.\n"
    "Блок Сюжетные ветки персонажей: для каждого ключевого лица — 2–4 подпункта (статус, мотивация, разворот, зависимости). ветки содержат: мотивацию, поворот \n"
    "Блок Договорённости и предметы: договорённости, предметы, условия, ограничения, статус, условия, проверяемые факты/источники.\n"
    "ПОМНИ, ТЫ ДОЛЖЕН ПИСАТЬ ПОЛНОЕ ПОДРОБНОЕ И ДЛИННОЕ РЕЗЮМЕ С УЧЕТОМ ПРОШЛЫХ ЧАСТЕЙ. ТВОЙ ОТВЕТ - ПОЛНОЦЕННОЕ РЕЗЮМЕ ВКЛЮЧАЯ СТАРУЮ И НОВУЮ ЧАСТИ. ТЕБЕ ЗАПРЕЩЕНО ПРИСЫЛАТЬ ЧИТШИТ В ОТВЕТЕ"
)
INCLUDE_SYSTEM_PROMPT: bool = True

USER_GUIDE: str = (
    "Ты — аналитик настольных RPG-сессий. Собирай структурированные резюме,"
    " держи факты и индексы и не выдумывай события, которых нет в источнике. Ты имеешь читшит, а так же информацию из прошлых частей резюме. Опиши новую часть, включив её в резюме. Если её нет - прост пиши резюме.\n"
    "Сделай мне резюме по событиям по следующей структуре. Ты обязан указывать все факты, детали и нюансы. \n"
    "Блок Сцены/События (по римским разделам): краткие сцены, контекст, участники, ставки. Каждая сцена фиксирует: кто участвовал, что произошло, к чему ведёт.\n"
    "Блок Сюжетные ветки персонажей: для каждого ключевого лица — 2–4 подпункта (статус, мотивация, разворот, зависимости). ветки содержат: мотивацию, поворот \n"
    "Блок Договорённости и предметы: договорённости, предметы, условия, ограничения, статус, условия, проверяемые факты/источники.\n"
    "ПОМНИ, ТЫ ДОЛЖЕН ПИСАТЬ ПОЛНОЕ ПОДРОБНОЕ И ДЛИННОЕ РЕЗЮМЕ С УЧЕТОМ ПРОШЛЫХ ЧАСТЕЙ. ТВОЙ ОТВЕТ - ПОЛНОЦЕННОЕ РЕЗЮМЕ ВКЛЮЧАЯ СТАРУЮ И НОВУЮ ЧАСТИ. ТЕБЕ ЗАПРЕЩЕНО ПРИСЫЛАТЬ ЧИТШИТ В ОТВЕТЕ"
)

PASS_LABELS: tuple[str, ...] = ("Первый проход", "Уточнение")
# ====================================================================================

import math
import json
from pathlib import Path
from typing import Any, Callable, Optional

from llm_clients import (
    ChatClient,
    GeminiClient,
    OpenRouterClient,
    OpenRouterRateLimitError,
    load_gemini_api_key,
    load_openrouter_api_key,
)


def load_segments(path: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                if ":" not in raw:
                    continue
                speaker_part, text_part = raw.split(":", 1)
                speaker = speaker_part.strip() or "Unknown"
                text_value = text_part.strip()
                if text_value.startswith('"') and text_value.endswith('"'):
                    text_value = text_value[1:-1]
                if not text_value:
                    continue
                data = {"speaker": speaker, "text": text_value}
            text_value = data.get("text")
            if not isinstance(text_value, str):
                continue
            text_value = text_value.strip()
            if not text_value:
                continue
            speaker_value = data.get("speaker") or data.get("audio") or "Unknown"
            if not isinstance(speaker_value, str):
                speaker_value = "Unknown"
            speaker_value = speaker_value.strip() or "Unknown"
            segment: dict[str, Any] = {
                "speaker": speaker_value,
                "text": text_value,
                "index": len(segments) + 1,
            }
            for extra_key in ("audio", "start", "end"):
                extra_value = data.get(extra_key)
                if extra_value is not None:
                    segment[extra_key] = extra_value
            segments.append(segment)
    return segments

def format_chunk(chunk: dict[str, Any]) -> str:
    segments = chunk.get("segments") or []
    lines = []
    for seg in segments:
        index = int(seg.get("index") or 0)
        prefix = f"[#{index:04d}]"
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        lines.append(f"{prefix} {speaker}: {text}")
    return "\n".join(lines)


def make_chunks(segments: list[dict[str, Any]], chunk_count: int, overlap: int) -> list[dict[str, Any]]:
    if not segments:
        raise ValueError("Нет сегментов для резюмирования")
    if chunk_count <= 0:
        raise ValueError("chunk_count должен быть > 0")
    total = len(segments)
    chunk_size = max(1, math.ceil(total / chunk_count))
    safe_overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0

    while start < total:
        end = min(total, start + chunk_size)
        if end < total:
            tail_speaker = segments[end - 1].get("speaker")
            while end < total and segments[end].get("speaker") == tail_speaker:
                end += 1
        idx += 1
        chunk_segments = segments[start:end]
        chunks.append({"index": idx, "segments": chunk_segments})
        if end >= total:
            break
        start = max(0, end - safe_overlap)
    return chunks


def load_cheat_sheet(path: Optional[Path]) -> str:
    if not path:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"Cheat sheet не найден: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return fp.read().strip()


def compose_messages(
    cheat_sheet: str,
    chunk: dict[str, Any],
    previous_summary: Optional[str],
    pass_index: int,
    total_passes: int,
) -> list[dict[str, Any]]:
    if pass_index < len(PASS_LABELS):
        pass_label = PASS_LABELS[pass_index]
    else:
        pass_label = f"Проход {pass_index + 1}"

    blocks = []
    if cheat_sheet:
        blocks.append("### Читшит\n" + cheat_sheet)
    blocks.append(f"### Текущий проход: {pass_label} ({pass_index + 1}/{total_passes})")

    if previous_summary:
        blocks.append("### Актуальное резюме\n" + previous_summary.strip())
    else:
        blocks.append("### Актуальное резюме\n(пусто — начни конспект)")

    blocks.append(f"### Фрагмент #{chunk['index']}\n" + format_chunk(chunk))
    blocks.append(USER_GUIDE.strip())

    user_payload = {"type": "text", "text": "\n\n".join(blocks)}
    system_prompt = SYSTEM_PROMPT.strip()

    if INCLUDE_SYSTEM_PROMPT and system_prompt:
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [user_payload]},
        ]

    content_blocks: list[dict[str, str]] = []
    if system_prompt:
        content_blocks.append({"type": "text", "text": f"{system_prompt}\n\n"})
    content_blocks.append(user_payload)
    return [{"role": "user", "content": content_blocks}]


def run_pass(
    client: ChatClient,
    cheat_sheet: str,
    chunks: list[dict[str, Any]],
    initial_summary: Optional[str],
    pass_index: int,
    total_passes: int,
    progress_callback: Optional[Callable[[str, Optional[int], int], None]] = None,
) -> str:
    summary_history: list[str] = []
    if initial_summary:
        cleaned_initial = initial_summary.strip()
        if cleaned_initial:
            summary_history.append(cleaned_initial)
    running_summary = initial_summary

    def emit_progress(text: Optional[str], chunk_no: Optional[int]) -> None:
        if progress_callback is None:
            return
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return
        progress_callback(cleaned_text, chunk_no, pass_index)

    for chunk in chunks:
        raw_index = chunk.get("index")
        if isinstance(raw_index, int):
            chunk_number: Optional[int] = raw_index
        else:
            try:
                chunk_number = int(str(raw_index))
            except (TypeError, ValueError):
                chunk_number = None
        previous_summary = "\n\n".join(summary_history).strip()
        composed = compose_messages(
            cheat_sheet=cheat_sheet,
            chunk=chunk,
            previous_summary=previous_summary or None,
            pass_index=pass_index,
            total_passes=total_passes,
        )
        conversation: list[dict[str, Any]] = []
        for past_summary in summary_history:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": past_summary}],
                }
            )
        conversation.extend(composed)
        try:
            response = client.chat(conversation)
        except OpenRouterRateLimitError as exc:
            exc.partial_summary = previous_summary
            exc.failed_chunk = chunk_number if isinstance(chunk_number, int) else None
            emit_progress(running_summary, chunk_number)
            raise
        except Exception:
            emit_progress(running_summary, chunk_number)
            raise
        running_summary = response
        cleaned = (response or "").strip()
        if cleaned:
            summary_history.append(cleaned)
            emit_progress(cleaned, chunk_number)
    return running_summary or ""


def summarise_dialogue(input_path: Path, client: Optional[ChatClient] = None) -> str:
    segments = load_segments(input_path)
    chunks = make_chunks(segments, CHUNK_COUNT, CHUNK_OVERLAP)

    cheat_path = Path(CHEAT_SHEET_FILE).expanduser().resolve() if CHEAT_SHEET_FILE else None
    cheat_sheet = load_cheat_sheet(cheat_path) if cheat_path else ""

    if client is None:
        client = build_client()

    running_summary: Optional[str] = None
    last_saved_summary = ""
    last_saved_path: Optional[Path] = None

    def handle_progress(summary: str, chunk_index: Optional[int], pass_no: int) -> None:
        nonlocal last_saved_summary, last_saved_path
        trimmed = (summary or "").strip()
        if not trimmed or trimmed == last_saved_summary:
            return
        path = _save_partial_summary(input_path, trimmed)
        last_saved_summary = trimmed
        if path is not None:
            last_saved_path = path
            if DEBUG:
                chunk_note = (
                    f", фрагмент {chunk_index}"
                    if isinstance(chunk_index, int) and chunk_index > 0
                    else ""
                )
                print(
                    "[Summariser] Промежуточный результат сохранён (проход %d%s): %s"
                    % (pass_no + 1, chunk_note, path)
                )

    for pass_index in range(TOTAL_PASSES):
        try:
            running_summary = run_pass(
                client=client,
                cheat_sheet=cheat_sheet,
                chunks=chunks,
                initial_summary=running_summary,
                pass_index=pass_index,
                total_passes=TOTAL_PASSES,
                progress_callback=handle_progress,
            )
        except OpenRouterRateLimitError as exc:
            summary = exc.partial_summary or (running_summary or "")
            summary = summary.strip()
            path: Optional[Path] = None
            if summary and summary != last_saved_summary:
                path = _save_partial_summary(input_path, summary)
                last_saved_summary = summary
                if path is not None:
                    last_saved_path = path
            exc.partial_path = last_saved_path or path
            raise
        except Exception:
            summary = (running_summary or "").strip()
            path: Optional[Path] = None
            if summary and summary != last_saved_summary:
                path = _save_partial_summary(input_path, summary)
                last_saved_summary = summary
                if path is not None:
                    last_saved_path = path
                    if DEBUG:
                        print(
                            f"[Summariser] Прогресс сохранён перед ошибкой: {path}"
                        )
            elif last_saved_path is not None and DEBUG:
                print(
                    f"[Summariser] Используем ранее сохранённый прогресс: {last_saved_path}"
                )
            raise
    return running_summary or ""


def build_client() -> ChatClient:
    config_path = Path(CONFIG_FILE)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    provider = (MODEL_PROVIDER or "").strip().lower()

    if provider in {"", "openrouter"}:
        api_key = load_openrouter_api_key(
            config_path,
            section=CONFIG_SECTION,
            key=CONFIG_KEY,
            debug=DEBUG,
        )
        return OpenRouterClient(
            model=MODEL_NAME,
            api_key=api_key,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT,
            dry_run=DRY_RUN,
            debug=DEBUG,
        )
    if provider == "gemini":
        api_key = load_gemini_api_key(
            config_path,
            section=GEMINI_CONFIG_SECTION,
            key=GEMINI_CONFIG_KEY,
            debug=DEBUG,
        )
        return GeminiClient(
            model=MODEL_NAME,
            api_key=api_key,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=REQUEST_TIMEOUT,
            dry_run=DRY_RUN,
            debug=DEBUG,
            thinking_mode=GEMINI_THINKING_MODE,
            thinking_budget=GEMINI_THINKING_BUDGET,
            include_thoughts=GEMINI_INCLUDE_THOUGHTS,
        )

    raise SystemExit("Неизвестный MODEL_PROVIDER. Используйте 'openrouter' или 'gemini'.")


def _save_partial_summary(input_path: Path, summary: str) -> Optional[Path]:
    summary = (summary or "").strip()
    if not summary:
        return None
    if OUTPUT_TEXT:
        target = Path(OUTPUT_TEXT).expanduser()
    else:
        stem = input_path.stem or input_path.name
        suffix = input_path.suffix
        if suffix:
            name = f"{stem}.partial{suffix}"
        else:
            name = f"{input_path.name}.partial"
        target = input_path.with_name(name)
    _write_text(target, summary)
    return target


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write(value)


def main() -> None:
    if not INPUT_DIALOG:
        raise SystemExit("INPUT_DIALOG пуст — укажите путь к dialog_*.txt")

    input_path = Path(INPUT_DIALOG).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Не найден входной файл: {input_path}")

    if DEBUG:
        print(f"[Summariser] Работаем с {input_path}")

    client = build_client()

    try:
        final_summary = summarise_dialogue(input_path, client)
    except OpenRouterRateLimitError as exc:
        chunk_note = f" на фрагменте #{exc.failed_chunk}" if exc.failed_chunk else ""
        print(
            f"[OpenRouter] Превышен лимит запросов{chunk_note} после {exc.attempts} попыток."
        )
        if exc.partial_path:
            print(f"[OpenRouter] Частичное резюме сохранено: {exc.partial_path}")
        elif exc.partial_summary.strip():
            print("[OpenRouter] Частичное резюме:\n")
            print(exc.partial_summary.strip())
        raise SystemExit(1)

    if OUTPUT_TEXT:
        _write_text(Path(OUTPUT_TEXT).expanduser(), final_summary)

    print("Итоговое резюме:\n")
    print(final_summary)


if __name__ == "__main__":
    main()