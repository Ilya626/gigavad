#!/usr/bin/env python
"""Простой двухпроходный резюмер для dialog_*.txt."""

from __future__ import annotations

# =============================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ===============================
INPUT_DIALOG: str = "out\\2025-11-14\\dialog_2025-11-14.jsonl"              # Финальный текст диалога (dialog_*.txt)
CHEAT_SHEET_FILE: str = "out\\Cheatsheet.txt"                     # Готовый cheat sheet (TXT), "" — не отправлять
OUTPUT_TEXT: str = "out\\2025-11-14\\summary_dialog_2025-11-14.txt"                         # Путь для финального текста ("" — пропустить)

MODEL_PROVIDER: str = "gemini"               # "openrouter" или "gemini"
MODEL_NAME: str = "gemini-2.5-pro"  # Имя модели выбранного провайдера
CHUNK_COUNT: int = 5                           # Сколько окон по ~20%
CHUNK_OVERLAP: int = 1                         # Перекрытие окон в сегментах
TOTAL_PASSES: int = 2                          # Первый прогон + уточнение
MAX_TOKENS: int = 20000                         # Лимит токенов на ответ
TEMPERATURE: float = 0.5                       # Температура модели
REQUEST_TIMEOUT: int = 210                     # Таймаут HTTP-запроса, сек
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
GEMINI_REQUESTS_PER_MINUTE: Optional[int] = None  # None — авто по названию модели
AUTO_CHUNK_COUNT: bool = True
TARGET_CHUNK_TOKENS: int = 17000
MAX_CHUNK_TOKENS: int = 18000
TOKEN_ENCODING: str = "cl100k_base"


SYSTEM_PROMPT: str = (
" Вы — летописец и хронист, ведущий подробный и структурированный лог игровой сессии (или главы). Ваша задача — создать подробный, объективный и насыщенный информацией пересказ, ориентированный на ключевые сюжетные повороты, развитие персонажей, магические события и дипломатические договорённости." 
"держи факты и не выдумывай события, которых нет в источнике. Ты имеешь ЧИТШИТ, а так же информацию из прошлых частей резюме. Опиши новую часть, включив её в резюме. Если её нет - прост пиши резюме.\n" 
" Инструкции по стилю и форматированию: Структура Секций: Используйте римские цифры для основных сюжетных разделов (I, II, III и т.д.), выделяя их жирным шрифтом. Названия секций должны быть краткими и отражать суть события (например, I. Праздник в Мирабаре: аномалия времени и откровения Аль'Шари)."
" Детализация: Каждая основная секция должна содержать один или несколько подпунктов, начинающихся с символа • (крупный маркер) для ключевых событий, и, при необходимости, с символа ◦ (маленький маркер) для дальнейшей детализации. Язык: Используйте формальный, насыщенный фэнтезийной терминологией язык."
" Имена персонажей (например, Церара, Йоруэль, Тэруан, Гелдбрир) и ключевые артефакты/места (например, Фейвальд, Баатор, Кинжал Песков Времени) должны быть выделены жирным шрифтом. Лаконичность: Описывайте события кратко, фокусируясь на результате, конфликте или откровении (например, «Аркебуза взрывается, раня червя осколками», или «От шока и унижения Церара инстинктивно телепортируется потоком искр, слетая с червя»)."
" Художественный пересказ В КОНЦЕ СЕКЦИИ ОТЕЛЬНОЙ ЧАСТЬЮ!: Ты пишешь красивый художественный рассказ в 3-4 абзаца по 3-5 предложений, который учитывает все указанные до этого нюансы. Пиши в третьем лице."
" Пример желаемого формата для начала: I. Название первой сюжетной точки • Ключевое действие первого персонажа: [Краткий итог]. • Конфликт или откровение: [Краткий итог]. II. Название второй сюжетной точки • Действие, приведшее к кризису: [Краткий итог]."
" Обязательные заключительные секции: После основного сюжета (Секции I, II, III...) добавьте следующие мета-секции для анализа сессии, используя те же правила форматирования (жирный шрифт, маркеры): СЮЖЕТНЫЕ ВЕТКИ КАЖДОГО ПЕРСОНАЖА: Кратко суммируйте ключевые прорывы, решения и конфликты для каждого основного персонажа, используя маркеры • и ◦ (например, «Пробуждение наследия: Ключевым событием стало спонтанное проявление её эладринских способностей»)."
" ЛУТ, ПРЕДМЕТЫ И ДОГОВОРЁННОСТИ: Отдельно суммируйте все важные соглашения, сделки, задания и полученные предметы или информацию, разделяя их на категории: ◦ Договорённости (указывай кого с кем!): (Например, «Сделка с Принцем Гулей: Предоставление золотого диска в обмен на поиск Псаря»). ◦ Предметы: (Например, «Кружка самонаполняющаяся элем»). ◦ Критически важная информация (Лут): (Включая глобальные угрозы и важные откровения, например, «Глобальная угроза Оркуса: Повелитель демонов пытается стать личом»)." 
"ПОМНИ, ТЫ ДОЛЖЕН ПИСАТЬ ПОЛНОЕ ПОДРОБНОЕ И ДЛИННОЕ ХУДОЖЕСТВЕННОЕ РЕЗЮМЕ С УЧЕТОМ ПРОШЛЫХ ЧАСТЕЙ. ТВОЙ ОТВЕТ - ПОЛНОЦЕННОЕ РЕЗЮМЕ ВКЛЮЧАЯ СТАРУЮ И НОВУЮ ЧАСТИ. ТЕБЕ ЗАПРЕЩЕНО ПРИСЫЛАТЬ ЧИТШИТ В ОТВЕТЕ. Начинай с общего названия всей главы на основании событий, описанных в ней."
)
INCLUDE_SYSTEM_PROMPT: bool = True

USER_GUIDE: str = (
" Вы — летописец и хронист, ведущий подробный и структурированный лог игровой сессии (или главы). Ваша задача — создать кподробный, объективный и насыщенный информацией пересказ, ориентированный на ключевые сюжетные повороты, развитие персонажей, магические события и дипломатические договорённости." 
"держи факты и не выдумывай события, которых нет в источнике. Ты имеешь ЧИТШИТ, а так же информацию из прошлых частей резюме. Опиши новую часть, включив её в резюме. Если её нет - прост пиши резюме.\n" 
" Инструкции по стилю и форматированию: Структура Секций: Используйте римские цифры для основных сюжетных разделов (I, II, III и т.д.), выделяя их жирным шрифтом. Названия секций должны быть краткими и отражать суть события (например, I. Праздник в Мирабаре: аномалия времени и откровения Аль'Шари)."
" Детализация: Каждая основная секция должна содержать один или несколько подпунктов, начинающихся с символа • (крупный маркер) для ключевых событий, и, при необходимости, с символа ◦ (маленький маркер) для дальнейшей детализации. Язык: Используйте формальный, насыщенный фэнтезийной терминологией язык."
" Имена персонажей (например, Церара, Йоруэль, Тэруан, Гелдбрир) и ключевые артефакты/места (например, Фейвальд, Баатор, Кинжал Песков Времени) должны быть выделены жирным шрифтом. Лаконичность: Описывайте события кратко, фокусируясь на результате, конфликте или откровении (например, «Аркебуза взрывается, раня червя осколками», или «От шока и унижения Церара инстинктивно телепортируется потоком искр, слетая с червя»)."
" Художественный пересказ В КОНЦЕ СЕКЦИИ ОТЕЛЬНОЙ ЧАСТЬЮ!: Ты пишешь красивый художественный рассказ в 3-4 абзаца по 3-5 предложений, который учитывает все указанные до этого нюансы. Пиши в третьем лице."
" Пример желаемого формата для начала: I. Название первой сюжетной точки • Ключевое действие первого персонажа: [Краткий итог]. • Конфликт или откровение: [Краткий итог]. II. Название второй сюжетной точки • Действие, приведшее к кризису: [Краткий итог]."
" Обязательные заключительные секции: После основного сюжета (Секции I, II, III...) добавьте следующие мета-секции для анализа сессии, используя те же правила форматирования (жирный шрифт, маркеры): СЮЖЕТНЫЕ ВЕТКИ КАЖДОГО ПЕРСОНАЖА: Кратко суммируйте ключевые прорывы, решения и конфликты для каждого основного персонажа, используя маркеры • и ◦ (например, «Пробуждение наследия: Ключевым событием стало спонтанное проявление её эладринских способностей»)."
" ЛУТ, ПРЕДМЕТЫ И ДОГОВОРЁННОСТИ: Отдельно суммируйте все важные соглашения, сделки, задания и полученные предметы или информацию, разделяя их на категории: ◦ Договорённости (указывай кого с кем!): (Например, «Сделка с Принцем Гулей: Предоставление золотого диска в обмен на поиск Псаря»). ◦ Предметы: (Например, «Кружка самонаполняющаяся элем»). ◦ Критически важная информация (Лут): (Включая глобальные угрозы и важные откровения, например, «Глобальная угроза Оркуса: Повелитель демонов пытается стать личом»)." 
"ПОМНИ, ТЫ ДОЛЖЕН ПИСАТЬ ПОЛНОЕ ПОДРОБНОЕ И ДЛИННОЕ ХУДОЖЕСТВЕННОЕ РЕЗЮМЕ С УЧЕТОМ ПРОШЛЫХ ЧАСТЕЙ. ТВОЙ ОТВЕТ - ПОЛНОЦЕННОЕ РЕЗЮМЕ ВКЛЮЧАЯ СТАРУЮ И НОВУЮ ЧАСТИ. ТЕБЕ ЗАПРЕЩЕНО ПРИСЫЛАТЬ ЧИТШИТ В ОТВЕТЕ"
)

PASS_LABELS: tuple[str, ...] = ("Первый проход", "Уточнение")
SECOND_PASS_SYSTEM_PROMPT: str = (
    "Ты должен уточнить и добавить детали резюме. "
    "Ты получаешь полное резюме и отдельный фрагмент исходного разговора № {fragment_label}. "
    "Твоя задача — дополнить резюме деталями из фрагмента, сохраняя структуру, "
    "точность формулировок и важные нюансы."
)
FINAL_RECHECK_MESSAGE: str = "ПРОВЕРЬ ЧТО ДАННОЕ РЕЗЮМЕ ОТФОРМАТИРОВАНО СОГЛАСНО ЗАДАЧЕ"

# ====================================================================================

import math
import json
from pathlib import Path
from typing import Any, Callable, Optional
from chunk_utils import estimate_chunk_count_from_segments

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
    plan = make_chunk_plan(segments, chunk_count, overlap, total_passes=1)
    return plan[0]


def make_chunk_plan(
    segments: list[dict[str, Any]],
    chunk_count: int,
    overlap: int,
    total_passes: int,
) -> list[list[dict[str, Any]]]:
    if not segments:
        raise ValueError("Нет сегментов для резюмирования")
    if chunk_count <= 0:
        raise ValueError("chunk_count должен быть > 0")
    if total_passes <= 0:
        raise ValueError("total_passes должен быть > 0")

    total = len(segments)
    chunk_size = max(1, math.ceil(total / chunk_count))
    safe_overlap = max(0, min(overlap, chunk_size - 1))

    offsets = _compute_chunk_offsets(total, chunk_size, total_passes)
    plans: list[list[dict[str, Any]]] = []
    for offset in offsets:
        plans.append(_make_chunks_with_offset(segments, chunk_size, safe_overlap, offset))
    return plans


def _compute_chunk_offsets(total: int, chunk_size: int, total_passes: int) -> list[int]:
    if total_passes <= 0:
        return []
    if total <= 0:
        return [0] * total_passes

    stride = max(1, chunk_size // 2)
    offsets: list[int] = []
    used: set[int] = set()
    current = 0
    for _ in range(total_passes):
        while current in used and len(used) < total:
            current = (current + 1) % total
        offsets.append(current)
        used.add(current)
        current = (current + stride) % total
    return offsets


def _make_chunks_with_offset(
    segments: list[dict[str, Any]],
    chunk_size: int,
    safe_overlap: int,
    offset: int,
) -> list[dict[str, Any]]:
    total = len(segments)
    if total == 0:
        return []

    normalized_offset = offset % total
    ranges: list[tuple[int, int]]
    if normalized_offset == 0:
        ranges = [(0, total)]
    else:
        ranges = [(normalized_offset, total), (0, normalized_offset)]

    chunks: list[dict[str, Any]] = []
    idx = 0
    for base_start, base_end in ranges:
        if base_start >= base_end:
            continue
        start = base_start
        while start < base_end:
            end = min(base_end, start + chunk_size)
            if end < base_end:
                tail_speaker = segments[end - 1].get("speaker")
                while end < base_end and segments[end].get("speaker") == tail_speaker:
                    end += 1
            idx += 1
            chunk_segments = segments[start:end]
            chunks.append({"index": idx, "segments": chunk_segments})
            if end >= base_end:
                break
            start = max(base_start, end - safe_overlap)
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
    system_prompt: str,
    chunk_total: int,
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

    try:
        fragment_index = int(chunk.get("index") or 1)
    except (TypeError, ValueError):
        fragment_index = 1
    try:
        total_fragments = int(chunk_total)
    except (TypeError, ValueError):
        total_fragments = 0
    total_fragments = max(1, total_fragments)
    fragment_label = f"{fragment_index}/{total_fragments}"
    blocks.append(f"### Фрагмент {fragment_label}\n" + format_chunk(chunk))
    blocks.append(USER_GUIDE.strip())

    user_payload = {"type": "text", "text": "\n\n".join(blocks)}
    resolved_system_prompt = (system_prompt or "").strip()
    if "{fragment_label" in resolved_system_prompt:
        try:
            resolved_system_prompt = resolved_system_prompt.format(fragment_label=fragment_label)
        except KeyError:
            pass

    if INCLUDE_SYSTEM_PROMPT and resolved_system_prompt:
        return [
            {"role": "system", "content": [{"type": "text", "text": resolved_system_prompt}]},
            {"role": "user", "content": [user_payload]},
        ]

    content_blocks: list[dict[str, str]] = []
    if resolved_system_prompt:
        content_blocks.append({"type": "text", "text": f"{resolved_system_prompt}\n\n"})
    content_blocks.append(user_payload)
    return [{"role": "user", "content": content_blocks}]



def _finalise_summary(
    client: ChatClient,
    summary: str,
    system_prompt: str,
) -> str:
    summary = (summary or "").strip()
    if not summary:
        return summary

    combined_system = (system_prompt or "").strip()
    if combined_system:
        combined_system = "{}\n\n{}".format(combined_system, FINAL_RECHECK_MESSAGE)
    else:
        combined_system = FINAL_RECHECK_MESSAGE

    conversation: list[dict[str, Any]] = []
    if INCLUDE_SYSTEM_PROMPT and combined_system:
        conversation.append(
            {"role": "system", "content": [{"type": "text", "text": combined_system}]}
        )
        conversation.append(
            {"role": "user", "content": [{"type": "text", "text": summary}]}
        )
    else:
        user_blocks: list[dict[str, str]] = []
        if combined_system:
            user_blocks.append({"type": "text", "text": combined_system + "\n\n"})
        user_blocks.append({"type": "text", "text": summary})
        conversation.append({"role": "user", "content": user_blocks})

    try:
        response = client.chat(conversation)
    except OpenRouterRateLimitError:
        raise
    except Exception as exc:
        if DEBUG:
            print(f"[Summariser] Final check skipped due to error: {exc}")
        return summary

    cleaned = (response or "").strip()
    return cleaned or summary

def run_pass(
    client: ChatClient,
    cheat_sheet: str,
    chunks: list[dict[str, Any]],
    initial_summary: Optional[str],
    pass_index: int,
    total_passes: int,
    system_prompt: str,
    start_chunk: Optional[int] = None,
    progress_callback: Optional[Callable[[str, Optional[int], int], None]] = None,
) -> str:
    summary_history: list[str] = []
    if initial_summary:
        cleaned_initial = initial_summary.strip()
        if cleaned_initial:
            summary_history.append(cleaned_initial)
    running_summary = initial_summary
    chunk_total = len(chunks) if chunks else 0
    if chunk_total <= 0:
        chunk_total = 1

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
        if (
            start_chunk is not None
            and isinstance(chunk_number, int)
            and chunk_number <= start_chunk
        ):
            continue
        previous_summary = summary_history[-1].strip() if summary_history else ""
        composed = compose_messages(
            cheat_sheet=cheat_sheet,
            chunk=chunk,
            previous_summary=previous_summary or None,
            pass_index=pass_index,
            total_passes=total_passes,
            system_prompt=system_prompt,
            chunk_total=chunk_total,
        )
        conversation: list[dict[str, Any]] = []
        if summary_history:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": summary_history[-1]}
                    ],
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
    chunk_count = CHUNK_COUNT
    segments = load_segments(input_path)
    if AUTO_CHUNK_COUNT:
        try:
            auto_count = estimate_chunk_count_from_segments(
                segments,
                target_chunk_tokens=TARGET_CHUNK_TOKENS,
                max_chunk_tokens=MAX_CHUNK_TOKENS,
                encoding_name=TOKEN_ENCODING,
            )
            chunk_count = max(1, int(auto_count))
            if DEBUG:
                print(f"[Summariser] Auto chunk count: {chunk_count}")
        except Exception as exc:
            if DEBUG:
                print(f"[Summariser] Auto chunk sizing failed: {exc}")
            chunk_count = max(1, CHUNK_COUNT)
    else:
        chunk_count = max(1, CHUNK_COUNT)

    base_chunks = make_chunks(segments, chunk_count, CHUNK_OVERLAP)
    chunk_plan: list[list[dict[str, Any]]] = [base_chunks]
    pass_prompts: list[str] = [SYSTEM_PROMPT]

    if TOTAL_PASSES >= 2:
        second_fragment_count = max(1, math.ceil(max(1, len(base_chunks)) / 2))
        second_chunks = make_chunks(segments, second_fragment_count, CHUNK_OVERLAP)
        chunk_plan.append(second_chunks)
        pass_prompts.append(SECOND_PASS_SYSTEM_PROMPT)

    while len(chunk_plan) < TOTAL_PASSES:
        chunk_plan.append(chunk_plan[-1])
    while len(pass_prompts) < TOTAL_PASSES:
        pass_prompts.append(pass_prompts[-1])

    cheat_path = Path(CHEAT_SHEET_FILE).expanduser().resolve() if CHEAT_SHEET_FILE else None
    cheat_sheet = load_cheat_sheet(cheat_path) if cheat_path else ""

    if client is None:
        client = build_client()

    resume_state = _load_resume_state(input_path)
    resume_pass: Optional[int] = None
    resume_chunk: Optional[int] = None
    resume_summary: Optional[str] = None
    resume_path: Optional[Path] = None
    if isinstance(resume_state, dict) and resume_state:
        raw_pass = resume_state.get("pass_index")
        raw_chunk = resume_state.get("chunk_index")
        raw_summary = (resume_state.get("summary") or "").strip()
        raw_path = (resume_state.get("summary_path") or "").strip()
        if isinstance(raw_pass, int) and raw_pass >= 0:
            resume_pass = raw_pass
        if isinstance(raw_chunk, int) and raw_chunk >= 0:
            resume_chunk = raw_chunk
        if raw_summary:
            resume_summary = raw_summary
        if raw_path:
            resume_path = Path(raw_path).expanduser()
        if DEBUG and resume_summary:
            chunk_note = (
                f", фрагмент {resume_chunk}"
                if isinstance(resume_chunk, int) and resume_chunk > 0
                else ""
            )
            print(
                "[Summariser] Найдено сохранённое состояние (проход %d%s)."
                % ((resume_pass or 0) + 1, chunk_note)
            )

    running_summary: Optional[str] = resume_summary
    last_saved_summary = resume_summary or ""
    last_saved_path: Optional[Path] = resume_path
    last_saved_chunk: Optional[int] = resume_chunk
    last_saved_pass: Optional[int] = resume_pass

    start_pass = 0
    if (
        resume_summary
        and isinstance(resume_pass, int)
        and 0 <= resume_pass < TOTAL_PASSES
    ):
        start_pass = resume_pass
    resume_pass_marker = resume_pass if resume_summary else None
    resume_chunk_marker = resume_chunk if resume_summary else None

    def handle_progress(summary: str, chunk_index: Optional[int], pass_no: int) -> None:
        nonlocal last_saved_summary, last_saved_path, last_saved_chunk, last_saved_pass
        trimmed = (summary or "").strip()
        if not trimmed or trimmed == last_saved_summary:
            return
        path = _save_partial_summary(input_path, trimmed)
        if path is not None:
            last_saved_path = path
        if isinstance(chunk_index, int):
            last_saved_chunk = chunk_index
        last_saved_pass = pass_no
        last_saved_summary = trimmed
        _store_resume_state(
            input_path,
            last_saved_summary,
            last_saved_pass,
            last_saved_chunk,
            last_saved_path,
        )
        if DEBUG:
            chunk_note = (
                f", фрагмент {chunk_index}"
                if isinstance(chunk_index, int) and chunk_index > 0
                else ""
            )
            print(
                "[Summariser] Промежуточный результат сохранён (проход %d%s): %s"
                % (pass_no + 1, chunk_note, last_saved_path)
            )

    for pass_index in range(start_pass, TOTAL_PASSES):
        if pass_index < len(chunk_plan):
            chunks = chunk_plan[pass_index]
        else:
            chunks = chunk_plan[-1]
        if pass_index < len(pass_prompts):
            current_prompt = pass_prompts[pass_index]
        else:
            current_prompt = pass_prompts[-1]
        start_chunk = (
            resume_chunk_marker if resume_pass_marker == pass_index else None
        )
        try:
            running_summary = run_pass(
                client=client,
                cheat_sheet=cheat_sheet,
                chunks=chunks,
                initial_summary=running_summary,
                pass_index=pass_index,
                total_passes=TOTAL_PASSES,
                system_prompt=current_prompt,
                start_chunk=start_chunk,
                progress_callback=handle_progress,
            )
            if resume_pass_marker == pass_index:
                resume_pass_marker = None
                resume_chunk_marker = None
        except OpenRouterRateLimitError as exc:
            summary = exc.partial_summary or (running_summary or "")
            summary = summary.strip()
            path: Optional[Path] = None
            if summary and summary != last_saved_summary:
                path = _save_partial_summary(input_path, summary)
                last_saved_summary = summary
                if path is not None:
                    last_saved_path = path
            store_pass = last_saved_pass if last_saved_pass is not None else pass_index
            store_chunk = last_saved_chunk
            store_summary = summary or last_saved_summary
            if store_summary:
                _store_resume_state(
                    input_path,
                    store_summary,
                    store_pass,
                    store_chunk,
                    last_saved_path or path,
                )
                last_saved_pass = store_pass
                last_saved_chunk = store_chunk
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
                            f"[Summariser] Промежуточный итог сохранён в файл: {path}"
                        )
            elif last_saved_path is not None and DEBUG:
                print(
                    f"[Summariser] Используем ранее сохранённый результат: {last_saved_path}"
                )
            store_pass = last_saved_pass if last_saved_pass is not None else pass_index
            store_chunk = last_saved_chunk
            store_summary = summary or last_saved_summary
            if store_summary:
                _store_resume_state(
                    input_path,
                    store_summary,
                    store_pass,
                    store_chunk,
                    last_saved_path or path,
                )
                last_saved_pass = store_pass
                last_saved_chunk = store_chunk
            raise
    final_output = (running_summary or "").strip()
    if final_output:
        base_prompt = pass_prompts[0] if pass_prompts else SYSTEM_PROMPT
        try:
            running_summary = _finalise_summary(client, final_output, base_prompt)
        except OpenRouterRateLimitError:
            raise
        except Exception as exc:
            if DEBUG:
                print(f"[Summariser] Final check skipped due to error: {exc}")
            running_summary = final_output
    else:
        running_summary = final_output

    _clear_resume_state(input_path)
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
        requests_per_minute = GEMINI_REQUESTS_PER_MINUTE
        if requests_per_minute is None:
            requests_per_minute = _auto_gemini_rpm(MODEL_NAME)
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
            max_requests_per_minute=requests_per_minute,
        )

    raise SystemExit("Неизвестный MODEL_PROVIDER. Используйте 'openrouter' или 'gemini'.")


def _auto_gemini_rpm(model_name: str) -> int:
    name = (model_name or "").strip().lower()
    if "flash" in name:
        return 5
    if "pro" in name:
        return 1
    return 1


def _resume_state_path(input_path: Path) -> Path:
    base = Path(OUTPUT_TEXT).expanduser() if OUTPUT_TEXT else input_path
    try:
        resolved = base.expanduser().resolve()
    except Exception:
        resolved = base.expanduser()
    return resolved.parent / f"{resolved.name}.state.json"


def _load_resume_state(input_path: Path) -> dict[str, Any]:
    path = _resume_state_path(input_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if DEBUG:
            print(f"[Summariser] Resume state read failed: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _store_resume_state(
    input_path: Path,
    summary: str,
    pass_index: Optional[int],
    chunk_index: Optional[int],
    summary_path: Optional[Path],
) -> None:
    payload: dict[str, Any] = {
        "summary": (summary or "").strip(),
        "pass_index": int(pass_index) if pass_index is not None else None,
        "chunk_index": int(chunk_index) if chunk_index is not None else None,
    }
    if summary_path:
        payload["summary_path"] = str(summary_path)
    path = _resume_state_path(input_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        if DEBUG:
            print(f"[Summariser] Resume state write failed: {exc}")


def _clear_resume_state(input_path: Path) -> None:
    path = _resume_state_path(input_path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception as exc:
        if DEBUG:
            print(f"[Summariser] Resume state cleanup failed: {exc}")


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
