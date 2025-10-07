#!/usr/bin/env python
"""Простой двухпроходный резюмер для dialog_*.txt."""

from __future__ import annotations

# =============================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ===============================
INPUT_DIALOG: str = "out\\2025-05-25\\dialog_2025-05-25.jsonl"              # Финальный текст диалога (dialog_*.txt)
CHEAT_SHEET_FILE: str = "out\\19_09\\Cheat_sheet.txt"                     # Готовый cheat sheet (TXT), "" — не отправлять
OUTPUT_TEXT: str = "out\\2025-05-25\\summary_2025-05-25.txt"                         # Путь для финального текста ("" — пропустить)

MODEL_NAME: str = "tngtech/deepseek-r1t2-chimera:free"  # Модель OpenRouter
CHUNK_COUNT: int = 3                           # Сколько окон по ~20%
CHUNK_OVERLAP: int = 1                         # Перекрытие окон в сегментах
TOTAL_PASSES: int = 1                          # Первый прогон + уточнение
MAX_TOKENS: int = 6000                         # Лимит токенов на ответ
TEMPERATURE: float = 0.3                       # Температура модели
REQUEST_TIMEOUT: int = 60                     # Таймаут HTTP-запроса, сек
DRY_RUN: bool = False                          # True — не звонить в OpenRouter
DEBUG: bool = True                             # Печатать служебные сообщения

CONFIG_FILE: str = "summarize_config.ini"      # Файл с ключом OpenRouter
CONFIG_SECTION: str = "openrouter"            # Секция в конфиге
CONFIG_KEY: str = "api_key"                   # Ключ с API-ключом

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
import random
import time
from collections import deque
from configparser import ConfigParser, Error as ConfigParserError
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Optional

import requests


class OpenRouterRateLimitError(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.partial_summary: str = ""
        self.failed_chunk: Optional[int] = None
        self.partial_path: Optional[Path] = None


class OpenRouterClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        max_tokens: int,
        temperature: float,
        timeout: int,
        dry_run: bool,
        debug: bool,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.dry_run = dry_run or not api_key
        self.debug = debug
        self.retry_base_delay: float = 10.0
        self.retry_max_delay: float = 60.0
        self.retry_min_delay: float = 5.0
        self.max_rate_limit_attempts: int = 100
        self.max_requests_per_minute: int = 20
        self._recent_requests: deque[float] = deque()

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if self.dry_run:
            return self._dry_stub(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Title": "GigaVAD dialogue summariser",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        model_name = self.model.lower()
        if "deepseek-r1" in model_name:
            payload.setdefault("reasoning", {"effort": "medium"})
            payload.setdefault("include_reasoning", True)

        retry_index = 0
        attempt = 0
        parse_retry_used = False
        raw_dumps: list[Path] = []

        while True:
            attempt += 1
            if self.debug:
                print(
                    f"[OpenRouter] POST model={self.model} messages={len(messages)} "
                    f"attempt={attempt}"
                )
            try:
                self._respect_rate_limit()
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Ошибка запроса OpenRouter: {exc}") from exc

            if resp.status_code == 429:
                retry_index += 1
                if retry_index > self.max_rate_limit_attempts:
                    if not self._ask_more_attempts():
                        err = OpenRouterRateLimitError(
                            "OpenRouter ответил 429 Too Many Requests", attempt
                        )
                        raise err
                    retry_index = 1
                delay, header_value, from_header = self._calculate_retry_delay(
                    resp, retry_index - 1
                )
                message = (
                    "Retry-After=%r" % header_value
                    if from_header
                    else "Retry-After отсутствует"
                )
                if self.debug:
                    print(
                        "[OpenRouter] 429 Too Many Requests, %s, ждём %.1f с перед повтором"
                        % (message, delay)
                    )
                time.sleep(delay)
                continue

            retry_index = 0

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                text = resp.text.strip()
                details = f" {text}" if text else ""
                raise RuntimeError(
                    f"OpenRouter запрос завершился ошибкой {resp.status_code}.{details}"
                ) from exc

            try:
                data = self._load_json_response(resp)
            except ValueError as exc:
                raw_body = resp.text or ""
                preview = self._make_preview(raw_body)
                raw_path = self._save_raw_response(raw_body, attempt)
                if raw_path is not None:
                    raw_dumps.append(raw_path)
                if self.debug:
                    print(
                        "[OpenRouter] Ответ не в формате JSON, попытка %d" % attempt
                    )
                    if preview:
                        print(preview)
                    if raw_path is not None:
                        print(f"[OpenRouter] Сырой ответ сохранён: {raw_path}")
                if not parse_retry_used:
                    parse_retry_used = True
                    if self.debug:
                        print("[OpenRouter] Повторяем запрос после ошибки парсинга")
                    continue
                message = "Не удалось распарсить JSON от OpenRouter"
                if preview:
                    message += ". Фрагмент ответа:\n" + preview
                if raw_dumps:
                    saved_list = "\n".join(f"- {path}" for path in raw_dumps)
                    message += "\nСырые ответы сохранены:\n" + saved_list
                raise RuntimeError(message) from exc

            error_payload = data.get("error")
            if isinstance(error_payload, dict):
                code = str(error_payload.get("code") or "").lower()
                message = error_payload.get("message") or "OpenRouter вернул ошибку"
                if code == "rate_limit_exceeded":
                    retry_index += 1
                    if retry_index > self.max_rate_limit_attempts:
                        if not self._ask_more_attempts():
                            err = OpenRouterRateLimitError(
                                "OpenRouter ответил 429 Too Many Requests", attempt
                            )
                            raise err
                        retry_index = 1
                    delay, header_value, from_header = self._calculate_retry_delay(
                        resp, retry_index - 1
                    )
                    message_extra = (
                        "Retry-After=%r" % header_value
                        if from_header
                        else "Retry-After отсутствует"
                    )
                    if self.debug:
                        print(
                            "[OpenRouter] rate_limit_exceeded, %s, ждём %.1f с перед повтором"
                            % (message_extra, delay)
                        )
                    time.sleep(delay)
                    continue
                raise RuntimeError(message)

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("OpenRouter вернул пустой ответ")
            message = choices[0].get("message") or {}
            content = self._extract_output_text(message)
            if not content:
                raise RuntimeError("OpenRouter ответ без content")
            return content

    def _load_json_response(self, resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError:
            body = resp.text or ""
            lines: list[str] = []
            for raw in body.splitlines():
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(("event:", "id:", "retry:")):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                lines.append(line)
            payload = "\n".join(lines) or body.strip()
            decoder = json.JSONDecoder()
            index = 0
            objects: list[dict[str, Any]] = []
            length = len(payload)
            while index < length:
                while index < length and payload[index].isspace():
                    index += 1
                if index >= length:
                    break
                try:
                    parsed, offset = decoder.raw_decode(payload, index)
                except json.JSONDecodeError:
                    index += 1
                    continue
                if isinstance(parsed, dict):
                    objects.append(parsed)
                index = offset
            if not objects:
                raise ValueError("Ответ OpenRouter не является JSON")

            message: Optional[dict[str, Any]] = None
            role = "assistant"
            pieces: list[str] = []

            def push_text(value: Any) -> None:
                if isinstance(value, str):
                    if value:
                        pieces.append(value)
                    return
                if isinstance(value, list):
                    for item in value:
                        push_text(item)
                    return
                if isinstance(value, dict):
                    kind = (value.get("type") or "").lower()
                    if kind in {"reasoning", "thinking"}:
                        return
                    push_text(value.get("text"))
                    push_text(value.get("content"))
                    return

            for obj in objects:
                choices = obj.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        candidate = choice.get("message")
                        if isinstance(candidate, dict):
                            message = candidate
                            break
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            new_role = delta.get("role")
                            if isinstance(new_role, str) and new_role:
                                role = new_role
                            push_text(delta.get("content"))
                            push_text(delta.get("text"))
                    if message is not None:
                        break
                direct = obj.get("message")
                if isinstance(direct, dict):
                    message = direct
                    break
                response = obj.get("response")
                if isinstance(response, dict):
                    push_text(response.get("output"))
                    push_text(response.get("text"))
                    candidate = response.get("message")
                    if isinstance(candidate, dict):
                        message = candidate
                        break
            if message is not None:
                data = {"choices": [{"message": message}]}
            elif pieces:
                data = {
                    "choices": [
                        {"message": {"role": role, "content": "".join(pieces)}}
                    ]
                }
            else:
                data = objects[0]
        if isinstance(data, dict):
            return data
        raise ValueError("Ответ OpenRouter имеет неподдерживаемый формат")

    def _make_preview(self, raw: str, limit: int = 500) -> str:
        text = (raw or "").strip()
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def _save_raw_response(self, body: str, attempt: int) -> Optional[Path]:
        if not body:
            return None
        directory = Path("openrouter_raw")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_attempt{attempt}"
        candidate = directory / f"{base_name}.txt"
        counter = 1
        while candidate.exists():
            candidate = directory / f"{base_name}_{counter}.txt"
            counter += 1
        try:
            candidate.write_text(body, encoding="utf-8")
        except OSError:
            return None
        return candidate

    def _respect_rate_limit(self) -> None:
        if self.max_requests_per_minute <= 0:
            return
        window = 60.0
        queue = self._recent_requests
        while True:
            now = time.monotonic()
            while queue and now - queue[0] > window:
                queue.popleft()
            if len(queue) < self.max_requests_per_minute:
                queue.append(now)
                return
            wait_time = window - (now - queue[0])
            if wait_time <= 0:
                queue.append(now)
                return
            if self.debug:
                print(
                    "[OpenRouter] Достигнут локальный лимит %d req/min, ждём %.1f с"
                    % (self.max_requests_per_minute, wait_time)
                )
            time.sleep(wait_time)

    def _calculate_retry_delay(
        self, resp: requests.Response, attempt_index: int
    ) -> tuple[float, str, bool]:
        header_value = resp.headers.get("Retry-After", "")
        retry_after = self._parse_retry_after(header_value)
        if retry_after is not None and retry_after > 0:
            return retry_after, header_value, True
        delay = self._exponential_backoff(attempt_index)
        return delay, header_value, False

    def _parse_retry_after(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        raw = value.strip()
        if not raw:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            try:
                retry_dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError, IndexError):
                return None
            if retry_dt is None:
                return None
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (retry_dt - now).total_seconds()
            return max(delta, 0.0)
        else:
            return max(seconds, 0.0)

    def _exponential_backoff(self, attempt_index: int) -> float:
        safe_index = max(0, attempt_index)
        delay = self.retry_base_delay * (2 ** safe_index)
        delay = min(delay, self.retry_max_delay)
        jitter = random.uniform(0.8, 1.2)
        delay *= jitter
        delay = max(self.retry_min_delay, min(delay, self.retry_max_delay))
        return delay

    def _dry_stub(self, messages: list[dict[str, Any]]) -> str:
        text = ""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
                break
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("text")
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                if parts:
                    text = "\n".join(parts)
                    break
        preview = text.splitlines()[-1] if text else ""
        return "[dry-run]" if not preview else f"[dry-run] {preview[:120]}".strip()

    def _extract_output_text(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if (item.get("type") or "").lower() == "thinking":
                    continue
                text = (item.get("text") or "").strip()
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts).strip()
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return ""

    def _ask_more_attempts(self) -> bool:
        prompt = "OpenRouter продолжает слать 429. Повторить ещё 5 попыток? [y/N]: "
        while True:
            try:
                reply = input(prompt)
            except EOFError:
                return False
            if reply is None:
                return False
            answer = reply.strip().lower()
            if not answer:
                return False
            if answer in {"y", "yes", "д", "да"}:
                return True
            if answer in {"n", "no", "н", "нет"}:
                return False
            print("Введите Y или N.")


def load_openrouter_api_key(config_path: Path) -> Optional[str]:
    config_path = config_path.expanduser()
    config_display = str(config_path)
    if not config_path.exists():
        if DEBUG:
            print(f"[Config] Не найден файл конфига: {config_display}")
        return None

    parser = ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            parser.read_file(fp)
    except (OSError, ConfigParserError) as exc:
        raise RuntimeError(f"Не удалось прочитать конфиг {config_display}: {exc}") from exc

    if not parser.has_section(CONFIG_SECTION):
        if DEBUG:
            print(f"[Config] Нет секции '{CONFIG_SECTION}' в {config_display}")
        return None

    value = parser.get(CONFIG_SECTION, CONFIG_KEY, fallback="").strip()
    if not value:
        if DEBUG:
            print(
                f"[Config] В секции '{CONFIG_SECTION}' отсутствует значение '{CONFIG_KEY}'"
            )
        return None
    return value




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
    client: OpenRouterClient,
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


def summarise_dialogue(input_path: Path, client: Optional[OpenRouterClient] = None) -> str:
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


def build_client() -> OpenRouterClient:
    config_path = Path(CONFIG_FILE)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    api_key = load_openrouter_api_key(config_path)
    return OpenRouterClient(
        model=MODEL_NAME,
        api_key=api_key,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT,
        dry_run=DRY_RUN,
        debug=DEBUG,
    )


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