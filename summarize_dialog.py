#!/usr/bin/env python
"""Простой двухпроходный резюмер для dialog_*.txt."""

from __future__ import annotations

# =============================== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ===============================
INPUT_DIALOG: str = "dialog.txt"              # Финальный текст диалога (dialog_*.txt)
CHEAT_SHEET_FILE: str = ""                     # Готовый cheat sheet (TXT), "" — не отправлять
OUTPUT_TEXT: str = ""                         # Путь для финального текста ("" — пропустить)

MODEL_NAME: str = "deepseek/deepseek-r1-0528:free"  # Модель OpenRouter
CHUNK_COUNT: int = 5                           # Сколько окон по ~20%
CHUNK_OVERLAP: int = 1                         # Перекрытие окон в сегментах
TOTAL_PASSES: int = 2                          # Первый прогон + уточнение
MAX_TOKENS: int = 1200                         # Лимит токенов на ответ
TEMPERATURE: float = 0.3                       # Температура модели
REQUEST_TIMEOUT: int = 120                     # Таймаут HTTP-запроса, сек
DRY_RUN: bool = False                          # True — не звонить в OpenRouter
DEBUG: bool = True                             # Печатать служебные сообщения

CONFIG_FILE: str = "summarize_config.ini"      # Файл с ключом OpenRouter
CONFIG_SECTION: str = "openrouter"            # Секция в конфиге
CONFIG_KEY: str = "api_key"                   # Ключ с API-ключом

SYSTEM_PROMPT: str = (
    "Ты — аналитик настольных RPG-сессий. Собирай структурированные резюме,"
    " держи факты и индексы и не выдумывай события, которых нет в источнике."
)

USER_GUIDE: str = (
    "Формат ответа:\n"
    "1. Хронология ключевых событий.\n"
    "2. Решения и конфликты.\n"
    "3. Персонажи, артефакты, термины.\n"
    "4. Открытые вопросы.\n"
    "Если встречается новое имя — фиксируй его."
)

PASS_LABELS: tuple[str, ...] = ("Первый проход", "Уточнение")
# ====================================================================================

import math
import time
from configparser import ConfigParser, Error as ConfigParserError
from pathlib import Path
from typing import Any, Optional

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
        self.retry_delays: tuple[int, ...] = (5, 10, 15, 20, 25, 30)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if self.dry_run:
            return self._dry_stub(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/salute-developers/gigavad",
            "X-Title": "GigaVAD dialogue summariser",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        model_name = self.model.lower()
        if model_name.startswith("deepseek/deepseek-r1"):
            payload.setdefault("reasoning", {"effort": "medium"})
            payload.setdefault("include_reasoning", True)

        retry_index = 0
        attempt = 0

        while True:
            attempt += 1
            if self.debug:
                print(
                    f"[OpenRouter] POST model={self.model} messages={len(messages)} "
                    f"attempt={attempt}"
                )
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Ошибка запроса OpenRouter: {exc}") from exc

            if resp.status_code == 429:
                delay = (
                    self.retry_delays[retry_index]
                    if retry_index < len(self.retry_delays)
                    else None
                )
                if delay is None:
                    if not self._ask_more_attempts():
                        err = OpenRouterRateLimitError(
                            "OpenRouter ответил 429 Too Many Requests", attempt
                        )
                        raise err
                    retry_index = 1
                    delay = self.retry_delays[0]
                else:
                    retry_index += 1
                if self.debug:
                    print(
                        f"[OpenRouter] 429 Too Many Requests, ждём {delay} с перед повтором"
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
                data = resp.json()
            except ValueError as exc:
                raise RuntimeError("Не удалось распарсить JSON от OpenRouter") from exc

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("OpenRouter вернул пустой ответ")
            message = choices[0].get("message") or {}
            content = self._extract_output_text(message)
            if not content:
                raise RuntimeError("OpenRouter ответ без content")
            return content

    def _dry_stub(self, messages: list[dict[str, Any]]) -> str:
        text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = msg.get("content", "")
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
            if not raw or ":" not in raw:
                continue
            speaker_part, text_part = raw.split(":", 1)
            speaker = speaker_part.strip() or "Unknown"
            text_value = text_part.strip()
            if text_value.startswith("\"") and text_value.endswith("\""):
                text_value = text_value[1:-1]
            if not text_value:
                continue
            segments.append(
                {
                    "speaker": speaker,
                    "text": text_value.strip(),
                    "index": len(segments) + 1,
                }
            )
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
) -> list[dict[str, str]]:
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

    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def run_pass(
    client: OpenRouterClient,
    cheat_sheet: str,
    chunks: list[dict[str, Any]],
    initial_summary: Optional[str],
    pass_index: int,
    total_passes: int,
) -> str:
    running_summary = initial_summary
    for chunk in chunks:
        messages = compose_messages(
            cheat_sheet=cheat_sheet,
            chunk=chunk,
            previous_summary=running_summary,
            pass_index=pass_index,
            total_passes=total_passes,
        )
        try:
            running_summary = client.chat(messages)
        except OpenRouterRateLimitError as exc:
            exc.partial_summary = (running_summary or "").strip()
            index = chunk.get("index")
            exc.failed_chunk = int(index) if isinstance(index, int) else None
            raise
    return running_summary or ""


def summarise_dialogue(input_path: Path, client: Optional[OpenRouterClient] = None) -> str:
    segments = load_segments(input_path)
    chunks = make_chunks(segments, CHUNK_COUNT, CHUNK_OVERLAP)

    cheat_path = Path(CHEAT_SHEET_FILE).expanduser().resolve() if CHEAT_SHEET_FILE else None
    cheat_sheet = load_cheat_sheet(cheat_path) if cheat_path else ""

    if client is None:
        client = build_client()

    running_summary: Optional[str] = None
    for pass_index in range(TOTAL_PASSES):
        try:
            running_summary = run_pass(
                client=client,
                cheat_sheet=cheat_sheet,
                chunks=chunks,
                initial_summary=running_summary,
                pass_index=pass_index,
                total_passes=TOTAL_PASSES,
            )
        except OpenRouterRateLimitError as exc:
            summary = exc.partial_summary or (running_summary or "")
            summary = summary.strip()
            path = _save_partial_summary(input_path, summary)
            exc.partial_path = path
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
