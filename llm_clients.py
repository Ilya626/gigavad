"""LLM client implementations for dialogue summarisation."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from configparser import ConfigParser, Error as ConfigParserError
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import requests


__all__ = [
    "ChatClient",
    "GeminiClient",
    "OpenRouterClient",
    "OpenRouterRateLimitError",
    "load_gemini_api_key",
    "load_openrouter_api_key",
]


class OpenRouterRateLimitError(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.partial_summary: str = ""
        self.failed_chunk: Optional[int] = None
        self.partial_path: Optional[Path] = None


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]]) -> str:
        ...


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
                    err = OpenRouterRateLimitError(
                        "OpenRouter ответил 429 Too Many Requests", attempt
                    )
                    raise err
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
                        err = OpenRouterRateLimitError(
                            "OpenRouter ответил 429 Too Many Requests", attempt
                        )
                        raise err
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

class GeminiClient:
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

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if self.dry_run:
            return self._dry_stub(messages)

        if not self.api_key:
            raise RuntimeError(
                "Gemini API-ключ не найден. Укажите его в конфиге или включите DRY_RUN."
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        payload = self._build_payload(messages)
        generation_config: dict[str, Any] = {"temperature": float(self.temperature)}
        if self.max_tokens and self.max_tokens > 0:
            generation_config["maxOutputTokens"] = int(self.max_tokens)
        payload["generationConfig"] = generation_config

        if self.debug:
            print(
                f"[Gemini] POST model={self.model} messages={len(messages)} generationConfig={generation_config}"
            )

        try:
            resp = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Ошибка запроса Gemini: {exc}") from exc

        if resp.status_code == 429:
            raise RuntimeError("Gemini ответил 429 Too Many Requests")

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            text = resp.text.strip()
            details = f" {text}" if text else ""
            raise RuntimeError(
                f"Gemini запрос завершился ошибкой {resp.status_code}.{details}"
            ) from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                "Gemini вернул ответ, который не удалось распарсить как JSON"
            ) from exc

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini вернул пустой ответ без candidates")

        for candidate in candidates:
            text = self._extract_candidate_text(candidate)
            if text:
                return text

        raise RuntimeError("Gemini не вернул текст в ответе")

    def _dry_stub(self, messages: list[dict[str, Any]]) -> str:
        text = ""
        for msg in reversed(messages):
            if (msg.get("role") or "").lower() != "user":
                continue
            content = msg.get("content", "")
            text = self._normalise_content(content)
            if text:
                break
        preview = text.splitlines()[-1] if text else ""
        return "[dry-run]" if not preview else f"[dry-run] {preview[:120]}".strip()

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system_instruction_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = (msg.get("role") or "").lower()
            text = self._normalise_content(msg.get("content"))
            if not text:
                continue
            if role == "system":
                system_instruction_parts.append(text)
                continue
            mapped_role = "model" if role == "assistant" else "user"
            contents.append({"role": mapped_role, "parts": [{"text": text}]})

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_instruction_parts)}]
            }
        if not contents:
            payload["contents"] = [{"role": "user", "parts": [{"text": ""}]}]
        return payload

    def _normalise_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    def _extract_candidate_text(self, candidate: Any) -> str:
        if not isinstance(candidate, dict):
            return ""
        content = candidate.get("content")
        if isinstance(content, dict):
            parts = content.get("parts")
        else:
            parts = None
        texts: list[str] = []
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return "\n".join(texts).strip()
        return ""


def load_openrouter_api_key(
    config_path: Path, *, section: str, key: str, debug: bool = False
) -> Optional[str]:
    return _load_api_key(
        config_path=config_path,
        section=section,
        key=key,
        provider_label="OpenRouter",
        debug=debug,
    )


def load_gemini_api_key(
    config_path: Path, *, section: str, key: str, debug: bool = False
) -> Optional[str]:
    return _load_api_key(
        config_path=config_path,
        section=section,
        key=key,
        provider_label="Gemini",
        debug=debug,
    )


def _load_api_key(
    *,
    config_path: Path,
    section: str,
    key: str,
    provider_label: str,
    debug: bool,
) -> Optional[str]:
    config_path = config_path.expanduser()
    config_display = str(config_path)
    if not config_path.exists():
        if debug:
            print(f"[Config] Не найден файл конфига: {config_display}")
        return None

    parser = ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            parser.read_file(fp)
    except (OSError, ConfigParserError) as exc:
        raise RuntimeError(f"Не удалось прочитать конфиг {config_display}: {exc}") from exc

    if not parser.has_section(section):
        if debug:
            print(
                f"[Config] Нет секции '{section}' в {config_display} для {provider_label}"
            )
        return None

    value = parser.get(section, key, fallback="").strip()
    if not value:
        if debug:
            print(
                f"[Config] В секции '{section}' отсутствует значение '{key}' для {provider_label}"
            )
        return None
    return value
