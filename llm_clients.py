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
    """Protocol describing the behaviour expected from chat clients."""

    def chat(self, messages: list[dict[str, Any]]) -> str:
        ...


class _MessageUtils:
    """Utility helpers shared across concrete client implementations."""

    @staticmethod
    def _dry_stub(messages: list[dict[str, Any]]) -> str:
        """Return a short preview of the latest user message during dry-run."""
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

    @staticmethod
    def _normalise_content(content: Any) -> str:
        """Collapse different content representations into a plain string."""
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

    @classmethod
    def _prepare_openrouter_messages(
        cls, messages: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Transform the internal message format into OpenRouter payload items."""
        prepared: list[dict[str, str]] = []
        for raw in messages:
            role = str(raw.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            prepared.append(
                {
                    "role": role,
                    "content": cls._normalise_content(raw.get("content")),
                }
            )
        return prepared


class GeminiClient(_MessageUtils):
    """Google Gemini client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: float = 120.0,
        dry_run: bool = False,
        debug: bool = False,
        thinking_mode: str = "default",
        thinking_budget: Optional[int] = None,
        include_thoughts: bool = False,
        max_requests_per_minute: Optional[int] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
        retry_max_delay: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or ""
        self.max_tokens = max_tokens if (max_tokens or 0) > 0 else None
        self.temperature = temperature
        self.timeout = timeout
        self.dry_run = dry_run
        self.debug = debug
        self.thinking_mode = (thinking_mode or "").strip().lower()
        self.thinking_budget = thinking_budget
        self.include_thoughts = bool(include_thoughts)
        self.max_requests_per_minute = (
            max(0, int(max_requests_per_minute)) if max_requests_per_minute else 0
        )
        self._retry_attempts = max(1, int(max_retries))
        self._retry_base_delay = max(0.1, float(retry_base_delay))
        self._retry_max_delay = max(self._retry_base_delay, float(retry_max_delay))
        self._recent_requests: deque[float] = deque()

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if self.dry_run:
            return self._dry_stub(messages)

        if not self.api_key:
            raise RuntimeError(
                "Gemini API key is missing. Set DRY_RUN=True to avoid live requests."
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = self._build_payload(messages)
        try:
            payload["generationConfig"] = self._make_generation_config()
        except ValueError as exc:
            raise RuntimeError(f"Invalid Gemini thinking configuration: {exc}") from exc

        attempts = max(1, self._retry_attempts)

        for attempt in range(1, attempts + 1):
            if self.debug:
                print(
                    f"[Gemini] POST model={self.model} messages={len(messages)} "
                    f"attempt={attempt}"
                )
            try:
                self._respect_rate_limit()
                resp = requests.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt >= attempts:
                    raise RuntimeError(f"Gemini request failed: {exc}") from exc
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[Gemini] Request error {exc!s}. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            if resp.status_code == 429:
                if attempt >= attempts:
                    raise RuntimeError("Gemini returned 429 Too Many Requests")
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[Gemini] 429 Too Many Requests. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            if 500 <= resp.status_code < 600:
                text_body = resp.text.strip()
                details = f" {text_body}" if text_body else ""
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Gemini server error {resp.status_code}.{details}"
                    )
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[Gemini] {resp.status_code} server error. Retrying in "
                        f"{delay:.1f}s ({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                text_body = resp.text.strip()
                details = f" {text_body}" if text_body else ""
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Gemini HTTP error {resp.status_code}.{details}"
                    ) from exc
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[Gemini] HTTP {resp.status_code}. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            try:
                data = resp.json()
            except ValueError as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        "Gemini responded with invalid JSON payload"
                    ) from exc
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[Gemini] Invalid JSON response. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            candidates = data.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                if attempt >= attempts:
                    raise RuntimeError("Gemini returned no candidates.")
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[Gemini] Empty candidates list. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            for candidate in candidates:
                text_value = self._extract_candidate_text(candidate)
                if text_value:
                    return text_value

            if attempt >= attempts:
                break
            delay = self._retry_delay(attempt)
            if self.debug:
                print(
                    f"[Gemini] No usable candidate text. Retrying in {delay:.1f}s "
                    f"({attempt}/{attempts})"
                )
            time.sleep(delay)

        raise RuntimeError("Gemini returned no usable text candidates.")

    def _extract_output_text(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_instruction_parts: list[str] = []

        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            text = self._normalise_content(message.get("content"))
            if role == "system":
                if text:
                    system_instruction_parts.append(text)
                continue
            if role == "assistant":
                target_role = "model"
            else:
                target_role = "user"
            parts = [{"text": text}] if text else [{"text": ""}]
            contents.append({"role": target_role, "parts": parts})

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_instruction_parts)}]
            }
        if not contents:
            payload["contents"] = [{"role": "user", "parts": [{"text": ""}]}]
        return payload

    def _extract_candidate_text(self, candidate: Any) -> str:
        if not isinstance(candidate, dict):
            return ""
        content = candidate.get("content")
        parts = None
        if isinstance(content, dict):
            parts = content.get("parts")
        elif isinstance(content, list):
            parts = content
        texts: list[str] = []
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if part.get("thought"):
                    continue
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if texts:
            return "\n".join(texts).strip()
        return ""

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
                    f"[Gemini] Rate limit {self.max_requests_per_minute} req/min. "
                    f"Waiting {wait_time:.1f}s"
                )
            time.sleep(wait_time)

    def _retry_delay(self, attempt: int) -> float:
        safe_attempt = max(1, attempt)
        base = self._retry_base_delay * (2 ** (safe_attempt - 1))
        delay = min(self._retry_max_delay, base)
        jitter = random.uniform(0.0, self._retry_base_delay)
        return min(self._retry_max_delay, delay + jitter)

    def _make_generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"temperature": float(self.temperature)}
        if self.max_tokens and self.max_tokens > 0:
            config["maxOutputTokens"] = int(self.max_tokens)
        thinking_config = self._build_thinking_config()
        if thinking_config is not None:
            config["thinkingConfig"] = thinking_config
        return config

    def _build_thinking_config(self) -> Optional[dict[str, Any]]:
        mode = self.thinking_mode
        config: dict[str, Any] = {}
        budget: Optional[int] = None

        if mode in {"off", "disable", "disabled"}:
            budget = 0
        elif mode in {"dynamic", "auto"}:
            budget = -1
        elif mode in {"fixed", "manual"}:
            budget = self._resolve_thinking_budget()
            if budget is None:
                raise ValueError(
                    "thinking_budget must be provided for fixed/manual thinking mode."
                )
            if budget <= 0:
                raise ValueError(
                    "thinking_budget must be greater than zero for fixed/manual mode."
                )
        elif mode in {"default", ""}:
            budget = self._resolve_thinking_budget()
            if budget is not None and budget < 0 and budget != -1:
                raise ValueError(
                    "thinking_budget must be positive, -1, or None when mode is default."
                )
        else:
            raise ValueError(
                "Unsupported thinking mode. Use default, dynamic, off, fixed, or manual."
            )

        if mode not in {"default", "", "fixed", "manual"}:
            config["thinkingBudget"] = budget
        elif budget is not None:
            config["thinkingBudget"] = budget

        if self.include_thoughts:
            config["includeThoughts"] = True

        return config or None

    def _resolve_thinking_budget(self) -> Optional[int]:
        value = self.thinking_budget
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("thinking_budget must not be a boolean.")
        try:
            budget = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("thinking_budget must be an integer value.") from exc
        return budget


class OpenRouterClient(_MessageUtils):
    """OpenRouter client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
        dry_run: bool = False,
        debug: bool = False,
        site_url: Optional[str] = None,
        referer: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 30.0,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ) -> None:
        self.model = model
        self.api_key = api_key or ""
        self.max_tokens = max_tokens if (max_tokens or 0) > 0 else None
        self.temperature = temperature
        self.timeout = timeout
        self.dry_run = dry_run
        self.debug = debug
        self.site_url = site_url
        self.referer = referer
        self.base_url = base_url
        self._retry_attempts = max(1, int(max_retries))
        self._retry_base_delay = max(0.1, float(retry_base_delay))
        self._retry_max_delay = max(self._retry_base_delay, float(retry_max_delay))

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if self.dry_run:
            return self._dry_stub(messages)

        if not self.api_key:
            raise RuntimeError(
                "OpenRouter API key is missing. Set DRY_RUN=True to avoid live requests."
            )

        prepared_messages = self._prepare_openrouter_messages(messages)
        payload: dict[str, Any] = {"model": self.model, "messages": prepared_messages}
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = int(self.max_tokens)
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)

        attempts = max(1, self._retry_attempts)

        for attempt in range(1, attempts + 1):
            if self.debug:
                print(
                    f"[OpenRouter] POST model={self.model} messages={len(prepared_messages)} "
                    f"attempt={attempt}"
                )
                if prepared_messages:
                    preview = json.dumps(prepared_messages[-1], ensure_ascii=False)
                    print(f"[OpenRouter] Last message preview: {preview}")
            try:
                resp = requests.post(
                    self.base_url,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt >= attempts:
                    raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[OpenRouter] Request error {exc!s}. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            if resp.status_code == 429:
                message = (
                    self._extract_error_message(resp)
                    or "OpenRouter rate limit reached."
                )
                retry_after = self._retry_after_seconds(resp)
                if retry_after is not None and self.debug:
                    print(
                        f"[OpenRouter] 429 Too Many Requests. Retry after ~{retry_after:.1f}s"
                    )
                error = OpenRouterRateLimitError(message, attempt)
                if retry_after is not None:
                    error.retry_after = retry_after  # type: ignore[attr-defined]
                raise error

            if 500 <= resp.status_code < 600:
                text_body = resp.text.strip()
                details = f" {text_body}" if text_body else ""
                if attempt >= attempts:
                    raise RuntimeError(
                        f"OpenRouter server error {resp.status_code}.{details}"
                    )
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[OpenRouter] {resp.status_code} server error. Retrying in "
                        f"{delay:.1f}s ({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                message = self._extract_error_message(resp)
                details = f" {message}" if message else ""
                if attempt >= attempts:
                    raise RuntimeError(
                        f"OpenRouter HTTP error {resp.status_code}.{details}"
                    ) from exc
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[OpenRouter] HTTP {resp.status_code}. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            try:
                data = resp.json()
            except ValueError as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        "OpenRouter responded with invalid JSON payload."
                    ) from exc
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[OpenRouter] Invalid JSON response. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                if attempt >= attempts:
                    raise RuntimeError("OpenRouter returned no choices.")
                delay = self._retry_delay(attempt)
                if self.debug:
                    print(
                        f"[OpenRouter] Empty choices list. Retrying in {delay:.1f}s "
                        f"({attempt}/{attempts})"
                    )
                time.sleep(delay)
                continue

            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                text = self._normalise_content(content)
                if text:
                    return text

            if attempt >= attempts:
                break
            delay = self._retry_delay(attempt)
            if self.debug:
                print(
                    f"[OpenRouter] No usable choice text. Retrying in {delay:.1f}s "
                    f"({attempt}/{attempts})"
                )
            time.sleep(delay)

        raise RuntimeError("OpenRouter returned no usable text choices.")

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["Referer"] = self.referer
            headers["HTTP-Referer"] = self.referer
        if self.site_url:
            headers["X-Title"] = self.site_url
        return headers

    def _retry_delay(self, attempt: int) -> float:
        safe_attempt = max(1, attempt)
        base = self._retry_base_delay * (2 ** (safe_attempt - 1))
        delay = min(self._retry_max_delay, base)
        jitter = random.uniform(0.0, self._retry_base_delay)
        return min(self._retry_max_delay, delay + jitter)

    def _retry_after_seconds(self, response: requests.Response) -> Optional[float]:
        retry_after = response.headers.get("Retry-After")
        if not retry_after:
            retry_after = (
                response.headers.get("X-RateLimit-Reset")
                or response.headers.get("RateLimit-Reset")
            )
        if not retry_after:
            return None
        retry_after = retry_after.strip()
        if not retry_after:
            return None
        try:
            value = float(retry_after)
            if value >= 0:
                return value
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            dt = None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = dt - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds())

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text.strip()
        error = data.get("error")
        if isinstance(error, str):
            return error.strip()
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if isinstance(message, str):
                return message.strip()
        detail = data.get("message")
        if isinstance(detail, str):
            return detail.strip()
        return response.text.strip()


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
            print(f"[Config] Missing config file: {config_display}")
        return None

    parser = ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            parser.read_file(fp)
    except (OSError, ConfigParserError) as exc:
        raise RuntimeError(
            f"Unable to read configuration file {config_display}: {exc}"
        ) from exc

    if not parser.has_section(section):
        if debug:
            print(
                f"[Config] Section '{section}' not found in {config_display} "
                f"for {provider_label}"
            )
        return None

    value = parser.get(section, key, fallback="").strip()
    if not value:
        if debug:
            print(
                f"[Config] Key '{key}' in section '{section}' is empty for "
                f"{provider_label}"
            )
        return None
    return value
