from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence


def _load_encoder(encoding_name: Optional[str] = None):
    try:
        import tiktoken
    except ModuleNotFoundError:
        return None

    if encoding_name:
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding_name: Optional[str] = None) -> int:
    encoder = _load_encoder(encoding_name)
    if encoder is None:
        # Fallback heuristic: assume ~4 characters per token
        return max(1, (len(text) + 3) // 4)
    return len(encoder.encode(text))


def count_tokens_from_parts(parts: Iterable[str], encoding_name: Optional[str] = None) -> int:
    encoder = _load_encoder(encoding_name)
    if encoder is None:
        total_chars = sum(len(part or "") for part in parts)
        return max(1, (total_chars + 3) // 4)
    total = 0
    for part in parts:
        if not part:
            continue
        total += len(encoder.encode(part))
    return max(1, total)


def estimate_chunk_count(
    total_tokens: int,
    *,
    target_chunk_tokens: int,
    max_chunk_tokens: Optional[int] = None,
) -> int:
    target = max(1, target_chunk_tokens)
    chunks = max(1, (total_tokens + target - 1) // target)
    if max_chunk_tokens and max_chunk_tokens > 0:
        # ensure each chunk does not exceed max_chunk_tokens
        while total_tokens / chunks > max_chunk_tokens and chunks < total_tokens:
            chunks += 1
    return chunks or 1


def estimate_chunk_count_from_segments(
    segments: Sequence[dict[str, object]],
    *,
    target_chunk_tokens: int,
    max_chunk_tokens: Optional[int] = None,
    encoding_name: Optional[str] = None,
) -> int:
    texts = [
        str(segment.get("text", ""))
        for segment in segments
        if isinstance(segment, dict)
    ]
    total_tokens = count_tokens_from_parts(texts, encoding_name=encoding_name)
    return estimate_chunk_count(
        total_tokens,
        target_chunk_tokens=target_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
    )


def estimate_chunk_count_from_file(
    path: Path,
    *,
    target_chunk_tokens: int,
    max_chunk_tokens: Optional[int] = None,
    encoding_name: Optional[str] = None,
) -> int:
    text = path.read_text(encoding="utf-8")
    total_tokens = count_tokens(text, encoding_name=encoding_name)
    return estimate_chunk_count(
        total_tokens,
        target_chunk_tokens=target_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
    )

