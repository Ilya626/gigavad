#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply Russian punctuation & capitalization with RUPunct_big to JSONL segments.

Input JSONL lines like:
{"audio":"Ilya_1_h.wav","start":0.0,"end":22.33,"text":"..."}
Output JSONL adds: "text_punct": "..."

Also writes transcript_punct.json -> {"Ilya_1_h.wav": "full punctuated text"}.
"""

import argparse, json, sys, re, logging
from pathlib import Path
from collections import defaultdict

from transformers import pipeline, AutoTokenizer

# ---- model setup (per the HF model card) ----
MODEL_ID = "RUPunct/RUPunct_big"  # https://huggingface.co/RUPunct/RUPunct_big

log = logging.getLogger("RUPunct")

def build_punct_pipeline():
    # Per model card: add_prefix_space=True, strip_accents=False
    tk = AutoTokenizer.from_pretrained(MODEL_ID, strip_accents=False, add_prefix_space=True)
    clf = pipeline("ner", model=MODEL_ID, tokenizer=tk, aggregation_strategy="first")
    return clf

# ---- label -> transformation mapping (mirrors the HF card exactly) ----
def process_token(token: str, label: str) -> str:
    # LOWER_*
    if label == "LOWER_O":                  return token
    if label == "LOWER_PERIOD":             return token + "."
    if label == "LOWER_COMMA":              return token + ","
    if label == "LOWER_QUESTION":           return token + "?"
    if label == "LOWER_TIRE":               return token + "—"
    if label == "LOWER_DVOETOCHIE":         return token + ":"
    if label == "LOWER_VOSKL":              return token + "!"
    if label == "LOWER_PERIODCOMMA":        return token + ";"
    if label == "LOWER_DEFIS":              return token + "-"
    if label == "LOWER_MNOGOTOCHIE":        return token + "..."
    if label == "LOWER_QUESTIONVOSKL":      return token + "?!"

    # UPPER_* (capitalize first letter)
    if label == "UPPER_O":                  return token.capitalize()
    if label == "UPPER_PERIOD":             return token.capitalize() + "."
    if label == "UPPER_COMMA":              return token.capitalize() + ","
    if label == "UPPER_QUESTION":           return token.capitalize() + "?"
    if label == "UPPER_TIRE":               return token.capitalize() + " —"
    if label == "UPPER_DVOETOCHIE":         return token.capitalize() + ":"
    if label == "UPPER_VOSKL":              return token.capitalize() + "!"
    if label == "UPPER_PERIODCOMMA":        return token.capitalize() + ";"
    if label == "UPPER_DEFIS":              return token.capitalize() + "-"
    if label == "UPPER_MNOGOTOCHIE":        return token.capitalize() + "..."
    if label == "UPPER_QUESTIONVOSKL":      return token.capitalize() + "?!"

    # UPPER_TOTAL_* (ALL CAPS)
    if label == "UPPER_TOTAL_O":            return token.upper()
    if label == "UPPER_TOTAL_PERIOD":       return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":     return token.upper() + "?"
    if label == "UPPER_TOTAL_TIRE":         return token.upper() + " —"
    if label == "UPPER_TOTAL_DVOETOCHIE":   return token.upper() + ":"
    if label == "UPPER_TOTAL_VOSKL":        return token.upper() + "!"
    if label == "UPPER_TOTAL_PERIODCOMMA":  return token.upper() + ";"
    if label == "UPPER_TOTAL_DEFIS":        return token.upper() + "-"
    if label == "UPPER_TOTAL_MNOGOTOCHIE":  return token.upper() + "..."
    if label == "UPPER_TOTAL_QUESTIONVOSKL":return token.upper() + "?!"

    # fallback
    return token

# simple detok fixes for spaces
_re_fix_spaces = [
    (re.compile(r"\s+([,.;:!?…])"), r"\1"),       # no space before punctuation
    (re.compile(r"\s+—\s*"), " — "),              # normalize em dash spaces
    (re.compile(r"\s{2,}"), " "),                 # squeeze multiple spaces
]

def postprocess_spaces(text: str) -> str:
    out = text.strip()
    for rx, rep in _re_fix_spaces:
        out = rx.sub(rep, out)
    return out.strip()

def _find_split_index(text: str) -> int:
    """Locate a whitespace character closest to the middle of *text*."""
    if len(text) < 2:
        return -1
    mid = len(text) // 2
    # try to find whitespace scanning left from the midpoint first, then right
    for idx in range(mid, 0, -1):
        if text[idx - 1].isspace():
            return idx - 1
    for idx in range(mid, len(text)):
        if text[idx].isspace():
            return idx
    return -1


def _apply_model(clf, text: str) -> str:
    preds = clf(text)
    parts = []
    for it in preds:
        token = (it.get("word") or "").strip()
        label = it.get("entity_group") or "LOWER_O"
        parts.append(process_token(token, label))
    return postprocess_spaces(" ".join(parts))


def punctuate_text(clf, text: str) -> str:
    if not text or not text.strip():
        return text

    def _punct_segment(segment: str, depth: int = 0) -> str:
        try:
            return _apply_model(clf, segment)
        except Exception as exc:  # pragma: no cover - defensive against HF pipeline failures
            split_idx = _find_split_index(segment)
            if split_idx == -1:
                log.error(
                    "RUPunct failed to punctuate segment (len=%d) with no split available",
                    len(segment),
                    exc_info=True,
                )
                raise
            left = segment[:split_idx].rstrip()
            right = segment[split_idx:].lstrip()
            if not left or not right:
                log.error(
                    "RUPunct failed to split segment cleanly (len=%d)",
                    len(segment),
                    exc_info=True,
                )
                raise
            log.warning(
                "RUPunct pipeline fallback: splitting text (len=%d, depth=%d) after %s",
                len(segment),
                depth,
                exc,
            )
            left_punct = _punct_segment(left, depth + 1)
            right_punct = _punct_segment(right, depth + 1)
            return postprocess_spaces(f"{left_punct} {right_punct}")

    cleaned = text.strip()
    try:
        return _punct_segment(cleaned)
    except Exception:
        log.error(
            "RUPunct pipeline failed even after attempting splits (len=%d)",
            len(cleaned),
            exc_info=True,
        )
        return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input segments JSONL")
    ap.add_argument("--out", dest="outp", required=True, help="Output segments JSONL with text_punct")
    ap.add_argument("--merge_json", dest="merge_json", default="", help="Optional merged transcript JSON")
    ap.add_argument("--batch", type=int, default=16, help="(not used; kept for symmetry)")
    args = ap.parse_args()

    clf = build_punct_pipeline()  # loads tokenizer + model, per HF card. :contentReference[oaicite:1]{index=1}

    in_path  = Path(args.inp)
    out_path = Path(args.outp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = defaultdict(list)
    total, nonempty = 0, 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            raw = obj.get("text", "") or ""
            punct = punctuate_text(clf, raw)
            obj["text_punct"] = punct
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if punct and punct.strip():
                nonempty += 1
                audio = obj.get("audio", "UNKNOWN")
                start = float(obj.get("start", 0.0))
                merged[audio].append((start, punct))

    if args.merge_json:
        # sort by start and join with spaces; keep very light normalization
        merged_out = {}
        for audio, items in merged.items():
            items.sort(key=lambda x: x[0])
            text = postprocess_spaces(" ".join(t for _, t in items))
            merged_out[audio] = text
        mp = Path(args.merge_json)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(merged_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[RUPunct] processed segments: {total}, non-empty: {nonempty}")
    print(f"[RUPunct] wrote: {out_path}")
    if args.merge_json:
        print(f"[RUPunct] merged transcript: {args.merge_json}")

if __name__ == "__main__":
    main()
