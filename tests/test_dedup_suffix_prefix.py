import sys
import types
from pathlib import Path

# Provide stub modules before importing the real code
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference_gigaam import _dedup_suffix_prefix


def test_dedup_handles_punctuation_and_case():
    tail = "Hello world,"
    new = "world! How are you?"
    assert _dedup_suffix_prefix(tail, new, min_overlap=1) == "How are you?"


def test_dedup_token_boundary():
    tail = "abc def"
    new = "defg hi"
    assert _dedup_suffix_prefix(tail, new, min_overlap=1) == "defg hi"


def test_dedup_nfkc_normalization():
    tail = "Café"
    new = "CAFÉ is open"
    assert _dedup_suffix_prefix(tail, new, min_overlap=1) == "is open"
