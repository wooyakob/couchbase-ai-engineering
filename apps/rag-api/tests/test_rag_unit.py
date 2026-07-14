"""Unit tests for app/rag.py's _clean_standalone — the defense against weaker
instruction-following models (observed with Capella's default Llama-3.1-8B)
ignoring "one line, never an answer" in the condense step (Ch. 6 §6.4)."""

from app.rag import _clean_standalone


def test_passes_through_clean_single_line():
    assert _clean_standalone("what to do after rotating credentials", "fallback") == \
        "what to do after rotating credentials"


def test_strips_wrapping_double_quotes():
    assert _clean_standalone('"Rotate credentials"', "fallback") == "Rotate credentials"


def test_strips_wrapping_single_quotes():
    assert _clean_standalone("'Rotate credentials'", "fallback") == "Rotate credentials"


def test_strips_surrounding_whitespace():
    assert _clean_standalone("  \n  what to do next  \n  ", "fallback") == "what to do next"


def test_falls_back_on_multiline_output():
    raw = "Rotate credentials:\n\n1. Create a new credential\n2. Deploy it"
    assert _clean_standalone(raw, "and what do I do right after?") == \
        "and what do I do right after?"


def test_falls_back_on_empty_output():
    assert _clean_standalone("", "original question") == "original question"


def test_falls_back_on_whitespace_only_output():
    assert _clean_standalone("   \n   ", "original question") == "original question"
