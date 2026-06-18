"""
pdf_extractor.py
----------------
PDF text extraction and Unicode noise filtering for Ragxiv.

Two cleaners are exposed at different strictness levels:

  clean_pdf_text(text)   — Moderate cleaning for raw PDF content.
                           Preserves Latin Extended (author names), Greek
                           letters (math notation), and common math symbols
                           while stripping Vietnamese tone-marked sequences,
                           Private Use Area glyphs, and other encoding noise.
                           Kept strict because PDF encoding noise is a real,
                           model-independent problem.

  clean_llm_output(text) — Light cleaning for LLM-generated responses.
                           Strips only genuine encoding noise: C0/C1 control
                           characters, Private Use Area glyphs, and Unicode
                           non-characters.  The previous strict Korean+ASCII
                           allowlist was a qwen2.5-era workaround for Chinese/
                           Vietnamese output leakage; EXAONE 3.5 is Korean-
                           native and does not exhibit that failure mode, so
                           the aggressive CJK/diacritic stripping is removed.
                           Trade-off: if the LLM unexpectedly emits a non-
                           Korean CJK character, it will now pass through
                           unfiltered -- acceptable given EXAONE's design.
"""

import logging
import re
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unicode allow-lists (compiled once at import time for speed)
# ---------------------------------------------------------------------------

# -- PDF content: permissive -- keeps Latin, Greek, math ----------------------
#
# Allowed blocks:
#   U+0009, U+000A, U+000D        tab / LF / CR
#   U+0020-U+007E                 ASCII printable
#   U+00A0-U+00FF  Latin-1 Supplement (accented chars, common in names)
#   U+0100-U+024F  Latin Extended-A/B (European author names)
#   U+0250-U+02AF  IPA Extensions (rare, but harmless)
#   U+0300-U+036F  Combining Diacritical Marks -- kept here because they
#                  legitimately appear on Latin bases; we strip them later
#                  only when they land on non-Latin bases.
#   U+0370-U+03FF  Greek and Coptic (alpha beta gamma ... common in math)
#   U+2000-U+206F  General Punctuation (em-dash, ellipsis, smart quotes ...)
#   U+2070-U+209F  Superscripts and Subscripts
#   U+2100-U+214F  Letterlike Symbols (R N Z ...)
#   U+2190-U+22FF  Arrows + Mathematical Operators
#   U+25A0-U+25FF  Geometric Shapes (bullet, arrow ... used in bullet lists)
#   U+1100-U+11FF  Korean Jamo
#   U+3130-U+318F  Korean Compatibility Jamo
#   U+AC00-U+D7A3  Korean Syllables
#
# Anything else (Vietnamese tone stacks on non-Latin, Private Use Area,
# CJK Unified Ideographs, Devanagari, Thai, ...) is stripped.

_PDF_ALLOWED = re.compile(
    r"[^\t\n\r"
    r" -~"
    r"\xa0-ÿ"
    r"Ā-ɏ"
    r"ɐ-ʯ"
    r"̀-ͯ"
    r"Ͱ-Ͽ"
    r" -⁯"
    r"⁰-₟"
    r"℀-⅏"
    r"←-⋿"
    r"■-◿"
    r"ᄀ-ᇿ"
    r"㄰-㆏"
    r"가-힣"
    r"]",
    re.UNICODE,
)

# -- LLM output: light denylist -- strip only genuine encoding noise ----------
#
# Stripped characters:
#   U+0000-U+0008, U+000B-U+000C, U+000E-U+001F   C0 control (keep \t \n \r)
#   U+007F                                          DEL
#   U+0080-U+009F                                   C1 control
#   U+E000-U+F8FF                                   BMP Private Use Area
#   U+FFFD-U+FFFF                                   replacement char + non-chars
#
# Everything else -- CJK, Latin Extended, diacritics -- passes through.
# The old strict Korean+ASCII allowlist was a qwen2.5 workaround; EXAONE 3.5
# is Korean-native and does not produce Chinese/Vietnamese output noise.

_LLM_NOISE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f"
    r"\x7f"
    r"\x80-\x9f"
    r"-"
    r"�-￿"
    r"]",
    re.UNICODE,
)

# Orphaned combining diacritical mark pattern:
# a combining mark (U+0300-U+036F) that follows a non-Latin base is noise.
# We detect runs of combining marks preceded by a space or Korean syllable.
_ORPHAN_COMBINING = re.compile(r"(?<=[ 가-힣])[̀-ͯ]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_pdf_text(text: str) -> str:
    """
    Remove Unicode noise from raw PDF-extracted text.

    Strips characters outside the allowed PDF block list, removes
    orphaned combining diacritical marks (the signature of Vietnamese
    tone characters appearing on wrong Unicode bases), and collapses
    the resulting whitespace.

    Args:
        text: Raw string from PyMuPDF page.get_text().

    Returns:
        Cleaned string with noise characters replaced by spaces.
    """
    # 1. Strip orphaned combining marks (Vietnamese tone noise)
    text = _ORPHAN_COMBINING.sub("", text)

    # 2. Replace disallowed characters with a single space
    text = _PDF_ALLOWED.sub(" ", text)

    # 3. Normalise whitespace: collapse runs, preserve paragraph breaks
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_llm_output(text: str) -> str:
    """
    Strip genuine encoding noise from LLM response strings.

    Removes C0/C1 control characters, Private Use Area glyphs, and Unicode
    non-characters.  All other content (CJK, diacritics, extended Latin)
    passes through unchanged -- EXAONE 3.5 is Korean-native and does not
    produce the Chinese/Vietnamese output noise that required the old strict
    Korean+ASCII allowlist.

    Args:
        text: Raw string from the Ollama /api/chat response field.

    Returns:
        Cleaned string with encoding noise removed.
    """
    # Remove encoding noise (C0/C1 control, PUA glyphs, non-characters)
    text = _LLM_NOISE.sub("", text)

    # Collapse multiple spaces (but preserve intentional newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_text(pdf_path: str) -> str:
    """
    Extract and clean all text from a PDF file.

    Runs :func:`clean_pdf_text` on every page before concatenating so
    downstream chunking only ever receives well-formed Unicode.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Full paper text as a single UTF-8 string.
        Returns empty string if extraction fails.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            raw = page.get_text("text")
            cleaned = clean_pdf_text(raw)
            if cleaned:
                pages.append(cleaned)
        doc.close()
        full_text = "\n\n".join(pages)
        logger.info(
            "Extracted %d pages / %d chars from %s",
            len(pages),
            len(full_text),
            pdf_path,
        )
        return full_text
    except Exception as exc:
        logger.error("PDF extraction failed for %s: %s", pdf_path, exc)
        return ""
