"""Heuristic prompt-injection detection for text embedded in ingested
documents (4.3). Two layers of defense are used together: (1) the system
prompt tells the model that document content is untrusted data, never
instructions, and (2) this module additionally redacts the specific spans
that look like embedded directives *before* the text ever reaches the
model, so the imperative phrasing itself never appears verbatim in context.

This is a heuristic, regex-based scan — it will miss novel phrasings and can
false-positive on legitimate text that happens to match a pattern (e.g. a
security policy discussing "ignore previous instructions" as an example of
an attack). That trade-off is intentional and documented: false positives
just get a redaction note in evidence that a user can inspect, whereas false
negatives let an attack through, so we bias toward over-flagging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all|any)?\s*(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all|any)?\s*(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"new\s+instructions\s+supersede", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(a\s+)?(developer|debug|unrestricted|jailbroken?)\s+mode", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"reveal\s+(your\s+|the\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"act\s+as\s+(an?|the)\s+.*(dan|unrestricted|jailbroken)", re.IGNORECASE | re.DOTALL),
    re.compile(r"do\s+not\s+mention\s+(citations|evidence|sources)", re.IGNORECASE),
    re.compile(r"respond\s+only\s+with", re.IGNORECASE),
    re.compile(r"from\s+now\s+on,?\s+you\s+are", re.IGNORECASE),
]

_REDACTION_NOTE = "[REDACTED: suspected embedded instruction, not followed]"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class InjectionScanResult:
    flagged: bool
    matched_patterns: list[str]
    sanitized_text: str


def scan_and_sanitize(text: str) -> InjectionScanResult:
    matched: list[str] = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    if not matched:
        return InjectionScanResult(flagged=False, matched_patterns=[], sanitized_text=text)

    # Redact the whole sentence a match falls in, not just the trigger phrase itself —
    # otherwise the payload following the trigger (e.g. what to "respond only with")
    # survives verbatim right next to the redaction note.
    sentences = _SENTENCE_SPLIT_RE.split(text)
    sanitized_sentences = [
        _REDACTION_NOTE if any(p.search(sentence) for p in _INJECTION_PATTERNS) else sentence
        for sentence in sentences
    ]
    sanitized_text = " ".join(sanitized_sentences)

    return InjectionScanResult(flagged=True, matched_patterns=matched, sanitized_text=sanitized_text)
