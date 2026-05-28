"""Country-specific plate cleaning, regex validation, and char fixes.

Each `CountryProfile` is a self-contained recipe for post-processing OCR output:

    raw OCR  →  clean()  →  fix_and_validate()  →  (text, valid)

`clean()` strips non-alphanumeric + country watermarks.
`fix_and_validate()` tries the country regex(es); if it fails, attempts
one-character substitutions from the country's confusable-char table
(O↔0, I↔1, etc.) and re-tries; optionally checks a state/region prefix list.

To add a country: append a `CountryProfile` to `COUNTRIES`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CountryProfile:
    code: str                           # ISO-ish code used in the UI
    name: str                           # display name
    regexes: list[re.Pattern]           # plate format patterns
    watermarks: list[str]               # strings to strip (e.g. "IND", "GB", "EU")
    char_fixes: dict[str, str]          # confusable-char one-step substitutions
    valid_prefixes: Optional[set[str]] = None   # state/province first-letters set

    def clean(self, raw: str) -> str:
        """Uppercase, alphanumeric-only, strip country watermarks at edges
        and any mid-string watermark that follows a plausible plate prefix.
        """
        s = "".join(ch for ch in (raw or "").upper() if ch.isalnum())
        for w in self.watermarks:
            n = len(w)
            # leading
            if s.startswith(w) and len(s) >= n + 8:
                s = s[n:]
            # trailing
            if s.endswith(w) and len(s) >= n + 8:
                s = s[:-n]
            # mid-string (after position 6 so we don't chew the front)
            idx = s.find(w, 6)
            if idx > 0:
                s = s[:idx]
        return s

    def regex_valid(self, text: str) -> bool:
        return any(rx.match(text) for rx in self.regexes)

    def prefix_valid(self, text: str) -> bool:
        if not self.valid_prefixes:
            return True
        return any(text.startswith(p) for p in self.valid_prefixes)

    def fix_and_validate(self, text: str) -> tuple[str, bool]:
        """Return (best_text, valid).  Tries up to one single-character
        substitution from `char_fixes` to coerce a near-miss into a valid plate.
        """
        if self.regex_valid(text) and self.prefix_valid(text):
            return text, True
        for i, ch in enumerate(text):
            if ch in self.char_fixes:
                alt = text[:i] + self.char_fixes[ch] + text[i + 1:]
                if self.regex_valid(alt) and self.prefix_valid(alt):
                    return alt, True
        return text, False


# ---------------------------------------------------------------- country data

# India: 36 state / UT prefixes (2-letter)
IN_STATES: set[str] = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UA",
    "UK", "UP", "WB",
}


COUNTRIES: dict[str, CountryProfile] = {
    "IN": CountryProfile(
        code="IN", name="India",
        regexes=[
            re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),   # standard
            re.compile(r"^\d{2}BH\d{4}[A-Z]$"),                # BH-series
        ],
        watermarks=["IND"],
        char_fixes={"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"},
        valid_prefixes=IN_STATES,
    ),

    "UK": CountryProfile(
        code="UK", name="United Kingdom",
        regexes=[
            # Current format (2001-present): YY70 ABC
            re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$"),
            # 1983-2001 prefix style: A123BCD
            re.compile(r"^[A-Z]\d{1,3}[A-Z]{3}$"),
            # 1963-1983 suffix style: ABC123D
            re.compile(r"^[A-Z]{3}\d{1,3}[A-Z]$"),
        ],
        watermarks=["GB", "UK"],
        # On UK plates "O" and "I" are NEVER used (avoid 0/1 confusion) — so
        # if OCR returns O/I they're almost always misread digits.  Strong fixes.
        char_fixes={"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"},
    ),

    "US": CountryProfile(
        code="US", name="United States",
        regexes=[
            # US plates vary wildly by state — use a permissive 3–8 alphanumeric.
            re.compile(r"^[A-Z0-9]{3,8}$"),
        ],
        watermarks=[],
        char_fixes={},   # too varied per state to apply universally
    ),

    "EU": CountryProfile(
        code="EU", name="Europe (generic)",
        regexes=[
            # 4–11 alphanumeric (covers DE 6-9, FR 7-9, IT 7, ES 7, NL 6-8, etc.)
            re.compile(r"^[A-Z0-9]{4,11}$"),
        ],
        # Only strip the literal "EU" string — single-letter country codes (D, F,
        # I, etc.) would corrupt plate text if blindly removed.
        watermarks=["EU"],
        char_fixes={"O": "0", "I": "1"},
    ),

    "XX": CountryProfile(
        code="XX", name="Generic / any",
        regexes=[
            re.compile(r"^[A-Z0-9]{4,12}$"),
        ],
        watermarks=[],
        char_fixes={},
    ),
}


def get_profile(code: str) -> CountryProfile:
    """Lookup by code, falling back to the generic profile."""
    return COUNTRIES.get((code or "XX").upper(), COUNTRIES["XX"])
