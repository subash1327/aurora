"""Prompt safety filter for Aurora.

Aurora allows adult (NSFW) content between consenting adults. It does NOT allow:

* any sexual or sexualised content involving minors (or characters presented as minors), and
* sexual / non-consensual content depicting real, identifiable people.

This module enforces the first rule mechanically: any prompt that mentions minors,
children or child-coded terms is refused before it ever reaches a model. The second
rule cannot be detected reliably from text alone and is enforced by policy (see README).

The filter is intentionally conservative: false positives (e.g. "kid gloves") are an
acceptable price. Rephrase the prompt if you hit one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Single words / short phrases that are refused anywhere in a prompt (word-boundary match,
# case-insensitive, simple plural forms included automatically).
BLOCKED_TERMS: tuple[str, ...] = (
    # generic
    "child",
    "children",
    "childlike",
    "child-like",
    "kid",
    "kiddie",
    "kiddy",
    "minor",
    "minors",
    "underage",
    "under age",
    "under-age",
    "underaged",
    "jailbait",
    "juvenile",
    "youngster",
    "infant",
    "toddler",
    "baby",
    "babies",
    "newborn",
    "preschool",
    "preschooler",
    "kindergarten",
    "nursery",
    # ages
    "teen",
    "teens",
    "teenage",
    "teenager",
    "teenaged",
    "preteen",
    "pre-teen",
    "tween",
    "tweens",
    "adolescent",
    "adolescence",
    "pubescent",
    "prepubescent",
    "pre-pubescent",
    "puberty",
    "barely legal",
    # school
    "schoolgirl",
    "schoolboy",
    "school girl",
    "school boy",
    "schoolkid",
    "school kid",
    "elementary school",
    "primary school",
    "middle school",
    "junior high",
    "grade school",
    "high schooler",
    "highschooler",
    "high school student",
    "jk",
    "joshi kosei",
    # descriptive
    "little girl",
    "little boy",
    "young girl",
    "young boy",
    "small girl",
    "small boy",
    "young-looking",
    "childish body",
    "flat chested child",
    "girl child",
    "boy child",
    # subculture terms
    "loli",
    "lolis",
    "lolicon",
    "lolita",
    "shota",
    "shotas",
    "shotacon",
    "toddlercon",
    "cub",
    "cunny",
    "pedo",
    "pedophile",
    "paedophile",
    "pedophilia",
    "paedophilia",
    # family-coded minors
    "son",
    "daughter",
    "stepdaughter",
    "stepson",
    "niece",
    "nephew",
    # non-English (common)
    "niña",
    "niño",
    "enfant",
    "enfants",
    "menor",
    "menores",
    "mädchen",
    "shoujo",
    "shōjo",
    "shounen",
    "shōnen",
    "chuugakusei",
    "shougakusei",
)

# Regexes for explicit ages under 18: "12 yo", "15-year-old", "age 16", "aged 9", "16yo".
_AGE_PATTERNS: tuple[str, ...] = (
    r"\b(?:[0-9]|1[0-7])\s*(?:yo|y/o|y\.o\.?|yrs?|years?|year|yr)[\s-]*(?:old)?\b",
    r"\b(?:[0-9]|1[0-7])[\s-]*years?[\s-]*old\b",
    r"\baged?\s*(?:[0-9]|1[0-7])\b",
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen)[\s-]*(?:years?|yrs?)[\s-]*old\b",
    r"\b(?:1st|2nd|3rd|[4-9]th|1[0-2]th)\s*grader?\b",
    r"\b(?:fresh(?:man|men)|sophomores?)\b",
)

# Harmless phrases that contain a blocked word; stripped before matching.
ALLOWED_PHRASES: tuple[str, ...] = (
    "baby blue",
    "baby pink",
    "baby oil",
    "baby doll",
    "babydoll",
    "baby's breath",
    "kid gloves",
    "kid leather",
    "lion cub",
    "bear cub",
    "wolf cub",
    "fox cub",
    "tiger cub",
    "minor key",
    "minor chord",
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # collapse separators people use to dodge filters: l.o.l.i, l_o_l_i, l-o-l-i
    text = re.sub(r"(?<=\b\w)[._*\-](?=\w\b)", "", text)
    text = re.sub(r"[_*]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _term_regex(term: str) -> str:
    escaped = re.escape(term).replace(r"\ ", r"[\s_\-]*").replace(r"\-", r"[\s_\-]*")
    return rf"(?<![\w]){escaped}(?:s|es)?(?![\w])"


_TERM_RE = re.compile("|".join(_term_regex(t) for t in sorted(set(BLOCKED_TERMS), key=len, reverse=True)))
_AGE_RE = re.compile("|".join(_AGE_PATTERNS))


@dataclass
class SafetyResult:
    allowed: bool
    matches: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # truthy == allowed
        return self.allowed


class PromptRefused(ValueError):
    """Raised when a prompt violates Aurora's content policy."""


def check_prompt(prompt: str | None) -> SafetyResult:
    """Return a :class:`SafetyResult` for ``prompt``. Empty prompts are allowed."""
    if not prompt:
        return SafetyResult(True)
    base = _normalise(prompt)
    for phrase in ALLOWED_PHRASES:
        base = base.replace(phrase, " ")
    variants = {base, base.translate(_LEET)}
    matches: set[str] = set()
    for text in variants:
        matches.update(m.group(0).strip() for m in _TERM_RE.finditer(text))
        matches.update(m.group(0).strip() for m in _AGE_RE.finditer(text))
    return SafetyResult(not matches, sorted(matches))


def assert_safe(*prompts: str | None) -> None:
    """Raise :class:`PromptRefused` if any of ``prompts`` is not allowed.

    Only pass *positive* prompts here; negative prompts are allowed to name blocked
    concepts (e.g. ``negative_prompt="child, teen"`` is a good idea for adult content).
    """
    found: list[str] = []
    for p in prompts:
        found.extend(check_prompt(p).matches)
    if found:
        raise PromptRefused(
            "Prompt refused by Aurora's content policy (minor/child-related terms found: "
            + ", ".join(sorted(set(found)))
            + "). Sexual or suggestive content involving minors is never allowed. "
            "If this is a false positive, rephrase the prompt."
        )


# Recommended additions to the negative prompt whenever adult content is generated.
ADULT_NEGATIVE_PROMPT = "child, kid, teen, underage, loli, shota, young-looking, childlike, minor"


__all__ = [
    "ADULT_NEGATIVE_PROMPT",
    "ALLOWED_PHRASES",
    "BLOCKED_TERMS",
    "PromptRefused",
    "SafetyResult",
    "assert_safe",
    "check_prompt",
]
