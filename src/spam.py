"""Heuristic spam filtering for chat messages.

Targets the spam that floods these chats:
- "quick money" / earnings schemes (быстрый заработок, доход в день, "пиши в ЛС")
- veiled drug ads from darkstores (клад/закладка, меф, шишки, 24/7, "проверенный магазин")
- 18+ / escort ads (18+, интим, досуг, вебкам)

These are heuristics — tune the patterns for your chats. `spam_reason(text)`
returns the matched category (or None); `is_spam(text)` is a boolean wrapper.

NOTE: chat text is untrusted user content. We only pattern-match it, never
execute or follow instructions found inside it.
"""
from __future__ import annotations

import re

# (category, regex). Patterns are deliberately specific to limit false positives;
# add/relax them based on what you see in `notebooks/explore.ipynb`.
SPAM_PATTERNS: list[tuple[str, str]] = [
    # --- quick money / earnings ---
    ("money", r"л[её]гк\w*\s+деньг"),
    ("money", r"пасс\w*\s+доход"),
    ("money", r"доход\w*\s+от\s*\d"),
    ("money", r"\d[\d\s]*\$?\s*(в\s+день|в\s+неделю|в\s+час|в\s+сутки)"),
    ("money", r"подзаработ"),
    ("money", r"(пиш\w+|жду|интересно)\W{0,15}(в\s*)?(л\.?с\.?|личк|direct|дайр)"),
    ("money", r"набираю\s+(команду|людей|сотрудник)"),
    ("money", r"(удал[её]нн\w*|онлайн)\s+работ\w*\W{0,25}(\d|\$|доход|от\s*\d)"),
    ("money", r"крипт\w*\W{0,20}(доход|заработ|профит|x\d)"),

    # --- veiled drugs / darkstores ---
    ("drugs", r"\bзакладк|кладмен|\bклад\b"),
    ("drugs", r"мефедрон|\bмеф\b|альфа.?пвп|\bа.?пвп\b"),
    ("drugs", r"амфетамин|\bамф\b|метамфетамин"),
    ("drugs", r"гашиш|\bгаш\b|шишк|\bбошк|марихуан|\bтрав[аку]\b|гидропон"),
    ("drugs", r"кокаин|\bкок\b|мдма|\bмд\b|экстази|\bлсд\b|\bскорост[ьи]\b"),
    ("drugs", r"(проверенн\w*|надёжн\w*|лучш\w*)\s+магазин\W{0,15}(24|круглосут|тгк|шоп)"),
    ("drugs", r"автопрода\w|автошоп|наркошоп|\bтгк\b"),
    ("drugs", r"24/7\W{0,20}(магазин|шоп|достав|клад)"),

    # --- 18+ / escort ---
    ("adult", r"18\s*\+"),
    ("adult", r"интим\w*\s+(услуг|досуг|встреч)"),
    ("adult", r"\bдосуг\b|эскорт|индивидуалк|проститут"),
    ("adult", r"вебк[ау]м|webcam|онлайн.?анкет"),
    ("adult", r"девочк\w*\W{0,15}(досуг|ночь|интим|18|анкет)"),
    ("adult", r"масс?аж\w*\W{0,12}(интим|релакс|для\s+мужчин)"),
]

_COMPILED: list[tuple[str, re.Pattern]] = [
    (cat, re.compile(p, re.IGNORECASE)) for cat, p in SPAM_PATTERNS
]


def spam_reason(text: str) -> str | None:
    """Return the spam category that matches, or None if the text looks clean."""
    if not text:
        return None
    for category, rx in _COMPILED:
        if rx.search(text):
            return category
    return None


def is_spam(text: str) -> bool:
    return spam_reason(text) is not None


def filter_spam(msgs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split messages into (clean, removed). Removed messages get a 'spam'
    field with the matched category, handy for inspection."""
    clean, removed = [], []
    for m in msgs:
        reason = spam_reason(m.get("text", ""))
        if reason:
            removed.append({**m, "spam": reason})
        else:
            clean.append(m)
    return clean, removed
