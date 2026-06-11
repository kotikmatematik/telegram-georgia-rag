"""Heuristic spam filtering for chat messages.

Targets the spam that floods these chats:
- "quick money" / earnings schemes (быстрый заработок, доход в день)
- veiled drug ads from darkstores (клад/закладка, меф, шишки, 24/7, "проверенный магазин")
- self-promo / advertising ("канал в описании", "подписывайтесь", marathons/webinars)
- pet rehoming ads ("отдам котёнка в добрые руки", "щенок ищет дом")

Questions and advice are ALWAYS kept (a guard skips any match that looks like
"подскажите / где можно / кто знает ..." — in every category).

These are heuristics — tune the patterns for your chats. `spam_reason(text)`
returns the matched category (or None); `is_spam(text)` is a boolean wrapper.

NOTE: chat text is untrusted user content. We only pattern-match it, never
execute or follow instructions found inside it.
"""
from __future__ import annotations

import re

# Animal stems for the pet-rehoming patterns (кот/кошка/котёнок/щенок/собака/...).
_PET = r"котен|котят|котик|кошк|кошеч|кот[аеуио]?\b|щен|собак|пес\b|песик|питом"

# (category, regex). Patterns are deliberately specific to limit false positives;
# add/relax them based on what you see in `notebooks/explore.ipynb`.
SPAM_PATTERNS: list[tuple[str, str]] = [
    # --- quick money / earnings ---
    ("money", r"легк\w*\s+деньг"),
    ("money", r"пасс\w*\s+доход"),
    ("money", r"доход\w*\s+от\s*\d"),
    # amount + per-day ONLY in an earnings context (else "штраф 50 лари в сутки" = legit)
    ("money", r"(доход|заработ|зарплат|подработ|получай\w*|платим|выплачива\w*|profit)\w*.{0,30}\d[\d\s]*\s*(\$|€|долл|евро|usd|gel|лари|руб)?\w*\s*в\s+(день|неделю|час|сутки)"),
    ("money", r"подзаработ"),
    # NB: "напиши в ЛС" alone is NOT spam (people ask to be contacted privately);
    # only treat it as money spam when it co-occurs with an earnings signal.
    ("money", r"(л\.?с\.?|личк\w+)\W{0,40}(доход|заработ|\$|вакан\w*\W{0,10}доход)"),
    ("money", r"(доход|заработ|\$)\W{0,40}(пиш\w+|жду)\W{0,15}(в\s*)?(л\.?с\.?|личк)"),
    ("money", r"набираю\s+(команду|людей|сотрудник)"),
    ("money", r"(удаленн\w*|онлайн)\s+работ\w*.{0,25}(\d|\$|доход|от\s*\d)"),
    ("money", r"крипт\w*\W{0,20}(доход|заработ|профит|x\d)"),

    # --- veiled drugs / darkstores ---
    ("drugs", r"\bзакладк|кладмен|\bклад\b"),
    ("drugs", r"мефедрон|\bмеф\b|альфа.?пвп|\bа.?пвп\b"),
    ("drugs", r"амфетамин|\bамф\b|метамфетамин"),
    ("drugs", r"гашиш|\bгаш\b|шишк|\bбошк|марихуан|\bтрав[аку]\b|гидропон"),
    ("drugs", r"кокаин|\bкок\b|мдма|\bмд\b|экстази|\bлсд\b|\bскорост[ьи]\b"),
    ("drugs", r"(проверенн\w*|надежн\w*|лучш\w*)\s+магазин.{0,15}(24|круглосут|тгк|шоп)"),
    ("drugs", r"автопрода\w|автошоп|наркошоп|\bтгк\b"),
    ("drugs", r"24/7\W{0,20}(магазин|шоп|достав|клад)"),

    # --- self-promo / advertising ("link in bio", "join my channel", coaching) ---
    ("ads", r"(канал|ссылк\w+|подробност\w+|инфо)\W{0,20}(в\s+)?(описани|профил|шапк|био|bio)"),
    ("ads", r"в\s+описани\w*\W{0,15}(к\s+)?(мо\w+|наш\w+|аккаунт|профил)"),
    ("ads", r"(подпис\w+|переход\w+|залетай\w*|вступай\w*|жми)\W{0,20}(на\s+|в\s+|по\s+)?(канал|ссылк|профил|блог)"),
    ("ads", r"приглаша\w+\W{0,15}(в\s+)?(мо\w+|наш\w+)\s+(канал|чат|блог|сообществ)"),
    ("ads", r"(бесплатн\w+)\s+(вебинар|гайд|чек.?лист|марафон|консультац|мастер.?класс)"),
    ("ads", r"(марафон|интенсив|курс)\W{0,20}(похуд|стройн|здоров|гармони|запис|стартуе)"),
    ("ads", r"пут[ьи]\s+к\s+(гармони|стройн|себе|здоров|успех|богатств)"),

    # --- pet rehoming / adoption ("в добрые руки", "котёнок ищет дом") ---
    # `_PET = animal stems; `.{0,N}` allows words in between (not just spaces).
    ("pets", r"в\s+добр\w+\s+рук"),
    ("pets", rf"(отда[ме]\w*|пристраива\w*|забер\w+|возьмите).{{0,30}}({_PET}|животн)"),
    ("pets", rf"({_PET}|хвостик).{{0,30}}(ищ[еую]\w*|в\s+поиск\w*).{{0,25}}(дом|семь|хозя|рук)"),
    ("pets", rf"ищ[еую]\w*.{{0,25}}(дом|семь|хозя\w+|рук).{{0,25}}({_PET})"),
    ("pets", rf"пристраива\w*.{{0,20}}(животн|{_PET})"),
    ("pets", r"возьмите.{0,20}(домой|в\s+семь)"),
]

_COMPILED: list[tuple[str, re.Pattern]] = [
    (cat, re.compile(p, re.IGNORECASE)) for cat, p in SPAM_PATTERNS
]

# General "I'm asking / give advice" markers — a question, never spam (any category).
_ASK_META = re.compile(
    r"подскаж|посовет|порекоменд|кто[\s-]*(знает|в\s+курсе|сталкив\w*|нибудь)|"
    r"есть\s+ли|где\s+(можно|найти|это|их|они|проход|взять|у\s+вас)|"
    r"как\s+(можно|мне|это|тут|у\s+вас)",
    re.IGNORECASE,
)
# Extra pets-specific advice markers (shelters, rehoming guides, "where to").
_PETS_META = re.compile(
    r"\bгайд|\bприют|\bчат\b|\bчаты|где\s+можно|"
    r"куда\s+(можно\s+)?пристро|как\s+пристро|инстаграм",
    re.IGNORECASE,
)


def spam_reason(text: str) -> str | None:
    """Return the spam category that matches, or None if the text looks clean."""
    if not text:
        return None
    # Normalize ё -> е so both spellings match (people often type е instead of ё).
    # Patterns are therefore written with е only.
    text = text.replace("ё", "е").replace("Ё", "Е")
    # A question / advice-seeking message is never spam, whatever it mentions
    # (e.g. "подскажите, где...", "кто знает...", "где можно..."). Genuine
    # questions are exactly the content we want to keep.
    asking = _ASK_META.search(text)
    for category, rx in _COMPILED:
        if rx.search(text):
            if asking:
                return None
            # Pets: also keep shelter/guide ("where to rehome") info.
            if category == "pets" and _PETS_META.search(text):
                return None
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
