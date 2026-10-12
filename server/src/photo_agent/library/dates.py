"""Reading dates out of everyday words: "last summer", "in July", "2023", "yesterday".

Seasons are the northern hemisphere's meteorological ones (summer is June to August). "Last
summer" is the most recent summer that is over; "this summer" is this year's. A plain
"summer" means the most recent one that has begun.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta

Span = tuple[date, date]

MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
SEASONS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}
"""Each season's first month; every season is three months long."""


def _month_span(year: int, month: int, months: int = 1) -> Span:
    end_month = month + months - 1
    end_year = year + (end_month - 1) // 12
    end_month = (end_month - 1) % 12 + 1
    return date(year, month, 1), date(end_year, end_month, monthrange(end_year, end_month)[1])


def _season(year: int, name: str) -> Span:
    return _month_span(year, SEASONS[name], 3)


def _recent(spans: list[Span], today: date, which: str) -> Span:
    """Of spans in time order: "last" is the newest that is over, "this" the one this year
    (or under way), and "" the newest that has begun."""
    if which == "last":
        return [s for s in spans if s[1] < today][-1]
    if which == "this":
        current = [s for s in spans if s[0] <= today <= s[1]]
        if current:
            return current[0]
        return next(s for s in spans if s[0].year == today.year)
    return [s for s in spans if s[0] <= today][-1]


DATE_PATTERN = re.compile(
    r"\b(?:(?P<which>last|this|past)\s+)?"
    r"(?P<unit>today|yesterday|week|month|year|"
    + "|".join(sorted([*SEASONS, *MONTHS], key=len, reverse=True))
    + r")(?:\s+(?:of\s+)?(?P<year>(?:19|20)\d\d))?\b"
    r"|\b(?:in\s+)?(?P<bare_year>(?:19|20)\d\d)\b",
    re.IGNORECASE,
)


def find_dates(text: str, today: date) -> tuple[Span | None, str]:
    """The date span a piece of text mentions (the first one), and the text without it."""
    for match in DATE_PATTERN.finditer(text):
        span = _span(match, today)
        if span is not None:
            rest = text[: match.start()] + " " + text[match.end() :]
            return span, " ".join(rest.split())
    return None, text


def _span(match: re.Match[str], today: date) -> Span | None:
    if match.group("bare_year"):
        whole = int(match.group("bare_year"))
        return date(whole, 1, 1), date(whole, 12, 31)
    unit = match.group("unit").lower()
    which = (match.group("which") or "").lower().replace("past", "last")
    year = int(match.group("year")) if match.group("year") else None
    if unit == "may" and not which and year is None:
        return None  # "may" is far more often the verb
    if unit == "today":
        return today, today
    if unit == "yesterday":
        day = today - timedelta(days=1)
        return day, day
    if unit == "week":
        if not which:
            return None
        monday = today - timedelta(days=today.weekday())
        if which == "last":
            monday -= timedelta(days=7)
        return monday, monday + timedelta(days=6)
    if unit == "month":
        if not which:
            return None
        if which == "this":
            return _month_span(today.year, today.month)
        first = today.replace(day=1) - timedelta(days=1)
        return _month_span(first.year, first.month)
    if unit == "year":
        if not which:
            return None
        y = today.year - (1 if which == "last" else 0)
        return date(y, 1, 1), date(y, 12, 31)
    years = [year] if year is not None else list(range(today.year - 3, today.year + 1))
    if unit in SEASONS:
        spans = [_season(y, unit) for y in years]
    else:
        spans = [_month_span(y, MONTHS[unit]) for y in years]
    if year is not None:
        return spans[0]
    try:
        return _recent(spans, today, which)
    except (IndexError, StopIteration):
        return None
