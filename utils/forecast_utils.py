import re
from typing import Pattern

# Strong signals (any hit => forecast-style)
_FORECAST_PATTERNS: list[Pattern[str]] = [
    # Explicit horizon
    re.compile(r"\bnext\s+\d+\s*(?:years?|yrs?|months?|weeks?)\b", re.I),
    re.compile(r"\bover\s+the\s+next\s+\d+\s*(?:years?|yrs?|months?|weeks?)\b", re.I),
    re.compile(r"\b(in|within)\s+\d+\s*(?:years?|yrs?|months?|weeks?)\b", re.I),
    re.compile(r"\b\d+\s*(?:–|-|to)\s*\d+\s*years?\b", re.I),  # 5-10 years, 5–10 years, 5 to 10 years
    re.compile(r"\bnext\s+decade\b", re.I),
    re.compile(r"\bcoming\s+(?:years?|decade)\b", re.I),

    # By-year phrasing (by 2030)
    re.compile(r"\bby\s+(19|20)\d{2}\b", re.I),

    # Forecast language
    re.compile(r"\bforecast\b", re.I),
    re.compile(r"\bprojection(s)?\b", re.I),
    re.compile(r"\btrend(s)?\b", re.I),
    re.compile(r"\btrajectory\b", re.I),
    re.compile(r"\bexpected\s+to\b", re.I),
    re.compile(r"\blikely\s+to\b", re.I),
    re.compile(r"\bwill\s+(increase|decrease|rise|fall|grow|decline)\b", re.I),
]

# Things that use "next" but aren't forecasts
_NON_FORECAST_PATTERNS: list[Pattern[str]] = [
    re.compile(r"\bnext\s+step\b", re.I),
    re.compile(r"\bnext\s+page\b", re.I),
    re.compile(r"\bnext\s+question\b", re.I),
    re.compile(r"\bwhat\s+should\s+i\s+do\s+next\b", re.I),
]

def is_forecast_question(query: str) -> bool:
    """
    Deterministic gate: returns True if the query expresses future time horizon
    or forecasting/projection intent.
    """
    if not query:
        return False

    q = " ".join(query.strip().split())  # normalize whitespace

    # If it's clearly a navigation "next ..." question, don't route to analytics mode.
    for pat in _NON_FORECAST_PATTERNS:
        if pat.search(q):
            return False

    for pat in _FORECAST_PATTERNS:
        if pat.search(q):
            return True

    return False
