import re
import math
from typing import Dict, List, Optional, Tuple, Any, Set, Optional
from utils.text_utils import TOKEN_NORMALIZATION

NUM_RE = re.compile(r"\b\d{1,3}\b")

def extract_numbers(text: str) -> Set[int]:
    return {int(x) for x in NUM_RE.findall(text)}

# keep STOP small; don't include "median", "ratio", etc.
STOP = {
    "what","is","the","of","in","for","a","an","and","or","on","at","by",
    "estimate","estimated","how","many","much","people","population",
    "city","county","state"
}

def tokenize(text: str):
    s = text.lower()

    # normalize synonyms early
    for k, v in TOKEN_NORMALIZATION.items():
        s = re.sub(rf"\b{k}\b", v, s)

    s = re.sub(r"[^a-z0-9\s\-\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    raw = s.split()

    toks = []
    for t in raw:
        if t in STOP:
            continue

        # keep numbers
        if t.isdigit():
            toks.append(t)
            continue

        # keep important connectors
        if t in {"to", "under", "over"}:
            toks.append(t)
            continue

        if len(t) <= 2:
            continue

        # plural normalization
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]

        # final normalization (safe)
        t = TOKEN_NORMALIZATION.get(t, t)
        toks.append(t)

    return toks


def build_vocab_token_stats(candidates: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Returns:
      df[token] = number of candidates containing token
      idf[token] = log((N+1)/(df+1)) + 1  (smooth, deterministic)
    """
    df: Dict[str, int] = {}
    N = 0
    for c in candidates:
        N += 1
        seen = set(tokenize(c))
        for t in seen:
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1) / (d + 1)) + 1.0
    return df, idf


def best_vocab_match(
    question: str,
    candidates: List[str],
    df: Dict[str, int],
    idf: Dict[str, float],
    max_df_frac: float = 0.35,
) -> Optional[Tuple[str, float]]:
    q_tokens = tokenize(question)

    if not q_tokens:
        return None

    vocab = set(df.keys())
    q_tokens = [t for t in q_tokens if t in vocab]
    if not q_tokens:
        return None

    N_candidates = max(1, len(candidates))

    q_tokens_kept = []
    for t in q_tokens:
        if df.get(t, 0) / N_candidates <= max_df_frac:
            q_tokens_kept.append(t)

    if not q_tokens_kept:
        q_tokens_kept = q_tokens

    q_set = set(q_tokens_kept)
    denom = sum(idf.get(t, 1.0) for t in q_set)
    if denom <= 0:
        return None

    q_nums = extract_numbers(question)

    best_candidate = None
    best_score = 0.0

    for c in candidates:
        c_tokens = [t for t in tokenize(c) if t in vocab]
        c_set = set(c_tokens)

        overlap = q_set & c_set
        if not overlap:
            continue

        numer = sum(idf.get(t, 1.0) for t in overlap)

        c_nums = extract_numbers(c)
        if c_nums and q_nums:
            num_overlap = len(q_nums & c_nums)
            num_precision = num_overlap / len(c_nums)
        elif c_nums and not q_nums:
            num_precision = 0.85
        else:
            num_precision = 1.0

        base_score = (numer / denom) * num_precision

        # reward candidates whose own tokens are well covered by the question
        cand_weight = sum(idf.get(t, 1.0) for t in c_set) or 1.0
        cand_coverage = sum(idf.get(t, 1.0) for t in overlap) / cand_weight

        # modest boost for more complete phrase match
        score = base_score * (1.0 + 0.35 * cand_coverage)

        if score > best_score:
            best_score = score
            best_candidate = c

    if best_candidate is None:
        return None

    return best_candidate, best_score


def best_typed_matches(question: str, meta: dict) -> Dict[str, Optional[Tuple[str, float]]]:
    out = {}

    for field in ("measures", "subjects", "labels", "stat_types"):
        vals = meta.get(field, []) or []
        if not vals:
            out[field] = None
            continue

        df, idf = build_vocab_token_stats(vals)

        if field == "subjects":   # SUBJECTS: use looser filtering
            out[field] = best_vocab_match(
                question,
                vals,
                df,
                idf,
                max_df_frac=0.8,  # <-- critical change
            )

        else:  # default behavior for others
            out[field] = best_vocab_match(
                question,
                vals,
                df,
                idf,
                max_df_frac=0.35,
            )

    return out

def extract_cell_lookup_slots(question: str, meta: dict, threshold: float = 0.55) -> Optional[Dict[str, Any]]:
    hits = best_typed_matches(question, meta)

    measure_hit = hits.get("measures")
    subject_hit = hits.get("subjects")
    label_hit = hits.get("labels")
    stat_hit = hits.get("stat_types")

    measure = measure_hit[0] if measure_hit and measure_hit[1] >= threshold else None
    subject = subject_hit[0] if subject_hit and subject_hit[1] >= threshold else None
    label = label_hit[0] if label_hit and label_hit[1] >= threshold else None
    stat_type = stat_hit[0] if stat_hit and stat_hit[1] >= threshold else None

    if not measure:
        return None

    return {
        "label": label,
        "measure": measure,
        "subject": subject or "Total",
        "stat_type": stat_type or "Estimate",
    }


AGG_HINT = re.compile(
    r"\b(total|sum|average|avg|mean|count|how many|max|maximum|min|minimum)\b",
    re.I,
)

GROUP_HINT = re.compile(
    r"\b(by|per|for each|across|grouped by|over time)\b",
    re.I,
)

CHART_HINT = re.compile(
     r"\b(chart|plot|graph|visualize|show|bar chart|line chart|scatter|histogram|trend|distribution|breakdown|broken down|by race|by disability|by age|by sex)\b",
     re.I,
)

BREAKDOWN_HINT = re.compile(r"\b(break(?:ing)?\s+down|broken\s+down|by)\b", re.I)

RANK_HINT = re.compile(
    r"\b(most|least|highest|lowest|largest|smallest|top|bottom|prominent|common)\b",
    re.I,
)

COMPARE_HINT = re.compile(
    r"\b(compare|comparison|versus|vs\.?|male and female|males and females|both)\b",
    re.I,
)

UNDER_HINT = re.compile(r"\b(under|less than|below|at most)\s+(?P<n>\d{1,3}(?:,\d{3})*|\d+)\b",
    re.I)
NUM_TOKEN = r"(?:\d{1,3}(?:,\d{3})*|\d+)"  # 75,999 or 75999

OVER_HINT = re.compile(
    rf"\b(?P<op>over|above|greater\s+than|at\s+least)\s+"
    rf"(?P<n>{NUM_TOKEN})"
    rf"(?!\s*(?:to|\-)\s*{NUM_TOKEN})"
    rf"(?:\s+years?)?\b",
    re.I,
)
RANGE_HINT = re.compile(
    rf"\b(?:from\s+|between\s+)?"
    rf"(?P<low>{NUM_TOKEN})\s*"
    rf"(?:to|\-)\s*"
    rf"(?P<high>{NUM_TOKEN})"
    rf"(?:\s+years?\s+old|\s+years?)?\b",
    re.I,
)

ROW_FILTER_HINT = re.compile(
    r"\b(most|least|highest|lowest|largest|smallest|top|bottom|prominent|most populated|most common|which one)\b",
    re.I,
)

def direct_measure_first_guard(
    question: str,
    meta: dict,
    threshold: float = 0.55,
) -> Optional[Dict[str, Any]]:
    """
    Only force a direct cell_lookup when the question appears to ask for
    one exact measure value, not an aggregation/comparison/chart/ranking task.
    """
    measures = meta.get("measures", [])
    if not measures:
        return None

    q = question.strip()

    # Block direct cell-lookup override for questions that are clearly not simple lookups
    if (
        AGG_HINT.search(q)
        or GROUP_HINT.search(q)
        or CHART_HINT.search(q)
        or RANK_HINT.search(q)
        or COMPARE_HINT.search(q)
    ):
        return None

    slots = extract_cell_lookup_slots(q, meta, threshold=threshold)
    if not slots:
        return None

    measure = slots["measure"]

    # Prevent "under X" from incorrectly matching unrelated bucket labels
    if UNDER_HINT.search(q):
        q_nums = extract_numbers(q)
        m_nums = extract_numbers(measure)
        if m_nums and q_nums and not m_nums.issubset(q_nums):
            return None

    return {
        "force_category": "cell_lookup",
        "force_query": slots,
        "direct_measure_score": 1.0,
        "reason": "direct_measure_match",
    }


RANGE_RE = re.compile(r"^\s*(\d{1,3})\s*(?:to|\-)\s*(\d{1,3})\s+year", re.I)
UNDER_MEASURE_RE = re.compile(r"^\s*under\s+(\d{1,3})\s+year", re.I)

def parse_age_band(measure: str) -> Optional[Tuple[int, int]]:
    """
    Returns (low, high) inclusive bounds for age bands in measure strings like:
    - "Under 5 years" -> (0, 4)
    - "20 to 24 years" -> (20, 24)
    - "85 years and over" -> (85, 10**9)  (not used for under-X)
    """
    s = (measure or "").lower()
    if "year" not in s:
        return None

    m = UNDER_MEASURE_RE.match(measure)
    if m:
        high = int(m.group(1)) - 1
        return (0, high)

    m = RANGE_RE.match(measure)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    if "and over" in s:
        nums = re.findall(r"\d{1,3}", measure)
        if nums:
            low = int(nums[0])
            return (low, 10**9)

    return None

def _band_overlaps(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])

def _select_disjoint_prefer_finer(bands: List[Tuple[int,int,str]]) -> List[str]:
    """
    bands: (low, high, name)
    Prefer finer (narrower) bands; avoid overlaps and avoid supersets like "Under 19 years".
    Deterministic greedy selection:
      - sort by (width asc, low asc, high asc, name)
      - take band if it doesn't overlap any already selected
    """
    def width(x): 
        low, high, _ = x
        return high - low

    bands_sorted = sorted(bands, key=lambda x: (width(x), x[0], x[1], x[2]))
    chosen: List[Tuple[int,int,str]] = []
    for low, high, name in bands_sorted:
        b = (low, high)
        if any(_band_overlaps(b, (c[0], c[1])) for c in chosen):
            continue
        chosen.append((low, high, name))

    # Output in natural age order
    chosen.sort(key=lambda x: x[0])
    return [name for _, _, name in chosen]


def age_under_measures_in(target: int, measures: List[str]) -> List[str]:
    """
    Select age bands whose upper bound <= target.
    For ACS-style, target=24 should include "20 to 24 years".
    """
    bands: List[Tuple[int,int,str]] = []
    for m in measures:
        b = parse_age_band(m)
        if not b:
            continue
        low, high = b
        if high <= target:
            bands.append((low, high, m))

    # sort by low bound to keep x-axis natural
    if not bands:
        return []

    # Prefer disjoint fine-grained partition; this will drop rolled-up buckets
    # like "Under 19 years" if "Under 6 years" + "6 to 18 years" exist.
    return _select_disjoint_prefer_finer(bands)


def age_over_measures_in(target: int, measures: List[str]) -> List[str]:
    """
    Select age bands whose lower bound >= target.
    Only age-like measures will be parsed by parse_age_band().
    """
    bands = []
    for m in measures:
        b = parse_age_band(m)
        if not b:
            continue
        low, high = b
        if low >= target:
            bands.append((low, high, m))

    bands.sort(key=lambda x: x[0])
    return [m for _, _, m in bands]


def over_age_guard(question: str, meta: dict) -> Optional[Dict[str, Any]]:
    """
    Handle queries like:
      - "over 65"
      - "65+"
      - "65 and over"

    Preference order:
      1) If dataset has a direct rolled-up bucket (e.g., "Over 65 years and older"),
         force cell_lookup to that exact measure.
      2) Else, deterministically sum all age bands whose lower bound >= target.
    """
    measures = meta.get("measures", []) or []
    q = (question or "").strip()
    q_lower = q.lower()

    m = OVER_HINT.search(q)
    if m:
        target = parse_int_token(m.group("n"))
    else:
        # Basic support for "65+" and "65 and over" phrasing
        plus = re.search(r"\b(?P<n>\d{1,3})\s*\+\b", q_lower)
        and_over = re.search(r"\b(?P<n>\d{1,3})\s+(and\s+over|or\s+older)\b", q_lower)
        if plus:
            target = parse_int_token(plus.group("n"))
        elif and_over:
            target = parse_int_token(and_over.group("n"))
        else:
            return None

    # Case 1: direct rolled-up bucket exists -> cell_lookup
    # (You said you have a measure exactly called "Over 65 years and older")
    direct_candidates = [
        f"Over {target} years and older",
        f"{target} years and over",
        f"{target} years and older",
    ]
    for d in direct_candidates:
        if d in measures:
            return {
                "force_category": "cell_lookup",
                "force_measure": d,
                "reason": "direct_over_bucket",
            }

    # Case 2: deterministic sum across age bands
    measures_in = age_over_measures_in(target, measures)
    if not measures_in:
        return None

    return {
        "force_category": "aggregation",
        "force_op": "sum",
        "force_measures_in": measures_in,
        "reason": "over_range_sum",
    }

def measures_overlapping_range(low: int, high: int, measures: List[str]) -> List[str]:
    """
    Select measures whose age-band overlaps the requested inclusive range [low, high].

    Examples:
      low=5, high=44
      - "5 to 14 years" overlaps  -> include
      - "15 to 17 years" overlaps -> include
      - "Under 18 years" overlaps -> include
      - "18 to 24 years" overlaps -> include
      - "15 to 44 years" overlaps -> include
      - "45 to 49 years" no overlap -> exclude
    """
    bands = []
    for m in measures:
        b = parse_age_band(m)
        if not b:
            continue

        m_low, m_high = b
        m_lower = m.lower()

        # reject open-ended categories for closed ranges
        if "and over" in m_lower:
            continue

        # overlap test for inclusive intervals
        if not (m_high < low or m_low > high):
            bands.append((m_low, m_high, m))

    bands.sort(key=lambda x: x[0])
    return [m for _, _, m in bands]


def parse_int_token(s: str) -> Optional[int]:
    if not s:
        return None
    # keep digits only (removes commas, spaces)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

def age_range_sum_guard(
    question: str,
    meta: dict,
    merged: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    merged = merged or {}

    # If an earlier guard already chose a measure group, respect it.
    measure_group = merged.get("force_measure_group")

    if measure_group:
        measures = meta.get("measure_groups", {}).get(measure_group, []) or []
    else:
        measures = meta.get("measures", []) or []

    # 1. explicit closed range wins
    range_m = RANGE_HINT.search(question)
    if range_m:
        low = int(range_m.group("low"))
        high = int(range_m.group("high"))

        # Look for exact measure otherwise look for overlapping ranges
        exact = [m for m in measures if parse_age_band(m) == (low, high)]
        if exact:
            measures_in = exact
        else:
            measures_in = measures_overlapping_range(low, high, measures)

        if not measures_in:
            return None

        # one exact measure -> let cell_lookup handle it
        if len(measures_in) == 1:
            return None

        return {
            "force_category": "aggregation",
            "force_op": "sum",
            "force_measures_in": measures_in,
            "reason": "closed_range_sum",
        }

    under_m = UNDER_HINT.search(question)
    if under_m:
        target = parse_int_token(under_m.group("n"))
        measures_in = age_under_measures_in(target, measures)
        if not measures_in:
            return None

        if len(measures_in) == 1:
            return None

        return {
            "force_category": "aggregation",
            "force_op": "sum",
            "force_measures_in": measures_in,
            "reason": "under_range_sum",
        }

    over_m = OVER_HINT.search(question)
    if over_m:
        target = parse_int_token(over_m.group("n"))
        measures_in = age_over_measures_in(target, measures)
        if not measures_in:
            return None

        if len(measures_in) == 1:
            return None

        return {
            "force_category": "aggregation",
            "force_op": "sum",
            "force_measures_in": measures_in,
            "reason": "over_range_sum",
        }

    return None


def ranking_intent_guard(question: str, meta: dict) -> Optional[Dict[str, Any]]:
    """
    Distinguish row_filter from chart_request only.
    Do not decide measure_group, subject, stat_type, etc.
    """
    q = question.strip()

    # Explicit chart language should stay with LLM / chart path
    if CHART_HINT.search(q):
        return {
            "force_category": "chart_request",
            "reason": "chart_intent"
        }

    # Ranking / "which one is highest" style questions should be row_filter
    if ROW_FILTER_HINT.search(q):
        return {
            "force_category": "row_filter",
            "reason": "ranking_intent"
        }

    return None

BREAKDOWN_PATTERNS = [
    re.compile(r"\bcategoris(?:ed|ed)\s+by\s+(.+)$", re.I),
    re.compile(r"\bbroken\s+down\s+by\s+(.+)$", re.I),
    re.compile(r"\bgroup(?:ed)?\s+by\s+(.+)$", re.I),
]

def extract_breakdown_span(question: str) -> str | None:
    q = question.strip()
    for pat in BREAKDOWN_PATTERNS:
        m = pat.search(q)
        if m:
            span = m.group(1).strip(" .?")
            # keep it short; often "age groups", "region", etc.
            return span
    return None

def measure_group_guard(question: str, meta: dict, threshold: float = 0.75, margin: float = 0.20):
    groups = meta.get("measure_headings", []) or []
    if not groups:
        return None

    breakdown = extract_breakdown_span(question)
    # Use the breakdown phrase if present; it’s what defines the measure group for charts/rankings.
    match_text = breakdown if breakdown else question

    df, idf = build_vocab_token_stats(groups)

    scored = []
    for g in groups:
        hit = best_vocab_match(match_text, [g], df, idf, max_df_frac=0.8)
        if hit:
            scored.append(hit)

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    best_group, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    if best_score < threshold:
        return None
    if (best_score - second_score) < margin:
        return None

    return {
        "force_measure_group": best_group,
        "measure_group_score": best_score,
        "reason": "measure_group_match",
        "debug_match_text": match_text,
    }

NEGATION_HINT = re.compile(r"\b(no|not|without|never|none)\b", re.I)


def _negates_token(question: str, token: str, window_tokens: int = 3) -> bool:
    """
    Returns True if question contains a negation within N tokens before `token`.
    Generic: catches patterns like "not insured", "without internet", "no vehicle".
    """
    q = tokenize(question)
    t = token.lower()
    for i, w in enumerate(q):
        if w != t:
            continue
        start = max(0, i - window_tokens)
        ctx = q[start:i]
        if any(NEGATION_HINT.fullmatch(x) for x in ctx):
            return True
    return False


def direct_subject_first_guard(
    question: str,
    meta: dict,
    threshold: float = 0.9,
    margin: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """
    Force subject when the question strongly matches a subject value.

    Uses best_typed_matches(question, meta) so that subjects get looser filtering
    (max_df_frac=0.8), which is important for short categorical subject vocabularies.
    """
    hits = best_typed_matches(question, meta)
    subj_hit = hits.get("subjects")
    if not subj_hit:
        return None

    subject, score = subj_hit

    if score < threshold:
        return None

    return {
        "force_subject": subject,
        "subject_score": score,
        "reason": "direct_subject_match(best_typed_matches)",
    }


def chart_intent_guard(question: str, meta: dict) -> Optional[Dict[str, Any]]:
    q = (question or "").strip()
    if not q:
        return None
    # Strong signals for chart_request
    if CHART_HINT.search(q) or BREAKDOWN_HINT.search(q):
        return {
            "force_category": "chart_request",
            "reason": "chart_or_breakdown_intent",
        }
    
    return None


################### FIND WHICH GUARD FIRES FIRST ###################
GUARDS = [
    chart_intent_guard,
    ranking_intent_guard,        # may force row_filter + limit
    measure_group_guard,         # pins correct measure_group if needed
    direct_subject_first_guard,  # prevents symmetric subject flips
    age_range_sum_guard,         # under X
    over_age_guard,              # over X / X+
    direct_measure_first_guard,  # fallback: single direct measure -> cell_lookup
]


def merge_constraints(
    merged: Optional[dict],
    override: dict,
    guard_name: str,
) -> dict:
    merged = merged or {}

    # reasons list
    merged.setdefault("reasons", [])
    reason = override.get("reason")
    if reason:
        merged["reasons"].append(f"{guard_name}:{reason}")

    # optional debug payload (keeps scores etc. without clobber)
    merged.setdefault("debug", {})
    merged["debug"][guard_name] = {k: v for k, v in override.items() if k != "reason"}

    # merge force_* keys (default: first-wins to avoid late guards clobbering category)
    for k, v in override.items():
        if k in {"reason", "score"}:
            continue
        if not k.startswith("force_"):
            continue

        # first wins to preserve earlier "shape" decisions
        if k not in merged:
            merged[k] = v

    # --- NEW RULE: measures_in beats earlier measure_group guesses ---
    if override.get("force_measures_in"):
        
        # If this same override doesn't explicitly provide a measure_group,
        # invalidate previously-chosen measure_group (often lexical "percent" noise).
        if "force_measure_group" not in override:
            merged.pop("force_measure_group", None)
            merged.pop("measure_group_score", None)

    # If you *want* some keys to be last-wins (e.g., force_measures_in from under-range should override),
    # you can allowlist them:
    LAST_WINS = {"force_measures_in", "force_measure", "force_measure_group"}
    for k in LAST_WINS:
        if k in override:
            merged[k] = override[k]

    return merged

def resolve_measure_override(question: str, meta: dict) -> Optional[dict]:
    merged = None
    for guard in GUARDS:
        override = guard(question, meta)
        if not override:
            continue
        merged = merge_constraints(merged, override, guard.__name__)

    return merged