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
    r"\b(chart|plot|graph|visualize|bar chart|line chart|scatter|histogram|trend)\b",
    re.I,
)

RANK_HINT = re.compile(
    r"\b(most|least|highest|lowest|largest|smallest|top|bottom|prominent|common)\b",
    re.I,
)

COMPARE_HINT = re.compile(
    r"\b(compare|comparison|versus|vs\.?|male and female|males and females|both)\b",
    re.I,
)

UNDER_HINT = re.compile(r"\b(under|less than|below|at most)\s+(?P<n>\d{1,3})?\b", re.I)

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
        "score": 1.0,
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
    m = UNDER_MEASURE_RE.match(measure)
    if m:
        high = int(m.group(1)) - 1
        return (0, high)

    m = RANGE_RE.match(measure)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    if "and over" in measure.lower():
        nums = re.findall(r"\d{1,3}", measure)
        if nums:
            low = int(nums[0])
            return (low, 10**9)

    return None

def age_under_measures_in(target: int, measures: List[str]) -> List[str]:
    """
    Select age bands whose upper bound <= target.
    For ACS-style, target=24 should include "20 to 24 years".
    """
    bands = []
    for m in measures:
        b = parse_age_band(m)
        if not b:
            continue
        low, high = b
        if high <= target:
            bands.append((low, high, m))

    # sort by low bound to keep x-axis natural
    bands.sort(key=lambda x: x[0])
    return [m for _, _, m in bands]

OVER_HINT = re.compile(
    r"\b(?P<op>over|above|greater than|at least)\s+(?P<n>\d{1,3})(?:\s+years?)?\b",
    re.I,
)

def age_over_measures_in(target: int, measures: List[str]) -> List[str]:
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


def age_range_sum_guard(question: str, meta: dict) -> Optional[Dict[str, Any]]:
    measures = meta.get("measures", []) or []

    under_m = UNDER_HINT.search(question)
    over_m = OVER_HINT.search(question)

    if under_m:
        target = int(under_m.group("n"))

        # # Case 1: dataset already has exact "Under X years" bucket
        # direct = f"Under {target} years"
        # if direct in measures:
        #     return {
        #         "force_category": "cell_lookup",
        #         "force_measure": direct,
        #         "reason": "direct_under_bucket",
        #     }

        # Case 2: deterministic aggregation over all matching lower-age bands
        measures_in = age_under_measures_in(target, measures)
        if not measures_in:
            return None

        return {
            "force_category": "aggregation",
            "force_op": "sum",
            "force_measures_in": measures_in,
            "reason": "under_range_sum",
        }

    if over_m:
        target = int(over_m.group("n"))

        measures_in = age_over_measures_in(target, measures)
        if not measures_in:
            return None

        return {
            "force_category": "aggregation",
            "force_op": "sum",
            "force_measures_in": measures_in,
            "reason": "over_range_sum",
        }

    return None


################### FIND WHICH GUARD FIRES FIRST ###################
GUARDS = [
    direct_measure_first_guard,
    age_range_sum_guard
]

def resolve_measure_override(question: str, meta: dict):
    for guard in GUARDS:
        override = guard(question, meta)
        if override:
            return override
    return None