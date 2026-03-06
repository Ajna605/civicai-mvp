import re
import math
from typing import Dict, List, Optional, Tuple, Any, Set

NUM_RE = re.compile(r"\b\d{1,3}\b")

def extract_numbers(text: str) -> Set[int]:
    return {int(x) for x in NUM_RE.findall(text)}

# keep STOP small; don't include "median", "ratio", etc.
STOP = {
    "what","is","the","of","in","for","a","an","and","or","on","at","by",
    "estimate","estimated","how","many","much","people","population","percent","percentage",
    "city","county","state"
}

def tokenize(text: str):
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s\-\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    raw = s.split()

    toks = []
    for t in raw:
        if t in STOP:
            continue
        # keep pure numbers always
        if t.isdigit():
            toks.append(t)
            continue
        # keep short tokens if they are important connectors
        if t in {"to", "under", "over"}:
            toks.append(t)
            continue
        if len(t) <= 2:
            continue
        # simple plural strip
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        toks.append(t)
    return toks

def build_measure_token_stats(measures: List[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Returns:
      df[token] = number of measures containing token
      idf[token] = log((N+1)/(df+1)) + 1  (smooth, deterministic)
    """
    df: Dict[str, int] = {}
    N = 0
    for m in measures:
        N += 1
        seen = set(tokenize(m))
        for t in seen:
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1) / (d + 1)) + 1.0
    return df, idf

def best_direct_measure_match(
    question: str,
    measures: List[str],
    df: Dict[str, int],
    idf: Dict[str, float],
    max_df_frac: float = 0.35,
) -> Optional[Tuple[str, float]]:
    """
    Deterministic weighted-overlap scoring:
      score = sum(idf[t] for t in overlap_tokens) / sum(idf[t] for t in question_tokens_kept)
    Only counts tokens that exist in metadata vocabulary.
    Prefers rare tokens (lower df -> higher idf).
    """
    q_tokens = tokenize(question)
    if not q_tokens:
        return None

    # keep only tokens that exist in metadata vocab
    vocab = set(df.keys())
    q_tokens = [t for t in q_tokens if t in vocab]
    if not q_tokens:
        return None

    # optional: drop extremely common tokens (e.g., "year" might be everywhere)
    N_measures = max(1, len(measures))
    q_tokens_kept = []
    for t in q_tokens:
        if df.get(t, 0) / N_measures <= max_df_frac:
            q_tokens_kept.append(t)
    if not q_tokens_kept:
        # if everything is common, keep them anyway
        q_tokens_kept = q_tokens

    denom = sum(idf.get(t, 1.0) for t in set(q_tokens_kept))

    best_m = None
    best_score = 0.0

    q_set = set(q_tokens_kept)
    for m in measures:
        m_set = set(tokenize(m))
        overlap = q_set & m_set
        if not overlap:
            continue

        numer = sum(idf.get(t, 1.0) for t in overlap)

        q_nums = extract_numbers(question)
        m_nums = extract_numbers(m)

        if m_nums:
            num_overlap = len(q_nums & m_nums)
            num_precision = num_overlap / len(m_nums)  # penalize extra numbers
        else:
            num_precision = 1.0

        score = (numer / max(1e-9, denom)) * num_precision

        if score > best_score:
            best_score = score
            best_m = m

    if best_m is None:
        return None
    return best_m, best_score

def direct_measure_first_guard(question: str, meta: dict, threshold: float = 0.55) -> Optional[Dict[str, Any]]:
    """
    Returns the exact measure string from metadata if question strongly implies a direct measure.
    threshold lower than before because IDF scoring is already conservative.
    """
    measures = meta.get("measures", [])
    if not measures:
        return None

    df, idf = build_measure_token_stats(measures)
    hit = best_direct_measure_match(question, measures, df, idf)
    if not hit:
        return None

    measure, score = hit

    if score < threshold:
        return None
    ## If question is "under X", reject a "A to X" bucket unless A is explicitly mentioned.
    UNDER_HINT = re.compile(r"\b(under|less than|below|at most)\b", re.I)
    if UNDER_HINT.search(question):
        q_nums = extract_numbers(question)
        m_nums = extract_numbers(measure)
        if m_nums and q_nums and not m_nums.issubset(q_nums):
            return None
        
    print("GUARD FIRE:", measure, score) 
    ## all return cell_lookup as info is in table
    return {
        "force_category": "cell_lookup",
        "force_measure": measure,
        "score": score,
        "reason": "direct_measure_match"
    }


UNDER_RE = re.compile(r"\b(under|less than|below)\s+(\d{1,3})\b", re.I)
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

def age_range_sum_guard(question: str, meta: dict) -> Optional[Dict[str, Any]]:
    measures = meta.get("measures", []) or []
    m = UNDER_RE.search(question)
    if not m:
        return None

    target = int(m.group(2))

    # Case 1: dataset already has "Under X years" bucket -> allow direct cell lookup
    direct = f"Under {target} years"
    if direct in measures:
        return {
            "force_category": "cell_lookup",
            "force_measure": direct,
            "reason": "direct_under_bucket"
        }

    # Case 2: build measures_in list deterministically -> force aggregation sum
    measures_in = age_under_measures_in(target, measures)
    if not measures_in:
        return None  # no usable bands found

    return {
        "force_category": "aggregation",
        "force_op": "sum",
        "force_measures_in": measures_in,
        "reason": "under_range_sum"
    }