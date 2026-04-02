import re
from typing import List, Dict, Any
import json
from pathlib import Path

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())

def contains_any(text: str, needles: List[str]) -> bool:
    t = normalize(text)
    return any(normalize(n) in t for n in (needles or []) if n and n.strip())

def contains_all(text: str, needles: List[str]) -> bool:
    t = normalize(text)
    return all(normalize(n) in t for n in (needles or []) if n and n.strip())


def load_tests(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
# ----------------------------
# ID detection
# ----------------------------

def extract_code_from_title(title: str | None) -> str | None:
    if not title:
        return None
    m = re.search(r"\b([A-Z]{2,8}-\d+(?:\.\d+)*)\b", title)
    return m.group(1) if m else None

###### FOR SQL Query Building
TOKEN_NORMALIZATION = {
    "percentage": "percent",
    "percentages": "percent",
    "avg": "average",
    "mean": "average",
    "males": "male",
    "females": "female",
}

###### EXTRACT 1ST JSON OBJECT FROM LLM RESPONSE
def extract_first_valid_param_obj(text: str):
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
            if (
                isinstance(obj, dict)
                and set(obj.keys()) == {"category", "query"}
                and isinstance(obj.get("category"), str)
                and isinstance(obj.get("query"), dict)
            ):
                return obj
        except Exception:
            pass
        i += 1

    raise ValueError("No valid top-level JSON object found")


## Make table name from filename
def infer_table_name(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    if stem.endswith("_facts"):
        return stem

    if stem and stem[0].isdigit():
        stem = f"t_{stem}"

    return f"{stem}_facts"

## CSV jsonl fields condensed for indexing
def csv_record_to_text(rec: dict) -> str:
    label = rec.get("label") or rec.get("geo") or ""
    measure = rec.get("measure") or ""
    subject = rec.get("subject") or ""
    stat_type = rec.get("stat_type") or ""
    year = rec.get("year")
    year_str = f"{year}" if year not in (None, "", "null") else ""

    raw_value = rec.get("raw_value")
    value = rec.get("value")
    value_str = ""
    if raw_value not in (None, "", "null"):
        value_str = str(raw_value)
    elif value not in (None, "", "null"):
        value_str = str(value)

    raw_row = rec.get("raw_row") or ""
    raw_col = rec.get("raw_col") or ""

    # Keep it short but information-dense
    parts = [
        f"Location: {label}" if label else None,
        f"Measure: {measure}" if measure else None,
        f"Subject: {subject}" if subject else None,
        f"Stat: {stat_type}" if stat_type else None,
        f"Year: {year_str}" if year_str else None,
        f"Value: {value_str}" if value_str else None,
        f"Row: {raw_row}" if raw_row else None,
        f"Column: {raw_col}" if raw_col else None,
    ]
    return " | ".join([p for p in parts if p])


