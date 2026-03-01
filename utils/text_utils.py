import re
import unicodedata
from typing import Iterable, Optional, List, Dict, Any
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