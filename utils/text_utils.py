import re

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ----------------------------
# ID detection
# ----------------------------

def extract_code_from_title(title: str | None) -> str | None:
    if not title:
        return None
    m = re.search(r"\b([A-Z]{2,8}-\d+(?:\.\d+)*)\b", title)
    return m.group(1) if m else None