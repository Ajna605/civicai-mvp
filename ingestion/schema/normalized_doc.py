from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Table:
    table_id: str
    page: int
    caption: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None


@dataclass
class Section:
    section_id: str
    heading_path: List[str]
    page_start: int
    page_end: int
    text: str
    tables: List[Table]


@dataclass
class SourceInfo:
    file_name: str
    file_type: str
    sha256: str
    ingested_at: str


@dataclass
class NormalizedDoc:
    doc_id: str
    source: SourceInfo
    sections: List[Section]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
