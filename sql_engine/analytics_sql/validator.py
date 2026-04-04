from typing import Any, Dict

def build_repair_prompt(question: str, bad_json: str, error: str, meta: Dict[str, Any]) -> str:
    return bad_json