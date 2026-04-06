# Convert ranking results to info ready to be fed to LLM
def llm_payload_from_ranking(
    *,
    geo: str,
    section: str,
    subject: str,
    years: list[int],
    stat_type: str,
    ranking: list[dict],
    risk_definition: str,
    top_k: int = 5,
    chart_paths: dict | None = None,
):
    def pick_evidence_points(series):
        # series: [(year,value), ...] sorted
        if not series:
            return []
        s = list(series)
        # first, last, plus 2 middle points (roughly)
        points = [s[0]]
        if len(s) >= 4:
            points.append(s[len(s)//3])
            points.append(s[(2*len(s))//3])
        if s[-1] != s[0]:
            points.append(s[-1])
        # unique by year
        seen = set()
        out = []
        for y, v in points:
            if y in seen:
                continue
            seen.add(y)
            out.append({"year": int(y), "value": float(v)})
        return out

    top = ranking[:top_k]

    return {
        "task": "Write a natural-language summary of the trend analysis and identify which groups are most at risk.",
        "definition_of_risk": risk_definition,
        "context": {
            "geo": geo,
            "section": section,
            "subject": subject,
            "stat_type": stat_type,
            "years": years,
        },
        "top_groups_by_risk": [
            {
                "group": r["group"],
                "latest_year": int(r["latest_year"]),
                "latest_value": float(r["latest_value"]),
                "first_year": int(r["first_year"]),
                "first_value": float(r["first_value"]),
                "delta": float(r["delta"]),
                "slope_per_year": float(r["slope_per_year"]),
                "forecast_value": float(r["forecast_value"]),
                "evidence_points": pick_evidence_points(r.get("series", [])),
            }
            for r in top
        ],
        "chart_artifacts": chart_paths or {},
        "constraints": {
            "do_not_invent_numbers": True,
            "do_not_explain_causes": True,
            "tone": "clear, neutral, non-technical",
            "length_words_max": 180
        }
    }


ANALYTICS_PROMPT = """
You are an assistant writing a short analytical summary for a general audience.

Rules:
- Use ONLY facts from FACT_PACK_JSON.
- Do NOT invent numbers or years.
- You may compare groups ONLY if the comparison is directly supported by the provided ranking/values.
- Explain (a) who is highest currently, and (b) what the next 5-year projection suggests (simple linear projection).
- Keep it to 4–6 sentences.

User question:
{user_question}

FACT_PACK_JSON:
{fact_pack_json}
""".strip()

import json
from typing import Any, Dict, Optional

ANALYTICS_PROMPT_TEMPLATE = """
You are an assistant that summarizes trend/risk analysis results for a general audience.

Rules:
- Use ONLY facts from FACT_PACK_JSON.
- Do NOT invent numbers, years, or groups.
- You MAY compare groups only when a ranking/value is provided in FACT_PACK_JSON.
- Clearly label projections as "simple linear projection" (not a guarantee).
- Mention:
  (1) the highest current value (latest year) and its group,
  (2) what the 5-year projection suggests for 1–2 groups,
  (3) one brief caveat about uncertainty/volatility.
- Output {max_sentences} sentences maximum.

User question:
{user_question}

FACT_PACK_JSON:
{fact_pack_json}
""".strip()

def llm_verbalize_analytics_summary(
    llm,
    fact_pack: Dict[str, Any],
    *,
    user_question: str,
    prompt_template: str = ANALYTICS_PROMPT_TEMPLATE,
    max_sentences: int = 6,
) -> str:
    prompt = prompt_template.format(
        user_question=user_question or "",
        max_sentences=max_sentences,
        fact_pack_json=json.dumps(fact_pack, ensure_ascii=False),
    ).strip()

    resp = llm.complete(prompt)
    return getattr(resp, "text", str(resp)).strip()