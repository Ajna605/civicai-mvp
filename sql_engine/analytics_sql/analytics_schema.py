# name=sql_engine/analytics_sql/analytics_schema.py
# (create this file if it doesn't exist)

SCHEMA_TEXT = r"""
You must output ONLY ONE valid JSON object. No prose. No markdown.
Output MUST be a single line of JSON (no newlines).

Return exactly:
{"category":"analytics_rank","query":{...}}

Hard constraints:
- NEVER output placeholders like "<EXACT ...>".
- Copy strings EXACTLY from METADATA (same casing/spaces).
- No extra keys anywhere.
- All required keys MUST be present.

Required schema:

{"category":"analytics_rank","query":{
  "table_name":"<EXACT; MUST equal METADATA.table_name>",
  "metric_family":"healthcare_access_pressure",
  "geo":"<EXACT from METADATA.geos>",
  "section":"<EXACT from METADATA.measure_headings>",
  "subject":"<EXACT from METADATA.metric_names>",
  "stat_type":"Estimate|Margin of Error",
  "time":{"mode":"snapshot|timeseries","years":[2015,2024]},
  "order":"asc|desc",
  "limit":10
}}

Defaults:
- If question asks for MOE -> stat_type="Margin of Error" else "Estimate"
- order="desc" if not specified
- limit=10 if not specified
"""

FEWSHOT_TEXT = """
EXAMPLE
Q: Which racial groups in <one place> are most likely to face the greatest <topic> pressure over the next 5–10 years?
A: {"category": "analytics_rank","query": {"table_name": "insurance_facts","geo": "Coral Gables city, Florida","section": "RACE AND HISPANIC OR LATINO ORIGIN","subject": "Percent Uninsured","stat_type": "Estimate","time": { "mode": "timeseries", "years": <START_YEAR>,<END_YEAR> },"order": "desc","limit": 10}}
"""

CATEGORY_SCHEMAS = {"analytics_rank": ('analytics_rank: {"category":"analytics_rank","query":{"table_name":"insurance_facts","metric_family":"<string>","geo":"<EXACT geo from METADATA.geos (or geo list)>","group_dim":"<EXACT one of: subject|label>","stat_type":"Estimate","time":{"mode":"snapshot|timeseries","years":[<START_YEAR>,<END_YEAR>]},"inputs":[{"metric_key":"<string>","measure_group":"<EXACT measure_group>","measure_match_any":["<string>","<string>"]}],"order":"desc","limit":10''}}')}

## metric family depicts enum decides what is being measured
## metric_key depicts what is being measured, not from metadata
## measure_match_any is measure from the right rows
