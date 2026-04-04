SCHEMA_TEXT = """
You must output ONLY ONE valid JSON. No prose. No markdown.

Hard constraints (critical):
- NEVER output placeholder text like "<EXACT ...>" in the final JSON.
- Every string value must be a real exact string copied from METADATA (or from FORCED CONSTRAINTS).
- If you cannot find an exact string in METADATA, output {"category":"analytics_rank","query":{...}} anyway but choose the closest valid value that exists in METADATA (do not use placeholders).
- Do NOT output multiple JSON objects.


For analytics_rank:
- inputs[].measure_group MUST be from METADATA.measure_headings (exact string match).
- inputs[].measure_match_any should be phrases to find the measure

Return exactly:
{"category":"analytics_rank","query":{...}}

Enum values:
- metric_family: healthcare_access_pressure|housing_cost_pressure|climate_risk_pressure
- group_dim: subject|label|measure|raw_row|raw_col
- order: asc|desc

analytics_rank: 
{"category":"analytics_rank","query":{
  "table_name":"<EXACT DuckDB table name>",
  "metric_family":"<string>",
  "geo":"<EXACT geo from METADATA.geo>",
  "group_dim":"<string>",
  "stat_type":"<EXACT stat_types from METADATA.stat_types>",
  "time":{"mode":"snapshot|timeseries","years":[<START_YEAR>,<END_YEAR>]},
  "inputs":[{"metric_key":"<string>","measure_group":"<EXACT measure_group>","measure_match_any":["<string>", "<string>"]}],
  "order":"desc",
  "limit":10
}}"""

FEWSHOT_TEXT = """
EXAMPLE
Q: Which racial groups in <one place> are most likely to face the greatest <topic> pressure over the next 5–10 years?
A: {"category":"analytics_rank","query":{"table_name":"insurance_facts","metric_family":"healthcare_access_pressure","geo":"<EXACT geo from METADATA.geos>","group_dim":"measure","stat_type":"Estimate","time":{"mode":"timeseries","years":[<START_YEAR>,<END_YEAR>]},"inputs":[{"metric_key":"uninsured_rate","measure_group":"RACE AND HISPANIC OR LATINO ORIGIN","measure_match_any":["uninsured", "no health insurance", "percent uninsured"]},{"metric_key":"poverty_rate","measure_group":"<EXACT measure_group>","measure_match_any":["poverty","below poverty"]},{"metric_key":"age65plus_share","measure_group":"<EXACT measure_group>","measure_match_any":["65 years","65 and over"]},{"metric_key":"disability_share","measure_group":"<EXACT measure_group>","measure_match_any":["disability","disabled"]}],"scoring":{"method":"zscore_weighted_sum"},"order":"desc","limit":10}}
"""

CATEGORY_SCHEMAS = {"analytics_rank": ('analytics_rank: {"category":"analytics_rank","query":{"table_name":"insurance_facts","metric_family":"<string>","geo":"<EXACT geo from METADATA.geos (or geo list)>","group_dim":"<EXACT one of: subject|label>","stat_type":"Estimate","time":{"mode":"snapshot|timeseries","years":[<START_YEAR>,<END_YEAR>]},"inputs":[{"metric_key":"<string>","measure_group":"<EXACT measure_group>","measure_match_any":["<string>","<string>"]}],"order":"desc","limit":10''}}')}

## metric family depicts enum decides what is being measured
## metric_key depicts what is being measured, not from metadata
## measure_match_any is measure from the right rows
