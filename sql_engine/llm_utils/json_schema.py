## Texts prompts

SCHEMA_TEXT = r"""
You must output ONLY ONE valid JSON. No prose. No markdown.

Return exactly:
{"category":"cell_lookup|aggregation|row_filter|chart_request","query":{...}}

Hard constraints:
- Choose label/measure/subject/stat_type by copying an EXACT string from METADATA.
- Do not invent strings. Do not normalize DB strings.
- Do NOT output multiple JSON objects.
- No extra keys anywhere.
- For enums, use exact allowed values.
- If query uses measures_in, you MUST also provide measure_group.
- measure_group must be chosen from METADATA.measure_headings.
- Every measure in measures_in must belong to METADATA.measure_groups[measure_group].
- Never mix measures from different measure groups.

Enum values:
- aggregation.query.op: "sum"|"avg"|"min"|"max"|"count"
- row_filter.query.order: "asc"|"desc"
- chart_request.query.chart_type: "categorical"|"time_series"|"compare_labels"
- chart_request.query.viz_type": "bar" | "pie" | "line" | "scatter"
- stat_type: "Estimate"|"Margin of Error"

Schemas:

cell_lookup:
{"category":"cell_lookup","query":{"label":"...","measure":"...","subject":"...","stat_type":"Estimate|Margin of Error"}}

aggregation:
{"category":"aggregation","query":{"measure_group":"...","op":"sum|avg|min|max|count","filters":{"label":"...","subject":"...","stat_type":"Estimate|Margin of Error","measures_in":[...]}}}

row_filter:
{"category":"row_filter","query":{"measure_group":"...","filters":{"label":"...","subject":"...","stat_type":"Estimate|Margin of Error","measures_in":[...]},"order":"asc|desc","limit":number}}

chart_request:
{"category":"chart_request","query":{"measure_group":"...","chart_type":"categorical|time_series|compare_labels","viz_type":"bar|pie|line|scatter","filters":{"label":"...","subject":"...","stat_type":"Estimate|Margin of Error","measures_in":[...]},"x":"measure|year|label|geo","y":"value"}}

Rules:
- If the question asks for margin of error / MOE -> stat_type="Margin of Error"
- Otherwise default stat_type="Estimate"
- Use cell_lookup for a single direct measure.
- Use aggregation, row_filter, or chart_request only when multiple measures are needed.
"""

# Few-shot examples (you can expand later)
FEWSHOT_TEXT = r"""
EXAMPLES:

Q: What is the value for <one specific measure> in <one specific place>?
A: {"category":"cell_lookup","query":{"label":"<EXACT label from METADATA.labels>","measure":"<EXACT measure from METADATA.measures>","subject":"<EXACT subject from METADATA.subjects>","stat_type":"Estimate"}}
 
Q: Which category has the highest value among a set of measures?
A: {"category":"row_filter","query":{"measure_group":"<EXACT measure_group>","filters":{"label":"<EXACT label>","subject":"<EXACT subject>","stat_type":"Estimate","measures_in":["<EXACT measure 1>","<EXACT measure 2>"]},"order":"desc","limit":1}}
 
Q: Create a bar chart comparing multiple measures for one label/subject.
A: {"category":"chart_request","query":{"measure_group":"<EXACT measure_group>","chart_type":"categorical","filters":{"label":"<EXACT label>","subject":"<EXACT subject>","stat_type":"Estimate","measures_in":["<EXACT measure 1>","<EXACT measure 2>"]},"x":"measure","y":"value","sort":"x_asc"}}
 
Q: What is the total/average/min/max across several measures?
A: {"category":"aggregation","query":{"measure_group":"<EXACT measure_group>","op":"sum","filters":{"label":"<EXACT label>","subject":"<EXACT subject>","stat_type":"Estimate","measures_in":["<EXACT measure 1>","<EXACT measure 2>"]}}}
 """