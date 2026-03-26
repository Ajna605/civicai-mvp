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

Q: How many males per 100 females?
A: {"category":"cell_lookup","query":{"label":"Coral Gables city, Florida","measure":"Sex ratio (males per 100 females)","subject":"Total","stat_type":"Estimate"}}

Q: What is the most populated age range of males in Coral Gables?
A: {"category":"row_filter","query":{"measure_group":"AGE","filters":{"label":"Coral Gables city, Florida","subject":"Male","stat_type":"Estimate","measures_in":["Under 5 years","5 to 9 years","10 to 14 years","15 to 19 years","20 to 24 years","25 to 29 years","30 to 34 years","35 to 39 years","40 to 44 years","45 to 49 years","50 to 54 years","55 to 59 years","60 to 64 years","65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over"]},"order":"desc","limit":1}}

Q: Create a bar chart of age groups vs total population.
A: {"category":"chart_request","query":{"measure_group":"AGE","chart_type":"categorical","viz_type":"bar","filters":{"label":"Coral Gables city, Florida","subject":"Total","stat_type":"Estimate","measures_in":["Under 5 years","5 to 9 years","10 to 14 years","15 to 19 years","20 to 24 years","25 to 29 years","30 to 34 years","35 to 39 years","40 to 44 years","45 to 49 years","50 to 54 years","55 to 59 years","60 to 64 years","65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over"]},"x":"measure","y":"value"}}

Q: What percentage of the population is aged less than 24?
A: {"category":"aggregation","query":{"measure_group":"AGE","op":"sum","filters":{"label":"Coral Gables city, Florida","subject":"Percent","stat_type":"Estimate","measures_in":["Under 5 years","5 to 9 years","10 to 14 years","15 to 19 years","20 to 24 years"]}}}
"""