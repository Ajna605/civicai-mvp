## Texts prompts

SCHEMA_TEXT = r"""
You must output ONLY valid JSON. No prose. No markdown.

Return exactly:
{"category":"cell_lookup|aggregation|row_filter|chart_request","query":{...}}

Hard constraints:
- Choose label/measure/subject/stat_type by copying an EXACT string from METADATA.
- Do not invent strings. Do not normalize DB strings.
- No extra keys anywhere.
- For enums, use exact allowed values.

Enum values:
- aggregation.query.op: "sum"|"avg"|"min"|"max"|"count"
- row_filter.query.order: "asc"|"desc"
- chart_request.query.chart_type: "categorical"|"time_series"|"compare_labels"
- chart_request.query.sort: "x_asc"|"x_desc"|"y_asc"|"y_desc"
- stat_type: "Estimate"|"Margin of Error"

Schemas:

cell_lookup:
{"category":"cell_lookup","query":{"label":"...","measure":"...","subject":"...","stat_type":"Estimate|Margin of Error"}}

aggregation:
{"category":"aggregation","query":{"op":"sum|avg|min|max|count","filters":{"label":"...","subject":"...","stat_type":"...","measures_in":[...]}}}

row_filter:
{"category":"row_filter","query":{"filters":{"label":"...","subject":"...","stat_type":"...","measures_in":[...]},"order":"asc|desc","limit":number}}

chart_request:
{"category":"chart_request","query":{"chart_type":"categorical|time_series|compare_labels","filters":{"label":"...","subject":"...","stat_type":"...","measures_in":[...]},"x":"measure|year|label|geo","y":"value","sort":"x_asc|x_desc|y_asc|y_desc"}}

Rules:
- If the question asks for margin of error / MOE -> stat_type="Margin of Error"
- Otherwise default stat_type="Estimate"
"""

# Few-shot examples (you can expand later)
FEWSHOT_TEXT = r"""
EXAMPLES:

Q: How many males per 100 females?
A: {"category":"cell_lookup","query":{"label":"Coral Gables city, Florida","measure":"Sex ratio (males per 100 females)","subject":"Total","stat_type":"Estimate"}}

Q: What is the most populated age range of males in Coral Gables?
A: {"category":"row_filter","query":{"filters":{"label":"Coral Gables city, Florida","subject":"Male","stat_type":"Estimate","measures_in":["Under 5 years","5 to 9 years","10 to 14 years","15 to 19 years","20 to 24 years","25 to 29 years","30 to 34 years","35 to 39 years","40 to 44 years","45 to 49 years","50 to 54 years","55 to 59 years","60 to 64 years","65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over"]},"order":"desc","limit":1}}

Q: Create a bar chart of age groups vs total population.
A: {"category":"chart_request","query":{"chart_type":"categorical","filters":{"label":"Coral Gables city, Florida","subject":"Total","stat_type":"Estimate","measures_in":["Under 5 years","5 to 9 years","10 to 14 years","15 to 19 years","20 to 24 years","25 to 29 years","30 to 34 years","35 to 39 years","40 to 44 years","45 to 49 years","50 to 54 years","55 to 59 years","60 to 64 years","65 to 69 years","70 to 74 years","75 to 79 years","80 to 84 years","85 years and over"]},"x":"measure","y":"value","sort":"x_asc"}}
"""