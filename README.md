CivicAI MVP for Hult Competition

# Project overview, architecture, setup, run instructions,
# demo flow, known limitations

# Env and Package Instructions
python3.11 -m venv .venv
cd civicai-mvp
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run instructions
# LOAD and PREPROCESS Docs
python ingestion/load_docs.py
python ingestion/preprocess.py
## Create Index

python -m rag.build_index
## Run RAG
python ./main.py


## Identifier-based queries (policy/objective codes) are handled deterministically at retrieval time, not via embeddings.”


## Types of questions

---

## Chart Generation (analytics module)

The `analytics/` package renders chart artifacts from `chart_request` query
objects produced by the LLM parameter generator.

### Quick start

```python
from analytics import AnalyticsRuntime

runtime = AnalyticsRuntime(
    db_path="storage/duckdb/unaris.duckdb",
    out_dir="outputs/charts",   # created automatically if missing
)

# obj is the validated LLM output for a chart_request query
obj = {
    "category": "chart_request",
    "query": {
        "measure_group": "RACE AND HISPANIC OR LATINO ORIGIN",
        "chart_type": "categorical",
        "viz_type": "bar",
        "filters": {
            "label": "Coral Gables city, Florida",
            "subject": "Insured",
            "stat_type": "Estimate",
            "measures_in": ["White alone", "Black or African American alone"],
        },
        "x": "measure",
        "y": "value",
    },
}

result = runtime.run(obj)
# result["ok"]            -> True / False
# result["chart_path"]    -> Path to the saved PNG file
# result["html_path"]     -> Path to the saved interactive HTML (Plotly), or None
# result["chart_summary"] -> {"chart_type", "x_field", "y_field", "row_count", "title", ...}
```

### Supported chart types

| chart_type        | Renderer              | x-axis    | y-axis |
|-------------------|-----------------------|-----------|--------|
| categorical       | Bar chart             | measure   | value  |
| time_series       | Line chart            | year      | value  |
| compare_labels    | Grouped bar chart     | label     | value  |

### Output files

- **PNG** – always produced via Matplotlib (Agg backend, headless-safe).
- **HTML** – produced via Plotly when it is installed (interactive, zoomable).

### Configuration

| Parameter  | Default           | Description                          |
|------------|-------------------|--------------------------------------|
| db_path    | (required)        | Path to the DuckDB database file     |
| out_dir    | outputs/charts    | Directory for chart artefacts        |

### Running tests

```bash
python -m pytest analytics/tests/ -v
```
