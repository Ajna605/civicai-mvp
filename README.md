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

