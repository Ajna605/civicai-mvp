# End-to-end demo entry point
# - load index
# - accept question
# - print answer + sources

from rag.settings import apply_settings
apply_settings()
from rag.query_engine import query_civicai

if __name__ == "__main__":
    # q = "Does the Coral Gables plan specify housing density limits?"
    q = "What is the restriction in regards to residential development throughout the coastal area of East of Old Cutler Road?"
    print(query_civicai(q))