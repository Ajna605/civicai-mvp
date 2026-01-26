# End-to-end demo entry point
# - load index
# - accept question
# - print answer + sources

import time
from rag.settings import apply_settings
from rag.query_engine import query_civicai
import torch


if __name__ == "__main__":
    apply_settings()     # loads model once

    # q = "Does the Coral Gables plan specify housing density limits?"
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    start = time.time()
    # q = "What is the restriction in regards to residential development throughout the coastal area of East of Old Cutler Road?"
    # q = "Does the Coral Gables plan specify housing density limits?"
    # q = "Explain what is mentioned in Policy FLU-1.1.2."
    # q = "What does the document say about Policy ADM-1.5.3.?"
    # q = "Who are partners of the City?"
    # q = "What are the residential density limits in the coastal area east of Old Cutler Road"
    # Q5
    # q =  "What is in the table showing Recreation facilities radius standard?"
    #Q6
    q = "Does the Coral Gables plan specify housing density limits?"

    print(query_civicai(q))
    end = time.time()
    print("Time taken: ", (end - start)/60, "minutes")
