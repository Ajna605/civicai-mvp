# Model + embeddings setup, generation parameters,
# device configuration

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

def apply_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = None

    # Settings.llm = HuggingFaceLLM(
    #     model_name="Qwen/Qwen2.5-3B-Instruct",
    #     tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
    #     # model_name="Qwen/Qwen2.5-7B-Instruct",   # safer than 14B for iteration
    #     # tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
    #     context_window=8192,
    #     max_new_tokens=250,
    #     device_map = "cuda",
    #     # generate_kwargs={
    #     #     "temperature": 0.1,
    #     #     "do_sample": False,
    #     #     "repetition_penalty": 1.2,
    #     #     "no_repeat_ngram_size": 4,
    #     # },
    #     model_kwargs={"dtype": torch.bfloat16}
    # )
