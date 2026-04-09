# Model + embeddings setup, generation parameters,
# device configuration

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

def rag_llm():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Settings.llm = None

    Settings.llm = HuggingFaceLLM(
        # 3B for RAG
        model_name="Qwen/Qwen2.5-3B-Instruct",
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        # model_name="Qwen/Qwen2.5-14B-Instruct",
        # tokenizer_name="Qwen/Qwen2.5-14B-Instruct",
        #7B Middleway
        # model_name="Qwen/Qwen2.5-7B-Instruct",   # safer than 14B for iteration
        # tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        context_window=8192,
        max_new_tokens=250,
        device_map = "cuda",
        generate_kwargs={
            "do_sample": False,
        },
        model_kwargs={"dtype": torch.bfloat16}
    )

def conv_llm() -> None:
    Settings.llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        context_window=8192,
        max_new_tokens=200,
        device_map="cuda",
        generate_kwargs={"do_sample": False, "repetition_penalty": 1.12},
        model_kwargs={"dtype": torch.bfloat16},
    )

def llm_verbalize_answer(question: str, context: str) -> str:
    llm = Settings.llm
    if llm is None:
        raise RuntimeError("Settings.llm is None")

    prompt = f"""Use ONLY the evidence. Answer in 1–2 sentences.
Do not repeat yourself. Do not add missing-details speculation unless the user asks.

Question: {question}

Evidence:
{context}

Answer:"""
    resp = llm.complete(prompt)
    return getattr(resp, "text", str(resp)).strip()