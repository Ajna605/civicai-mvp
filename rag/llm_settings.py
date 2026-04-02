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
        max_new_tokens=250,
        device_map="cuda",
        generate_kwargs={"do_sample": False, "repetition_penalty": 1.08},
        model_kwargs={"dtype": torch.bfloat16},
    )


def llm_verbalize_answer(question: str, context: str) -> str:
    llm = Settings.llm
   
    prompt = f"""You are answering a user question using ONLY the evidence below.

Write a natural, synthesized answer that combines relevant information across the evidence.
Do NOT include citations like [1], do NOT mention the evidence block numbers, and do NOT quote long passages.
If the evidence is insufficient, say what is missing.

Question: {question}

Evidence:
{context}

Answer:"""
    resp = llm.complete(prompt)
    return getattr(resp, "text", str(resp)).strip()