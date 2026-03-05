# llm_settings.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from llama_index.llms.huggingface import HuggingFaceLLM

@dataclass
class ParamLLMConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    context_window: int = 8192
    max_new_tokens: int = 600
    temperature: float = 0.0
    top_p: float = 0.1

def build_param_llm(cfg: Optional[ParamLLMConfig] = None) -> HuggingFaceLLM:
    cfg = cfg or ParamLLMConfig()

    # Note: HuggingFaceLLM will use transformers under the hood.
    # If you’re using 4-bit/8-bit quantization, you can pass model_kwargs accordingly.
    llm = HuggingFaceLLM(
        model_name=cfg.model_name,
        tokenizer_name=cfg.model_name,
        context_window=cfg.context_window,
        max_new_tokens=cfg.max_new_tokens,
        generate_kwargs={
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "do_sample": False,   # important for determinism
        },
        # If needed:
        model_kwargs={"dtype": "auto"}
    )
    return llm

def generate_json_only(llm: HuggingFaceLLM, prompt: str) -> str:
    # LlamaIndex HF LLM returns a completion object sometimes;
    # safest is to call .complete(prompt).text
    resp = llm.complete(prompt)
    text = getattr(resp, "text", str(resp))
    return text.strip()