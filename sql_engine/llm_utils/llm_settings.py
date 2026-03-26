from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from llama_index.llms.huggingface import HuggingFaceLLM

@dataclass
class ParamLLMConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # for faster run
    context_window: int = 8192
    max_new_tokens: int = 300

def build_param_llm(cfg: Optional[ParamLLMConfig] = None) -> HuggingFaceLLM:
    cfg = cfg or ParamLLMConfig()

    llm = HuggingFaceLLM(
        model_name=cfg.model_name,
        tokenizer_name=cfg.model_name,
        context_window=cfg.context_window,
        max_new_tokens=cfg.max_new_tokens,
        generate_kwargs={
            "do_sample": False,
        },
        model_kwargs={
            "torch_dtype": "auto"
        },
    )
    return llm

def generate_json_only(llm: HuggingFaceLLM, prompt: str) -> str:
    resp = llm.complete(prompt)
    text = getattr(resp, "text", str(resp))
    return text.strip()