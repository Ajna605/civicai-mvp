from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from llama_index.llms.huggingface import HuggingFaceLLM
import json
import torch
from typing import Any, Dict, Optional

@dataclass
class ParamLLMConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # for faster run
    context_window: int = 2048   # or 4096 if you truly need it
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
            "torch_dtype": torch.bfloat16
        },
    )
    return llm

def generate_json_only(llm: HuggingFaceLLM, prompt: str) -> str:
    resp = llm.complete(prompt)
    text = getattr(resp, "text", str(resp))
    return text.strip()

@dataclass
class AnswerLLMConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    context_window: int = 8192
    max_new_tokens: int = 160  # shorter than params is fine too

def build_answer_llm(cfg: Optional[AnswerLLMConfig] = None) -> HuggingFaceLLM:
    cfg = cfg or AnswerLLMConfig()
    return HuggingFaceLLM(
        model_name=cfg.model_name,
        tokenizer_name=cfg.model_name,
        context_window=cfg.context_window,
        max_new_tokens=cfg.max_new_tokens,
        generate_kwargs={"do_sample": False},
        model_kwargs={"dtype": torch.float16,},
    )


DEFAULT_PROMPT_TEMPLATE = """
You are an assistant that turns database query results into a concise answer.

Rules:
- Use ONLY facts from FACT_PACK_JSON.
- Do NOT add new numbers.
- Do NOT mention SQL or implementation details.
- If FACT_PACK_JSON contains a numeric value, include it exactly.

User question:
{user_question}

FACT_PACK_JSON:
{fact_pack_json}
""".strip()

def llm_verbalize_answer(
    llm,
    fact_pack: Dict[str, Any],
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    user_question: Optional[str] = None,
) -> str:
    """
    Convert fact_pack -> natural language using an LLM.
    prompt_template can be overridden by the caller depending on task type.
    """
    uq = user_question or fact_pack.get("query") or ""
    prompt = prompt_template.format(
        user_question=uq,
        fact_pack_json=json.dumps(fact_pack, ensure_ascii=False),
    ).strip()

    resp = llm.complete(prompt)
    return getattr(resp, "text", str(resp)).strip()
