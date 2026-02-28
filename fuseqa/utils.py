
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

 

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationConfig:
    max_input_tokens: int = 4096
    max_new_tokens: int = 6
    do_sample: bool = False
    repetition_penalty: float = 1.1



# ─────────────────────────────────────────────────────────────────────────────
# Filename utilities
# ─────────────────────────────────────────────────────────────────────────────

def hf_model_to_filename(model_id: str, max_len: int = 120) -> str:
    s = unicodedata.normalize("NFKD", model_id)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_").replace("@", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    s = f"{s}_{ts}"
    return s[:max_len] if len(s) > max_len else s


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(counts: Dict[str, int], em_hits: Dict[str, int], file_name: str, model_name: str, 
                    run_type: str, total_time: float, 
                    groups: Sequence[str] = ("ALL", "LONG-TAIL", "INFREQUENT", "FREQUENT")) -> str:
    _ensure_dir("results")

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("PopQA Exact Match Report — Tiers")
    lines.append("=" * 80)
    lines.append(f"Model:    {model_name}")
    lines.append(f"Mode:     {run_type}")
    lines.append(f"Run Time: {datetime.now().strftime('%Y%m%d-%H%M')}")
    lines.append("")
    lines.append(f"Duration: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    lines.append("")
    lines.append("Exact Match (EM):")

    for name in groups:
        n = int(counts.get(name, 0))
        em = safe_div(int(em_hits.get(name, 0)), n)
        lines.append(f"  {name:<12} n={n:<6} EM={em:.4f}")

    report_file = os.path.join("results", file_name + ".report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(
        f"Saved report: {report_file} | "
        f"Time={total_time:.2f}s | "
        f"ALL={safe_div(int(em_hits.get('ALL', 0)), int(counts.get('ALL', 0))):.4f} | "
        f"LONG-TAIL={safe_div(int(em_hits.get('LONG-TAIL', 0)), int(counts.get('LONG-TAIL', 0))):.4f} | "
        f"INFREQUENT={safe_div(int(em_hits.get('INFREQUENT', 0)), int(counts.get('INFREQUENT', 0))):.4f} | "
        f"FREQUENT={safe_div(int(em_hits.get('FREQUENT', 0)), int(counts.get('FREQUENT', 0))):.4f}"
    )

    return report_file


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm(s: Any) -> str:
    s = "" if s is None else str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)  
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_record(f, record: Dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(a) for a in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return [str(a) for a in v]
            except Exception:
                pass
        return [s]
    return [str(x)]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_context(docs: Optional[Sequence[Any]], k: int = 10, clip_chars: int = 350) -> str:
    parts: List[str] = []
    for d in (docs or [])[:k]:
        txt = d.get("text", "") if isinstance(d, dict) else str(d)
        txt = (txt or "").strip()
        if txt:
            parts.append(txt[:clip_chars])
    return "\n\n".join(parts)


def tier_from_spop(s_pop: int) -> str:
    if s_pop < 100:
        return "LONG-TAIL"
    if s_pop < 10_000:
        return "INFREQUENT"
    return "FREQUENT"

# -------------------------
# Helpers (SRE-aware context)
# -------------------------
SRE_RETR_COL   = "retrieved_docs_sre"
BASE_RETR_COL  = "retrieved_docs"

def build_context_from_sre_list(sre_list, score_th: float, k: int, clip_chars: int) -> str:
    if not isinstance(sre_list, list):
        return ""
    passed = []
    for d in sre_list:
        if not isinstance(d, dict):
            continue
        score = float(d.get("score", -1))
        txt   = (d.get("text") or "").strip()
        if txt and score >= score_th:
            passed.append(txt[:clip_chars])
        if len(passed) >= k:
            break
    return "\n\n".join(passed)

def get_context_for_run(ex: dict, run_type: str, use_context: bool, sre_score_th: float, top_k: int, clip_chars: int) -> str:
    if not use_context:
        return ""

    rt = (run_type or "").upper()

    # SRE run: try retrieved_docs_sre with threshold, else fall back
    if "SRE" in rt:
        sre_list = ex.get(SRE_RETR_COL) or []
        ctx = build_context_from_sre_list(sre_list, score_th=sre_score_th, k=top_k, clip_chars=clip_chars)
        if ctx.strip():
            return ctx

        base_docs = ex.get(BASE_RETR_COL) or []
        return build_context(base_docs, k=top_k, clip_chars=clip_chars)

    # non-SRE run: old behavior
    base_docs = ex.get(BASE_RETR_COL) or []
    return build_context(base_docs, k=top_k, clip_chars=clip_chars)
# ─────────────────────────────────────────────────────────────────────────────
# Prediction post-processing
# ─────────────────────────────────────────────────────────────────────────────

def clean_pred(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return "UNKNOWN"

    s = re.sub(r"\*+", "", s)

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN"

    s = lines[0]

    s = re.sub(r"^[^\w]+", "", s)

    s = re.sub(r"^(answer\s*:)\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("#", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.split(r"[\.!\?:;()\[\]{}]", s, maxsplit=1)[0].strip()
    s = s.strip("\"'`[](){} ").strip()

    if "," in s or " and " in s.lower():
        tmp = s.replace(" and ", ", ")
        parts = [p.strip() for p in tmp.split(",") if p.strip()]
        if parts:
            s = parts[-1]

    return s if s else "UNKNOWN"

def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unified prompting + generation (works across HF LLMs)
# ─────────────────────────────────────────────────────────────────────────────

class LLMAnswerer:
    
    def __init__(self, model, tokenizer, 
                 device: Union[str, torch.device] = "cuda:0", 
                  gen_cfg: GenerationConfig = GenerationConfig(),
                  prompt_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device 
        self.gen_cfg = gen_cfg
        self.prompt_fn = prompt_fn
        self._ensure_pad_token()

    def answer(self, question: str, context: str = "", use_context: bool = True, print_prompt: bool = False) -> str:
        prompt_or_messages = self.prompt_fn(question=question, context=context, use_context=use_context)
        if print_prompt:
            print(f"Prompt:\n{prompt_or_messages}\n{'-'*40}")
        inputs = self._tokenize(prompt_or_messages)
        gen_text = self._generate_and_strip(inputs)
        return clean_pred(self._pick_best_line(gen_text))

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _ensure_pad_token(self) -> None:
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos
 


    def _tokenize(self, prompt_or_messages):
        if isinstance(prompt_or_messages, list) and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    prompt_or_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = prompt_or_messages[0]["content"]
        else:
            text = prompt_or_messages

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.gen_cfg.max_input_tokens
        )
        return {k: v.to(self.device) for k, v in inputs.items()}


    def _generate_and_strip(self, inputs: Dict[str, torch.Tensor]) -> str:
        with torch.inference_mode():
            out = self.model.generate(
            **inputs,
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=False,
            temperature=1.0,   # neutral
            top_p=1.0,         # neutral
            repetition_penalty=self.gen_cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Robust “strip prompt” logic that works for both chat-templated and plain prompts:
        # Use input length in tokens, not string-prefix matching (more reliable across templates).
        input_len = int(inputs["input_ids"].shape[-1])
        gen_ids = out[0][input_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return gen_text

    def _pick_best_line(self, gen_text: str) -> str:
        lines = [ln.strip() for ln in (gen_text or "").splitlines() if ln.strip()]
        if not lines:
            return "UNKNOWN"

        # Prefer the first short span (1–4 tokens) to avoid explanations.
        for ln in lines:
            if 1 <= len(ln.split()) <= 4:
                return ln
        return lines[0]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible function (drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

def ask_llm_generate(model, tokenizer, question: str, context: str, use_context: bool, device: Union[str, torch.device], print_prompt: bool = False, prompt_fn=None) -> str:
    return LLMAnswerer(model=model, tokenizer=tokenizer, device=device, prompt_fn=prompt_fn)\
            .answer(question=question, context=context, use_context=use_context, print_prompt=print_prompt) # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Private filesystem helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
