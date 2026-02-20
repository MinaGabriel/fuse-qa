
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
    max_new_tokens: int = 16
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.1


@dataclass(frozen=True)
class PromptConfig:
    system_no_context: str = (
    "You are a precise factual question answering system.\n"
    "Return only the exact answer span."
    )

    system_with_context: str = (
        "You are a strict answer extraction system.\n"
        "Extract the exact answer span from the context."
    )

    rules: Tuple[str, ...] = (
    "Answer with the shortest possible span (1-3 words).",
    "Do not explain.",
    "Do not repeat the question.",
)   



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

def build_context(docs: Optional[Sequence[Any]], k: int = 10, clip_chars: int = 250) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Prediction post-processing
# ─────────────────────────────────────────────────────────────────────────────

def clean_pred(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s:
        return "UNKNOWN"

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN"

    s = lines[0]
    s = re.sub(r"^(answer\s*:)\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("#", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.split(r"[\.!\?:;()\[\]{}]", s, maxsplit=1)[0].strip()
    s = s.strip("\"'`[](){} ").strip()

    # If the model returns multiple items, prefer the last (often most specific) token span.
    if "," in s or " and " in s.lower():
        tmp = s.replace(" and ", ", ")
        parts = [p.strip() for p in tmp.split(",") if p.strip()]
        if parts:
            s = parts[-1]

    s = s.strip()
    return s if s else "UNKNOWN"


def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unified prompting + generation (works across HF LLMs)
# ─────────────────────────────────────────────────────────────────────────────

class LLMAnswerer:
    
    def __init__(self, model, tokenizer, device: Union[str, torch.device] = "cuda:0", 
                 prompt_cfg: PromptConfig = PromptConfig(), gen_cfg: GenerationConfig = GenerationConfig()):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_cfg = prompt_cfg
        self.gen_cfg = gen_cfg
        self._ensure_pad_token()

    def answer(self, question: str, context: str = "", use_context: bool = True) -> str:
        prompt_or_messages = self._build_prompt(question=question, context=context, use_context=use_context)
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

    def _build_prompt(self, question: str, context: str, use_context: bool):
        system = self.prompt_cfg.system_with_context if use_context else self.prompt_cfg.system_no_context
        rules = "Rules:\n- " + "\n- ".join(self.prompt_cfg.rules)

        if use_context:
            user = f"{rules}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        else:
            user = f"{rules}\n\nQuestion: {question}\nAnswer:"

        return f"{system}\n\n{user}"


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
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                do_sample=self.gen_cfg.do_sample,
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                repetition_penalty=self.gen_cfg.repetition_penalty,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
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

def ask_llm_generate(model, tokenizer, question: str, context: str, use_context: bool, device: Union[str, torch.device]) -> str:
    return LLMAnswerer(model=model, tokenizer=tokenizer, device=device).answer(question=question, context=context, use_context=use_context)


# ─────────────────────────────────────────────────────────────────────────────
# Private filesystem helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
