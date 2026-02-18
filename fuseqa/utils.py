import re
import unicodedata
from datetime import datetime
import os
import json
import re
import torch

def hf_model_to_filename(model_id: str, max_len: int = 120) -> str:
    s = unicodedata.normalize("NFKD", model_id)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_").replace("@", "_")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    ts = datetime.now().strftime("%Y%m%d-%H%M")

    s = f"{s}_{ts}"

    if len(s) > max_len:
        s = s[:max_len]

    return s

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

def norm(s: str) -> str:
    s = "" if s is None else str(s).lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def parse_list(x):
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

def build_context(docs, k=10):
    parts = []
    for d in (docs or [])[:k]:
        txt = d.get("text", "") if isinstance(d, dict) else str(d)
        txt = (txt or "").strip()
        if txt:
            parts.append(txt[:250])
    return "\n\n".join(parts)

def tier_from_spop(s_pop: int) -> str:
    if s_pop < 100:
        return "LONG-TAIL"
    elif s_pop < 10000:
        return "INFREQUENT"
    else:
        return "FREQUENT"

def clean_pred(s: str) -> str:
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
    if "," in s or " and " in s.lower():
        tmp = s.replace(" and ", ", ")
        parts = [p.strip() for p in tmp.split(",") if p.strip()]
        if parts:
            s = parts[-1]
    s = s.strip()
    return s if s else "UNKNOWN"

def safe_div(a, b):
    return (a / b) if b else 0.0

def ask_llm_generate(model, tokenizer, question: str, context: str, use_context: bool, llama_device: str) -> str:
    if use_context:
        system = (
            "You are a factual QA assistant.\n"
            "Use ONLY the provided context.\n"
            "Return ONLY the answer as a short phrase (1–5 words).\n"
            "If the answer is not explicitly stated in the context, return UNKNOWN.\n"
            "Make sure when you answer you pick the correct context. \n"
            "Dont return the subject name when answering the question. \n"
            "You have to return an answer extract the answer from the given context, but first find the correct context corresponding to the answer.\n"
            "return the most popular answer you think is the correct answer.\n"
            "No explanations. No full sentences."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        system = (
            "You are a factual QA assistant.\n"
            "Answer using your parametric knowledge.\n"
            "Return ONLY the answer as a short phrase (1–5 words).\n"
            "If you do not know, return UNKNOWN.\n"
            "No explanations. No full sentences."
        )
        user = f"Question: {question}\nAnswer:"

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role":"system","content":system},{"role":"user","content":user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = system + "\n\n" + user

    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(llama_device) for k, v in inputs.items()}  # Explicit device
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_pred(gen)