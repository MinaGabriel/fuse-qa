import re
import unicodedata
from datetime import datetime
import os
import json
import re
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Filename utilities
# ─────────────────────────────────────────────────────────────────────────────

def hf_model_to_filename(model_id: str, max_len: int = 120) -> str:
    s  = unicodedata.normalize("NFKD", model_id)
    s  = s.replace("/", "_").replace("\\", "_").replace(":", "_").replace("@", "_")
    s  = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s  = re.sub(r"_+", "_", s).strip("_")
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    s  = f"{s}_{ts}"
    if len(s) > max_len:
        s = s[:max_len]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    counts,
    em_hits,
    file_name,
    model_name,
    run_type,
    total_time,
    groups=("ALL", "LONG-TAIL", "INFREQUENT", "FREQUENT"),
):
    lines = []

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
        n  = counts.get(name, 0)
        em = safe_div(em_hits.get(name, 0), n)
        lines.append(f"  {name:<12} n={n:<6} EM={em:.4f}")

    report_file = file_name + ".report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(
        f"Saved report: {report_file} | "
        f"Time={total_time:.2f}s | "
        f"ALL={safe_div(em_hits['ALL'],           counts['ALL']):.4f} | "
        f"LONG-TAIL={safe_div(em_hits['LONG-TAIL'],   counts['LONG-TAIL']):.4f} | "
        f"INFREQUENT={safe_div(em_hits['INFREQUENT'], counts['INFREQUENT']):.4f} | "
        f"FREQUENT={safe_div(em_hits['FREQUENT'],     counts['FREQUENT']):.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_WS    = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    s = "" if s is None else str(s).lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_record(f, record):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    elif s_pop < 10_000:
        return "INFREQUENT"
    else:
        return "FREQUENT"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction post-processing
# ─────────────────────────────────────────────────────────────────────────────

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
        tmp   = s.replace(" and ", ", ")
        parts = [p.strip() for p in tmp.split(",") if p.strip()]
        if parts:
            s = parts[-1]

    s = s.strip()
    return s if s else "UNKNOWN"


def safe_div(a, b):
    return (a / b) if b else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM inference
# ─────────────────────────────────────────────────────────────────────────────

def ask_llm_generate(
    model,
    tokenizer,
    question: str,
    context:  str,
    use_context: bool,
    device,
) -> str:

    # Strong anti-reasoning prefix (model-agnostic)
    base_guard = (
        "You MUST output ONLY the final answer.\n"
        "DO NOT output reasoning.\n"
        "DO NOT output analysis.\n"
        "DO NOT use words like 'analysis', 'final', or explanations.\n"
        "Output ONLY a short phrase (1–5 words).\n"
    )

    if use_context:
        system = (
            base_guard
            + "You are a factual QA assistant.\n"
            + "Use ONLY the provided context.\n"
            + "If the answer is not explicitly stated in the context, return UNKNOWN.\n"
            + "Make sure when you answer you pick the correct context.\n"
            + "Don't return the subject name when answering the question.\n"
            + "Extract the answer from the correct context.\n"
            + "Return the most likely correct answer.\n"
            + "No explanations. No full sentences."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        system = (
            base_guard
            + "You are a factual QA assistant.\n"
            + "Answer using your parametric knowledge.\n"
            + "If you do not know, return UNKNOWN.\n"
            + "No explanations. No full sentences."
        )
        user = f"Question: {question}\nAnswer:"

    # Ensure pad token (Gemma fix)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build prompt (robust across ALL models)
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Gemma fallback: merge system into user
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{system}\n\n{user}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = f"{system}\n\n{user}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Hard-block reasoning tokens (critical for gpt-oss)
    bad_words_ids = tokenizer(
        ["analysis", "Analysis", "analysis:", "final:", "Final:"],
        add_special_tokens=False,
    ).input_ids

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6,                  # reduces reasoning chance
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bad_words_ids=bad_words_ids,       # key fix for gpt-oss
            use_cache=True,
        )

    gen = tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Final safety cleanup (cross-model)
    gen = gen.strip()
    if gen.lower().startswith("analysis"):
        gen = gen.split("\n")[-1].strip()

    return clean_pred(gen)