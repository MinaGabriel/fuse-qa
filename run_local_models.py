#!/usr/bin/env python3
"""Run retrieval QA experiments against a local vLLM-hosted endpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import requests
import tqdm
from datasets import load_dataset

from fuseqa.utils import (
    build_context,
    clean_pred,
    generate_report,
    hf_model_to_filename,
    norm,
    parse_list,
    safe_div,
    tier_from_spop,
    write_record,
)

GROUPS = ("ALL", "LONG-TAIL", "INFREQUENT", "FREQUENT")
RUN_TYPES = ("NO-CONTEXT", "FUSEQA", "FUSEQA-SRE", "PARAMETRIC")
GPT_OSS_NAME_TOKEN = "gpt-oss"
DEFAULT_DATASET = "auto"
DEFAULT_DATASET_BASE = "MinaGabriel/popqa-with-retrieval-20"
DEFAULT_DATASET_SRE = "MinaGabriel/popqa-retrieval20-with-sre"
DEFAULT_DATASET_ENTITY = "MinaGabriel/entityquestions-retrieval20-with-sre"
DEFAULT_SPLIT = "auto"
DEFAULT_BASE_RETR_COL = "retrieved_docs"
DEFAULT_SRE_RETR_COL = "retrieved_docs_sre"
DEFAULT_CLIP_CHARS = 250
DEFAULT_S_POP_COL = "auto"
DATASET_ALIASES = {
    "POPQA": DEFAULT_DATASET_BASE,
    "POPQASRE": DEFAULT_DATASET_SRE,
    "ENTITYQUESTIONS": DEFAULT_DATASET_ENTITY,
}
RUN_TYPE_ALIASES = {
    "NO-CONTEXT": "NO-CONTEXT",
    "PARAMETRIC": "NO-CONTEXT",
    "FUSEQA": "FUSEQA",
    "FUSEQA-SRE": "FUSEQA-SRE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run POPQA/EntityQuestions-style evaluations using local vLLM."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="vLLM server base URL (default: http://127.0.0.1:8001)",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help='Model id for /v1/chat/completions. Use "auto" to pick /v1/models[0].',
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            'Hugging Face dataset id. Use "auto" to choose by --run-type. '
            'Shorthand aliases: "POPQA", "POPQA-SRE", "EntityQuestions".'
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help='Dataset split to evaluate. Use "auto" to pick a split available in the dataset.',
    )
    parser.add_argument(
        "--run-type",
        default="FUSEQA",
        choices=RUN_TYPES,
        help="Notebook-compatible mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved docs to include as context when run-type uses context.",
    )
    parser.add_argument(
        "--base-retr-col",
        default=DEFAULT_BASE_RETR_COL,
        help="Column for base retrieved docs (default: retrieved_docs).",
    )
    parser.add_argument(
        "--sre-retr-col",
        default=DEFAULT_SRE_RETR_COL,
        help="Column for SRE-ranked retrieved docs (default: retrieved_docs_sre).",
    )
    parser.add_argument(
        "--sre-score-th",
        type=float,
        default=0.90,
        help="Minimum SRE score for docs used in FUSEQA-SRE (default: 0.90).",
    )
    parser.add_argument(
        "--clip-chars",
        type=int,
        default=DEFAULT_CLIP_CHARS,
        help="Max characters per retrieved chunk in context (default: 250).",
    )
    parser.add_argument(
        "--s-pop-col",
        default=DEFAULT_S_POP_COL,
        help='Popularity column for tiering. Use "auto" to prefer s_pop_avg, then s_pop.',
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Evaluate first N examples. 0 means full split.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="max_tokens for /v1/chat/completions.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=("low", "medium", "high"),
        help="Reasoning effort for GPT-OSS on vLLM (default: low).",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Request reasoning text in API response (off by default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per request on transient failures.",
    )
    parser.add_argument(
        "--write-outputs",
        action="store_true",
        help="Write per-example predictions to <run_name>.jsonl.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Override output file prefix. Default derives from model + run-type + timestamp.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=3,
        help="Print first N examples with question/context/pred/gold.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Update tqdm postfix every N examples.",
    )
    parser.add_argument(
        "--skip-model-list",
        action="store_true",
        help="Skip querying /v1/models (requires --model to be valid).",
    )
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def list_models(session: requests.Session, base_url_v1: str, timeout: float) -> list[str]:
    resp = session.get(f"{base_url_v1}/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def pick_model(model_arg: str, served_models: list[str]) -> str:
    if model_arg != "auto":
        return model_arg
    if not served_models:
        raise ValueError('No served models available. Pass --model "<model-id>".')
    return served_models[0]


def supports_reasoning_controls(model_id: str) -> bool:
    return GPT_OSS_NAME_TOKEN in (model_id or "").lower()


def canonical_run_type(run_type: str) -> str:
    key = (run_type or "").strip().upper()
    if key in RUN_TYPE_ALIASES:
        return RUN_TYPE_ALIASES[key]
    raise ValueError(f"Unsupported run type: {run_type}")


def resolve_dataset(dataset_arg: str, run_type: str) -> str:
    ds = (dataset_arg or "").strip()
    if ds and ds.lower() != "auto":
        alias_key = re.sub(r"[^A-Za-z0-9]+", "", ds).upper()
        return DATASET_ALIASES.get(alias_key, ds)
    if canonical_run_type(run_type) == "FUSEQA-SRE":
        return DEFAULT_DATASET_SRE
    return DEFAULT_DATASET_BASE


def resolve_split(split_arg: str, available_splits: list[str]) -> str:
    splits = [s for s in available_splits if s]
    if not splits:
        raise ValueError("Dataset has no available splits.")

    split = (split_arg or "").strip()
    if split and split.lower() != "auto":
        if split in splits:
            return split
        raise ValueError(
            f"Split '{split}' not found. Available splits: {', '.join(sorted(splits))}."
        )

    for preferred in ("test", "validation", "train"):
        if preferred in splits:
            return preferred
    return splits[0]


def resolve_s_pop_col(s_pop_col_arg: str, column_names: list[str]) -> str:
    col = (s_pop_col_arg or "").strip()
    if col and col.lower() != "auto":
        return col
    if "s_pop_avg" in column_names:
        return "s_pop_avg"
    if "s_pop" in column_names:
        return "s_pop"
    return "s_pop"


def parse_s_pop(ex: dict[str, Any], s_pop_col: str) -> int:
    val = ex.get(s_pop_col, 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def build_merged_prompt(question: str, context: str, use_context: bool) -> str:
    base_guard = (
        "You MUST output ONLY the final answer.\n"
        "DO NOT output reasoning.\n"
        "DO NOT output analysis.\n"
        "DO NOT use words like 'analysis', 'commentary', or 'final'.\n"
        "Output ONLY a short phrase (1-5 words).\n"
    )
    if use_context:
        prompt = (
            base_guard
            + "You are a factual QA assistant.\n"
            + "Use ONLY the provided context.\n"
            + "If the answer is not explicitly stated in the context, return UNKNOWN.\n"
            + "Do not return the subject name.\n"
            + "No explanations.\n\n"
            + f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    else:
        prompt = (
            base_guard
            + "You are a factual QA assistant.\n"
            + "Answer using your parametric knowledge.\n"
            + "If you do not know, return UNKNOWN.\n"
            + "No explanations.\n\n"
            + f"Question: {question}\nAnswer:"
        )
    return prompt


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text", "")))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    if content:
        return str(content)
    text = choices[0].get("text")
    return "" if text is None else str(text)


def postprocess_prediction(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "UNKNOWN"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        for ln in reversed(lines):
            if not re.search(
                r"\b(analysis|commentary|reasoning|because|therefore)\b",
                ln,
                flags=re.IGNORECASE,
            ):
                text = ln
                break
        else:
            text = lines[-1]

    text = re.sub(
        r"^(analysis|commentary|final|assistant)\b[:\-\s]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(analysis|commentary|final)(?=[A-Z])",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(the\s+)?(final\s+)?answer\s*(is|:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    pred = clean_pred(text)
    if norm(pred) in {"none", "null", "nil", "n a", "na"}:
        return "UNKNOWN"
    return pred


def build_context_from_sre_list(
    sre_list: Any, score_th: float, k: int, clip_chars: int = 250
) -> str:
    if not isinstance(sre_list, list):
        return ""
    passed: list[str] = []
    for d in sre_list:
        if not isinstance(d, dict):
            continue
        txt = (d.get("text") or "").strip()
        if not txt:
            continue
        try:
            score = float(d.get("score", -1.0))
        except (TypeError, ValueError):
            score = -1.0
        if score >= score_th:
            passed.append(txt[:clip_chars])
        if len(passed) >= k:
            break
    return "\n\n".join(passed)


def get_context_for_run(
    ex: dict[str, Any],
    run_type: str,
    use_context: bool,
    top_k: int,
    base_retr_col: str,
    sre_retr_col: str,
    sre_score_th: float,
    clip_chars: int,
) -> str:
    if not use_context or top_k <= 0:
        return ""

    rt = (run_type or "").upper()
    if "SRE" in rt:
        sre_list = ex.get(sre_retr_col) or []
        context = build_context_from_sre_list(
            sre_list,
            score_th=sre_score_th,
            k=top_k,
            clip_chars=max(clip_chars, 1),
        )
        if context.strip():
            return context

    base_docs = ex.get(base_retr_col) or []
    return build_context(base_docs, k=top_k, clip_chars=max(clip_chars, 1))


@dataclass
class VLLMClient:
    session: requests.Session
    base_url_v1: str
    model: str
    timeout: float
    retries: int
    reasoning_effort: str
    include_reasoning: bool
    use_reasoning_controls: bool

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.use_reasoning_controls:
            payload["reasoning_effort"] = self.reasoning_effort
            payload["include_reasoning"] = self.include_reasoning
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.post(
                    f"{self.base_url_v1}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return _extract_content(data), (data.get("usage") or {})
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(1.0 * (attempt + 1), 3.0))
        raise RuntimeError(f"Request failed after retries: {last_error}") from last_error


def run_popqa_eval(
    client: VLLMClient,
    ds: Any,
    run_type: str,
    top_k: int,
    base_retr_col: str,
    sre_retr_col: str,
    sre_score_th: float,
    clip_chars: int,
    s_pop_col: str,
    write_outputs: bool,
    run_name: str,
    max_tokens: int,
    temperature: float,
    print_samples: int,
    log_every: int,
) -> tuple[dict[str, int], dict[str, int], str, float, dict[str, int], bool]:
    run_type = canonical_run_type(run_type)
    use_context = run_type in ("FUSEQA", "FUSEQA-SRE")
    counts = {g: 0 for g in GROUPS}
    em_hits = {g: 0 for g in GROUPS}
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "empty_predictions": 0,
    }

    output_name = run_name or hf_model_to_filename(f"{client.model}-{run_type}")
    outfile = f"{output_name}.jsonl"

    start_time = time.time()
    interrupted = False
    writer_ctx = open(outfile, "w", encoding="utf-8", buffering=1) if write_outputs else nullcontext()
    pbar = tqdm.tqdm(total=len(ds), desc="Generating + Evaluating", dynamic_ncols=True)

    def update_metrics(tier: str, em: int) -> None:
        counts["ALL"] += 1
        em_hits["ALL"] += em
        if tier in counts:
            counts[tier] += 1
            em_hits[tier] += em

    def current_scores() -> dict[str, float]:
        return {
            "ALL_EM": safe_div(em_hits["ALL"], counts["ALL"]),
            "Long_Tail": safe_div(em_hits["LONG-TAIL"], counts["LONG-TAIL"]),
            "Infrequent": safe_div(em_hits["INFREQUENT"], counts["INFREQUENT"]),
            "Frequent": safe_div(em_hits["FREQUENT"], counts["FREQUENT"]),
        }

    try:
        with writer_ctx as writer:
            for i in range(len(ds)):
                ex = {k: ds[k][i] for k in ds.column_names}
                q = ex["question"]
                s_pop = parse_s_pop(ex, s_pop_col)
                tier = tier_from_spop(s_pop)

                gold = parse_list(ex.get("possible_answers"))
                gold_norm_set = {norm(g) for g in gold if norm(g)}

                context = get_context_for_run(
                    ex=ex,
                    run_type=run_type,
                    use_context=use_context,
                    top_k=top_k,
                    base_retr_col=base_retr_col,
                    sre_retr_col=sre_retr_col,
                    sre_score_th=sre_score_th,
                    clip_chars=clip_chars,
                )
                prompt = build_merged_prompt(q, context, use_context=use_context)

                raw_pred, usage = client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if not (raw_pred or "").strip():
                    token_totals["empty_predictions"] += 1
                pred = postprocess_prediction(raw_pred)
                pred_norm = norm(pred)
                em = int(pred_norm in gold_norm_set) if gold_norm_set else 0
                update_metrics(tier, em)

                token_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                token_totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                token_totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

                if i < max(print_samples, 0):
                    print(f"\nQ: {q}\nContext: {context}\nPred: {pred}\nGold: {gold}\n")

                if write_outputs:
                    write_record(
                        writer,
                        {
                            "i": i,
                            "s_pop": s_pop,
                            "tier": tier,
                            "question": q,
                            "gold": gold,
                            "pred": pred,
                            "raw_pred": raw_pred,
                            "em": em,
                        },
                    )

                pbar.update(1)
                if log_every > 0 and i % log_every == 0:
                    pbar.set_postfix({k: f"{v:.4f}" for k, v in current_scores().items()})
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user; generating report from completed examples.")
    finally:
        pbar.close()

    total_time = time.time() - start_time
    return counts, em_hits, output_name, total_time, token_totals, interrupted


def main() -> int:
    args = parse_args()
    run_type = canonical_run_type(args.run_type)
    base_url_v1 = normalize_base_url(args.base_url)
    print(f"Requested model: {args.model}")
    print(f"Using endpoint: {base_url_v1}")
    print(f"Run type: {run_type} | Use context: {run_type in ('FUSEQA', 'FUSEQA-SRE')}")
    print(
        "Retrieval setup:"
        f" base_col={args.base_retr_col},"
        f" sre_col={args.sre_retr_col},"
        f" sre_score_th={args.sre_score_th},"
        f" clip_chars={max(args.clip_chars, 1)}"
    )
    print(f"Generation: max_tokens={args.max_tokens}, temperature={args.temperature}")

    session = requests.Session()
    served_models: list[str] = []
    if not args.skip_model_list:
        try:
            served_models = list_models(session, base_url_v1, args.timeout)
            print("Served models:")
            for m in served_models:
                print(f"  - {m}")
        except requests.RequestException as exc:
            if args.model != "auto":
                print(
                    "Warning: failed to list models; continuing because --model was provided.\n"
                    f"Details: {exc}",
                    file=sys.stderr,
                )
            else:
                print(f"Failed to list models: {exc}", file=sys.stderr)
                return 1

    try:
        model = pick_model(args.model, served_models)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f'Model: "{model}"')
    use_reasoning_controls = supports_reasoning_controls(model)
    if use_reasoning_controls:
        print(
            "Model supports reasoning controls:"
            f" reasoning_effort={args.reasoning_effort}, include_reasoning={args.include_reasoning}"
        )
    else:
        if args.include_reasoning or args.reasoning_effort != "low":
            print(
                "Note: --reasoning-effort/--include-reasoning are ignored for non-GPT-OSS models."
            )
        print("Model does not use reasoning controls; sending standard chat-completions payload.")

    dataset_name = resolve_dataset(args.dataset, run_type)
    ds_all = load_dataset(dataset_name)
    split_name = resolve_split(args.split, list(ds_all.keys()))
    print(f"Loading dataset: {dataset_name} [{split_name}]")
    ds = ds_all[split_name]
    if args.max_examples > 0:
        n = min(args.max_examples, len(ds))
        ds = ds.select(range(n))
    if run_type == "FUSEQA-SRE" and args.sre_retr_col not in ds.column_names:
        print(
            f"Error: SRE column '{args.sre_retr_col}' not found in dataset '{dataset_name}'.",
            file=sys.stderr,
        )
        print(
            "For FUSEQA-SRE, use a dataset with SRE scores, e.g. "
            f"'{DEFAULT_DATASET_SRE}'.",
            file=sys.stderr,
        )
        return 1
    if run_type in ("FUSEQA", "FUSEQA-SRE") and args.base_retr_col not in ds.column_names:
        print(
            f"Warning: base retrieval column '{args.base_retr_col}' not found; "
            "context may be empty."
        )
    s_pop_col = resolve_s_pop_col(args.s_pop_col, list(ds.column_names))
    if s_pop_col not in ds.column_names:
        print(
            f"Warning: popularity column '{s_pop_col}' not found; tiering will default to 0."
        )
    print(f"Tier popularity column: {s_pop_col}")
    print(f"Examples to evaluate: {len(ds)}")

    client = VLLMClient(
        session=session,
        base_url_v1=base_url_v1,
        model=model,
        timeout=args.timeout,
        retries=max(args.retries, 0),
        reasoning_effort=args.reasoning_effort,
        include_reasoning=args.include_reasoning,
        use_reasoning_controls=use_reasoning_controls,
    )

    counts, em_hits, run_name, total_time, token_totals, interrupted = run_popqa_eval(
        client=client,
        ds=ds,
        run_type=run_type,
        top_k=max(args.top_k, 0),
        base_retr_col=args.base_retr_col,
        sre_retr_col=args.sre_retr_col,
        sre_score_th=args.sre_score_th,
        clip_chars=max(args.clip_chars, 1),
        s_pop_col=s_pop_col,
        write_outputs=args.write_outputs,
        run_name=args.run_name.strip(),
        max_tokens=max(args.max_tokens, 1),
        temperature=args.temperature,
        print_samples=max(args.print_samples, 0),
        log_every=args.log_every,
    )

    generate_report(
        counts=counts,
        em_hits=em_hits,
        file_name=run_name,
        model_name=model,
        run_type=run_type,
        total_time=total_time,
    )

    print("\nToken usage totals:")
    print(json.dumps(token_totals, indent=2))

    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
