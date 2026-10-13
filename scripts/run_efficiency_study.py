#!/usr/bin/env python3
"""Run a lightweight efficiency study for FUSE-QA vs FUSE-QA+DistillSRE.

This script intentionally reuses existing project helpers:
- vLLM OpenAI-compatible generation client and prompt/postprocessing from run_local_models.py
- text normalization and EM helpers from fuseqa.utils
- optional live DistillSRE scorer from fuseqa.SRE

Timing note:
- By default, retrieval/fusion/reranking are read from precomputed dataset columns.
  Their stage timings are therefore recorded as 0.0 placeholders.
- DistillSRE filtering time is measurable in live mode (--distillsre-mode live).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
import tqdm
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fuseqa.utils import norm, parse_list, tier_from_spop, write_record
from run_local_models import (
    DEFAULT_DATASET_BASE,
    DEFAULT_DATASET_SRE,
    DEFAULT_SPLIT,
    build_merged_prompt,
    list_models,
    normalize_base_url,
    parse_s_pop,
    postprocess_prediction,
    resolve_split,
    supports_reasoning_controls,
)

METHOD_FUSE_QA = "fuse_qa"
METHOD_FUSE_QA_DISTILLSRE = "fuse_qa_distillsre"
METHOD_BOTH = "both"

DEFAULT_OUTPUT_DIR = "results/efficiency_study"
DEFAULT_BASE_RETR_COL = "retrieved_docs"
DEFAULT_SRE_RETR_COL = "retrieved_docs_sre"
DEFAULT_DATASET = "auto"

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")


class StudyVLLMClient:
    """Small OpenAI-compatible vLLM wrapper with optional served-model fallback."""

    def __init__(
        self,
        session: requests.Session,
        base_url_v1: str,
        model: str,
        timeout: float,
        retries: int,
        reasoning_effort: str,
        include_reasoning: bool,
        use_reasoning_controls: bool,
        allow_model_fallback: bool = True,
    ):
        self.session = session
        self.base_url_v1 = base_url_v1.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = max(retries, 0)
        self.reasoning_effort = reasoning_effort
        self.include_reasoning = include_reasoning
        self.use_reasoning_controls = use_reasoning_controls
        self.allow_model_fallback = allow_model_fallback

    def _extract_content(self, data: dict[str, Any]) -> str:
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

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        try:
            data = resp.json()
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    return str(err.get("message", ""))
                return str(data.get("message", ""))
        except Exception:
            pass
        return resp.text

    @staticmethod
    def _is_missing_model_error(resp: requests.Response) -> bool:
        if resp.status_code != 404:
            return False
        msg = StudyVLLMClient._error_message(resp).lower()
        return "model" in msg and ("does not exist" in msg or "not found" in msg)

    def _pick_fallback_model(self) -> Optional[str]:
        try:
            models = list_models(self.session, self.base_url_v1, self.timeout)
        except requests.RequestException:
            return None
        if not models:
            return None
        fallback = models[0]
        return fallback if fallback != self.model else None

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, Any]]:
        last_error: Exception | None = None
        model_for_attempt = self.model
        did_fallback = False

        for attempt in range(self.retries + 1):
            payload: dict[str, Any] = {
                "model": model_for_attempt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if self.use_reasoning_controls:
                payload["reasoning_effort"] = self.reasoning_effort
                payload["include_reasoning"] = self.include_reasoning

            try:
                resp = self.session.post(
                    f"{self.base_url_v1}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                if (
                    self.allow_model_fallback
                    and not did_fallback
                    and self._is_missing_model_error(resp)
                ):
                    fallback = self._pick_fallback_model()
                    if fallback:
                        did_fallback = True
                        model_for_attempt = fallback
                        self.model = fallback
                        continue

                resp.raise_for_status()
                data = resp.json()
                return self._extract_content(data), (data.get("usage") or {})
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(1.0 * (attempt + 1), 3.0))

        raise RuntimeError(f"Request failed after retries: {last_error}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight efficiency study for FUSE-QA and FUSE-QA + DistillSRE."
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET, help='Dataset id, or "auto".')
    parser.add_argument("--split", default=DEFAULT_SPLIT, help='Split name, or "auto".')
    parser.add_argument(
        "--method",
        choices=(METHOD_FUSE_QA, METHOD_FUSE_QA_DISTILLSRE, METHOD_BOTH),
        default=METHOD_BOTH,
        help="Which method(s) to run.",
    )
    parser.add_argument(
        "--sample_strategy",
        choices=("random", "stratified"),
        default="random",
        help="Sampling strategy when --max_examples is set.",
    )
    parser.add_argument("--max_examples", type=int, default=0, help="0 = full split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Where outputs are saved.")

    parser.add_argument("--base_retr_col", default=DEFAULT_BASE_RETR_COL)
    parser.add_argument("--sre_retr_col", default=DEFAULT_SRE_RETR_COL)
    parser.add_argument(
        "--s_pop_col",
        default="auto",
        help='Popularity column for tier bucket. "auto" tries s_pop_avg, then s_pop.',
    )
    parser.add_argument(
        "--bucket_col",
        default="bucket",
        help="Existing bucket column name if present; otherwise bucket is derived from s_pop.",
    )

    parser.add_argument("--top_k_passages", type=int, default=3)
    parser.add_argument("--clip_chars", type=int, default=250)

    parser.add_argument(
        "--distillsre_mode",
        choices=("live", "precomputed"),
        default="live",
        help="live: run DistillSRE on the fly. precomputed: use retrieved_docs_sre from dataset.",
    )
    parser.add_argument("--max_selected_sentences", type=int, default=5)
    parser.add_argument("--distillsre_score_th", type=float, default=0.0)
    parser.add_argument("--sentence_min_words", type=int, default=4)
    parser.add_argument("--sentence_max_words", type=int, default=40)
    parser.add_argument(
        "--fallback_to_base_context",
        action="store_true",
        help="If DistillSRE returns no sentences, fall back to passage context.",
    )

    parser.add_argument("--distillsre_device", default="cuda:0")
    parser.add_argument("--distillsre_base_model_name", default="google/gemma-2-2b-it")
    parser.add_argument(
        "--distillsre_lora_dir",
        default="MinaGabriel/gemma-2-2b-it-SRE-LoRA",
    )
    parser.add_argument("--distillsre_batch_size", type=int, default=16)
    parser.add_argument("--distillsre_max_length", type=int, default=1024)

    parser.add_argument(
        "--vllm_base_url",
        default="http://127.0.0.1:8001",
        help="vLLM base URL (OpenAI-compatible).",
    )
    parser.add_argument(
        "--vllm_model_name",
        default="llama-3.1-8b-instruct",
        help="Served model name.",
    )
    parser.add_argument("--skip_model_list", action="store_true")
    parser.add_argument(
        "--disable_model_fallback",
        action="store_true",
        help="Disable one-time fallback to /v1/models[0] on model-not-found errors.",
    )
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--reasoning_effort",
        choices=("low", "medium", "high"),
        default="low",
        help="Used only for GPT-OSS model names.",
    )
    parser.add_argument("--include_reasoning", action="store_true")

    parser.add_argument(
        "--context_tokenizer",
        default="",
        help=(
            "Optional tokenizer name/path to compute context_num_tokens. "
            "If empty, context_num_tokens will be null."
        ),
    )

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--omit_context_text",
        action="store_true",
        help="Do not store full context text in per-example JSONL.",
    )

    return parser.parse_args()


def resolve_dataset_for_study(dataset_arg: str, method: str) -> str:
    ds = (dataset_arg or "").strip()
    if ds and ds.lower() != "auto":
        return ds
    if method in (METHOD_FUSE_QA_DISTILLSRE, METHOD_BOTH):
        return DEFAULT_DATASET_SRE
    return DEFAULT_DATASET_BASE


def resolve_s_pop_col_for_study(s_pop_col_arg: str, column_names: list[str]) -> str:
    if s_pop_col_arg and s_pop_col_arg.lower() != "auto":
        return s_pop_col_arg
    if "s_pop_avg" in column_names:
        return "s_pop_avg"
    if "s_pop" in column_names:
        return "s_pop"
    return "s_pop"


def resolve_bucket(ex: dict[str, Any], bucket_col: str, s_pop_col: str) -> str:
    if bucket_col and bucket_col in ex and str(ex.get(bucket_col, "")).strip():
        return str(ex[bucket_col])
    if s_pop_col in ex:
        return tier_from_spop(parse_s_pop(ex, s_pop_col))
    if "s_pop" in ex:
        return tier_from_spop(parse_s_pop(ex, "s_pop"))
    if "s_pop_avg" in ex:
        return tier_from_spop(parse_s_pop(ex, "s_pop_avg"))
    return "UNKNOWN"


def extract_question_id(ex: dict[str, Any], fallback_idx: int) -> str:
    for key in ("question_id", "id", "qid"):
        if key in ex:
            return str(ex[key])
    return str(fallback_idx)


def parse_doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        for key in ("text", "contents", "content"):
            value = doc.get(key)
            if value is not None:
                return str(value)
        return ""
    return str(doc or "")


def clean_doc_for_sentences(doc: str) -> str:
    if not doc:
        return ""
    text = doc
    text = re.sub(r'^"\s*[^"]+\s*"\s*', "", text)
    text = re.sub(r"^[A-Z][^\n]{0,120}\n", "", text)
    text = text.replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def select_base_passages(ex: dict[str, Any], base_retr_col: str, top_k: int, clip_chars: int) -> tuple[list[str], int]:
    raw_docs = ex.get(base_retr_col) or []
    if not isinstance(raw_docs, list):
        raw_docs = []
    passages: list[str] = []
    for doc in raw_docs[: max(top_k, 0)]:
        text = parse_doc_text(doc).strip()
        if not text:
            continue
        passages.append(text[: max(clip_chars, 1)])
    return passages, len(raw_docs)


def split_passages_to_sentences(
    passages: list[str],
    min_words: int,
    max_words: int,
) -> list[str]:
    out: list[str] = []
    for passage in passages:
        clean = clean_doc_for_sentences(passage)
        if not clean:
            continue
        parts = SENT_SPLIT_RE.split(clean)
        if not parts:
            parts = [clean]
        for sentence in parts:
            s = sentence.strip().rstrip(" .")
            words = s.split()
            if len(words) < max(min_words, 0):
                continue
            if max_words > 0 and len(words) > max_words:
                s = " ".join(words[:max_words])
            if s:
                out.append(s)

    deduped: list[str] = []
    seen: set[str] = set()
    for s in out:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def select_precomputed_sre_sentences(
    ex: dict[str, Any],
    sre_retr_col: str,
    max_selected_sentences: int,
    score_th: float,
) -> list[dict[str, Any]]:
    rows = ex.get(sre_retr_col) or []
    if not isinstance(rows, list):
        return []

    selected: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            text = str(row.get("text") or "").strip()
            try:
                score = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
        else:
            text = str(row or "").strip()
            score = 1.0

        if not text:
            continue
        if score < score_th:
            continue

        selected.append({"sentence": text, "score": score})
        if 0 < max_selected_sentences <= len(selected):
            break
    return selected


class OptionalContextTokenizer:
    def __init__(self, tokenizer_name_or_path: str):
        self.enabled = False
        self.tokenizer = None
        name = (tokenizer_name_or_path or "").strip()
        if not name:
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.enabled = True

    def count(self, text: str) -> Optional[int]:
        if not self.enabled or self.tokenizer is None:
            return None
        encoded = self.tokenizer(text, add_special_tokens=False)
        return len(encoded.get("input_ids", []))


class DistillSREFilter:
    def __init__(
        self,
        device: str,
        base_model_name: str,
        lora_dir: str,
        batch_size: int,
        max_length: int,
    ):
        from fuseqa import SRE

        self.model = SRE(
            classifier_device=device,
            base_model_name=base_model_name,
            lora_dir=lora_dir,
        )
        self.batch_size = max(batch_size, 1)
        self.max_length = max(max_length, 64)

    def rank_sentences(
        self,
        question: str,
        full_context: str,
        sentences: list[str],
    ) -> list[dict[str, Any]]:
        if not sentences:
            return []
        return self.model.score_sentences_batch(
            question=question,
            full_context=full_context,
            sentences=sentences,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )


def safe_mean(values: list[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def choose_indices(
    ds: Any,
    max_examples: int,
    sample_strategy: str,
    seed: int,
    bucket_col: str,
    s_pop_col: str,
) -> list[int]:
    n = len(ds)
    all_indices = list(range(n))
    if max_examples <= 0 or max_examples >= n:
        return all_indices

    rng = random.Random(seed)

    if sample_strategy == "random":
        rng.shuffle(all_indices)
        return all_indices[:max_examples]

    # Fast stratified path:
    # Only touch bucket/s_pop columns to avoid materializing full rows (retrieved docs are large).
    bucket_values = ds[bucket_col] if bucket_col in ds.column_names else None
    s_pop_values = ds[s_pop_col] if s_pop_col in ds.column_names else None

    by_bucket: dict[str, list[int]] = defaultdict(list)
    for idx in all_indices:
        bucket = "UNKNOWN"
        if bucket_values is not None:
            val = bucket_values[idx]
            if str(val).strip():
                bucket = str(val)
        elif s_pop_values is not None:
            try:
                s_pop = int(s_pop_values[idx])
            except (TypeError, ValueError):
                s_pop = 0
            bucket = tier_from_spop(s_pop)
        by_bucket[bucket].append(idx)

    buckets = sorted(by_bucket.keys())
    if not buckets:
        rng.shuffle(all_indices)
        return all_indices[:max_examples]

    for b in buckets:
        rng.shuffle(by_bucket[b])

    per_bucket = max_examples // len(buckets)
    remainder = max_examples % len(buckets)

    selected: list[int] = []
    leftovers: list[int] = []
    for i, b in enumerate(buckets):
        target = per_bucket + (1 if i < remainder else 0)
        take = min(target, len(by_bucket[b]))
        selected.extend(by_bucket[b][:take])
        leftovers.extend(by_bucket[b][take:])

    if len(selected) < max_examples:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_examples - len(selected)])

    rng.shuffle(selected)
    return selected[:max_examples]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "em",
        "context_num_chars",
        "context_num_words",
        "context_num_tokens",
        "num_selected_sentences",
        "num_sentences_before_filtering",
        "num_sentences_after_filtering",
        "retrieval_time_sec",
        "fusion_time_sec",
        "rerank_time_sec",
        "sentence_split_time_sec",
        "filter_time_sec",
        "generation_time_sec",
        "postprocess_time_sec",
        "total_measured_time_sec",
    ]

    summary: dict[str, Any] = {"num_examples": len(records)}
    for key in metric_keys:
        values = [r.get(key) for r in records]
        summary[f"avg_{key}"] = safe_mean(values)
    return summary


def build_summary_payload(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    overall = summarize_records(records)

    by_method: dict[str, Any] = {}
    grouped_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped_method[str(r.get("method", "unknown"))].append(r)

    for method, rows in grouped_method.items():
        by_method[method] = summarize_records(rows)

    by_method_bucket: dict[str, Any] = {}
    for method, rows in grouped_method.items():
        bucket_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            bucket_groups[str(r.get("bucket", "UNKNOWN"))].append(r)
        by_method_bucket[method] = {
            bucket: summarize_records(bucket_rows)
            for bucket, bucket_rows in sorted(bucket_groups.items())
        }

    return {
        "config": config,
        "overall": overall,
        "by_method": by_method,
        "by_method_bucket": by_method_bucket,
        "timing_caveats": [
            "retrieval/fusion/rerank are offline precomputed in dataset columns; "
            "their stage timings are placeholder 0.0 in this runner.",
            "generation timings include network/queueing/server effects from vLLM.",
            "token usage and timings may vary with vLLM async scheduling, batching, and prefix caching.",
        ],
    }


def write_table_files(summary_payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    method_rows = summary_payload.get("by_method", {})
    columns = [
        ("method", "Method"),
        ("avg_em", "EM"),
        ("avg_context_num_chars", "Avg. context chars"),
        ("avg_context_num_tokens", "Avg. context tokens"),
        ("avg_num_selected_sentences", "Avg. selected sentences"),
        ("avg_filter_time_sec", "Filter time (s)"),
        ("avg_generation_time_sec", "Generation time (s)"),
        ("avg_total_measured_time_sec", "Total measured time (s)"),
    ]

    display_name = {
        METHOD_FUSE_QA: "FUSE-QA",
        METHOD_FUSE_QA_DISTILLSRE: "FUSE-QA + DistillSRE",
    }

    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in columns])
        for method in (METHOD_FUSE_QA, METHOD_FUSE_QA_DISTILLSRE):
            if method not in method_rows:
                continue
            row = method_rows[method]
            writer.writerow(
                [
                    display_name.get(method, method),
                    row.get("avg_em"),
                    row.get("avg_context_num_chars"),
                    row.get("avg_context_num_tokens"),
                    row.get("avg_num_selected_sentences"),
                    row.get("avg_filter_time_sec"),
                    row.get("avg_generation_time_sec"),
                    row.get("avg_total_measured_time_sec"),
                ]
            )

    md_path = output_dir / "summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        headers = [label for _, label in columns]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for method in (METHOD_FUSE_QA, METHOD_FUSE_QA_DISTILLSRE):
            if method not in method_rows:
                continue
            row = method_rows[method]
            vals = [
                display_name.get(method, method),
                row.get("avg_em"),
                row.get("avg_context_num_chars"),
                row.get("avg_context_num_tokens"),
                row.get("avg_num_selected_sentences"),
                row.get("avg_filter_time_sec"),
                row.get("avg_generation_time_sec"),
                row.get("avg_total_measured_time_sec"),
            ]
            rendered = []
            for v in vals:
                if isinstance(v, float):
                    rendered.append(f"{v:.6f}")
                elif v is None:
                    rendered.append("null")
                else:
                    rendered.append(str(v))
            f.write("| " + " | ".join(rendered) + " |\n")

    return csv_path, md_path


def main() -> int:
    args = parse_args()

    print("Loading dataset metadata...")
    dataset_name = resolve_dataset_for_study(args.dataset, args.method)
    ds_all = load_dataset(dataset_name)
    split_name = resolve_split(args.split, list(ds_all.keys()))
    ds = ds_all[split_name]
    print(f"Loaded dataset: {dataset_name} [{split_name}] | rows={len(ds)}")

    s_pop_col = resolve_s_pop_col_for_study(args.s_pop_col, list(ds.column_names))
    if args.method in (METHOD_FUSE_QA, METHOD_BOTH) and args.base_retr_col not in ds.column_names:
        raise ValueError(
            f"Base retrieval column '{args.base_retr_col}' not found in dataset '{dataset_name}'."
        )
    if (
        args.method in (METHOD_FUSE_QA_DISTILLSRE, METHOD_BOTH)
        and args.distillsre_mode == "precomputed"
        and args.sre_retr_col not in ds.column_names
    ):
        raise ValueError(
            f"SRE column '{args.sre_retr_col}' not found for precomputed mode in dataset '{dataset_name}'."
        )

    print(
        f"Sampling indices: strategy={args.sample_strategy}, "
        f"max_examples={max(args.max_examples, 0)}, seed={args.seed}"
    )
    selected_indices = choose_indices(
        ds=ds,
        max_examples=max(args.max_examples, 0),
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        bucket_col=args.bucket_col,
        s_pop_col=s_pop_col,
    )
    print(f"Sampled rows: {len(selected_indices)}")

    base_url_v1 = normalize_base_url(args.vllm_base_url)
    session = requests.Session()
    served_models: list[str] = []
    if not args.skip_model_list:
        try:
            served_models = list_models(session, base_url_v1, timeout=args.timeout)
            if served_models and args.vllm_model_name not in served_models:
                print(
                    "Warning: requested model not found in /models list; continuing anyway. "
                    f"requested={args.vllm_model_name}, served={served_models}"
                )
        except requests.RequestException as exc:
            print(f"Warning: failed to query /models ({exc}); continuing with provided model name.")

    use_reasoning_controls = supports_reasoning_controls(args.vllm_model_name)
    if not use_reasoning_controls and (args.include_reasoning or args.reasoning_effort != "low"):
        print("Note: reasoning controls are ignored for non-GPT-OSS model names.")

    client = StudyVLLMClient(
        session=session,
        base_url_v1=base_url_v1,
        model=args.vllm_model_name,
        timeout=args.timeout,
        retries=max(args.retries, 0),
        reasoning_effort=args.reasoning_effort,
        include_reasoning=args.include_reasoning,
        use_reasoning_controls=use_reasoning_controls,
        allow_model_fallback=not args.disable_model_fallback,
    )

    context_tokenizer = OptionalContextTokenizer(args.context_tokenizer)

    methods: list[str]
    if args.method == METHOD_BOTH:
        methods = [METHOD_FUSE_QA, METHOD_FUSE_QA_DISTILLSRE]
    else:
        methods = [args.method]

    distillsre_filter: Optional[DistillSREFilter] = None
    if METHOD_FUSE_QA_DISTILLSRE in methods and args.distillsre_mode == "live":
        print(
            "Initializing DistillSRE (live mode) "
            f"on device {args.distillsre_device} ..."
        )
        distillsre_filter = DistillSREFilter(
            device=args.distillsre_device,
            base_model_name=args.distillsre_base_model_name,
            lora_dir=args.distillsre_lora_dir,
            batch_size=args.distillsre_batch_size,
            max_length=args.distillsre_max_length,
        )
        print("DistillSRE initialization complete.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir).resolve() / f"efficiency_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_example_path = output_dir / "per_example_metrics.jsonl"

    print(f"Dataset: {dataset_name} [{split_name}] | rows={len(ds)}")
    print(f"Evaluating sampled rows: {len(selected_indices)}")
    print(f"Methods: {methods}")
    print(f"Output dir: {output_dir}")

    records: list[dict[str, Any]] = []

    with open(per_example_path, "w", encoding="utf-8", buffering=1) as writer:
        for method in methods:
            pbar = tqdm.tqdm(
                selected_indices,
                total=len(selected_indices),
                desc=f"{method}",
                dynamic_ncols=True,
            )

            for step_i, idx in enumerate(pbar):
                ex = {k: ds[k][idx] for k in ds.column_names}

                total_start = time.perf_counter()

                question = str(ex.get("question") or "").strip()
                question_id = extract_question_id(ex, fallback_idx=idx)
                bucket = resolve_bucket(ex, bucket_col=args.bucket_col, s_pop_col=s_pop_col)
                gold_answers = parse_list(ex.get("possible_answers"))
                gold_norm_set = {norm(g) for g in gold_answers if norm(g)}

                retrieval_time_sec = 0.0
                fusion_time_sec = 0.0
                rerank_time_sec = 0.0

                stage_start = time.perf_counter()
                base_passages, num_reranked_candidates = select_base_passages(
                    ex=ex,
                    base_retr_col=args.base_retr_col,
                    top_k=args.top_k_passages,
                    clip_chars=args.clip_chars,
                )
                retrieval_time_sec += time.perf_counter() - stage_start

                sentence_split_time_sec = 0.0
                filter_time_sec = 0.0
                num_sentences_before_filtering = 0
                num_sentences_after_filtering = 0
                num_selected_sentences = 0

                if method == METHOD_FUSE_QA:
                    split_start = time.perf_counter()
                    sentences_before = split_passages_to_sentences(
                        base_passages,
                        min_words=args.sentence_min_words,
                        max_words=args.sentence_max_words,
                    )
                    sentence_split_time_sec += time.perf_counter() - split_start
                    num_sentences_before_filtering = len(sentences_before)
                    num_sentences_after_filtering = len(sentences_before)
                    num_selected_sentences = len(sentences_before)
                    context_text = "\n\n".join(base_passages)
                else:
                    split_start = time.perf_counter()
                    sentences_before = split_passages_to_sentences(
                        base_passages,
                        min_words=args.sentence_min_words,
                        max_words=args.sentence_max_words,
                    )
                    sentence_split_time_sec += time.perf_counter() - split_start
                    num_sentences_before_filtering = len(sentences_before)

                    selected_sentences: list[str] = []
                    if args.distillsre_mode == "precomputed":
                        precomputed_rows = select_precomputed_sre_sentences(
                            ex=ex,
                            sre_retr_col=args.sre_retr_col,
                            max_selected_sentences=args.max_selected_sentences,
                            score_th=args.distillsre_score_th,
                        )
                        selected_sentences = [r["sentence"] for r in precomputed_rows]
                        filter_time_sec = 0.0
                    else:
                        if distillsre_filter is None:
                            raise RuntimeError("DistillSRE filter is not initialized.")
                        filter_start = time.perf_counter()
                        ranked = distillsre_filter.rank_sentences(
                            question=question,
                            full_context="\n".join(base_passages),
                            sentences=sentences_before,
                        )
                        filter_time_sec += time.perf_counter() - filter_start
                        selected_rows = [
                            r
                            for r in ranked
                            if float(r.get("score", 0.0)) >= args.distillsre_score_th
                        ]
                        if args.max_selected_sentences > 0:
                            selected_rows = selected_rows[: args.max_selected_sentences]
                        selected_sentences = [str(r.get("sentence") or "").strip() for r in selected_rows]
                        selected_sentences = [s for s in selected_sentences if s]

                    num_sentences_after_filtering = len(selected_sentences)
                    num_selected_sentences = len(selected_sentences)
                    context_text = "\n".join(selected_sentences)
                    if not context_text and args.fallback_to_base_context:
                        context_text = "\n\n".join(base_passages)

                generation_start = time.perf_counter()
                prompt = build_merged_prompt(question, context_text, use_context=True)
                raw_pred, usage = client.generate(
                    prompt=prompt,
                    max_tokens=max(args.max_tokens, 1),
                    temperature=args.temperature,
                )
                generation_time_sec = time.perf_counter() - generation_start

                post_start = time.perf_counter()
                prediction = postprocess_prediction(raw_pred)
                pred_norm = norm(prediction)
                em = int(pred_norm in gold_norm_set) if gold_norm_set else 0
                postprocess_time_sec = time.perf_counter() - post_start

                context_num_chars = len(context_text)
                context_num_words = len(context_text.split()) if context_text else 0
                if context_tokenizer.enabled:
                    context_num_tokens = context_tokenizer.count(context_text)
                else:
                    context_num_tokens = None

                total_measured_time_sec = time.perf_counter() - total_start

                record = {
                    "question_id": question_id,
                    "question": question,
                    "dataset_name": dataset_name,
                    "split": split_name,
                    "bucket": bucket,
                    "method": method,
                    "prediction": prediction,
                    "gold_answers": gold_answers,
                    "em": em,
                    "context_text": None if args.omit_context_text else context_text,
                    "context_num_chars": context_num_chars,
                    "context_num_words": context_num_words,
                    "context_num_tokens": context_num_tokens,
                    "num_selected_sentences": num_selected_sentences,
                    "num_selected_passages": len(base_passages),
                    "num_reranked_candidates": num_reranked_candidates,
                    "num_sentences_before_filtering": num_sentences_before_filtering,
                    "num_sentences_after_filtering": num_sentences_after_filtering,
                    "retrieval_time_sec": retrieval_time_sec,
                    "fusion_time_sec": fusion_time_sec,
                    "rerank_time_sec": rerank_time_sec,
                    "sentence_split_time_sec": sentence_split_time_sec,
                    "filter_time_sec": filter_time_sec,
                    "generation_time_sec": generation_time_sec,
                    "postprocess_time_sec": postprocess_time_sec,
                    "total_measured_time_sec": total_measured_time_sec,
                    "usage_prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "usage_completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "usage_total_tokens": int(usage.get("total_tokens", 0) or 0),
                    "vllm_model_name_used": client.model,
                    "timing_mode": "precomputed_retrieval_rerank",
                    "distillsre_mode": args.distillsre_mode if method == METHOD_FUSE_QA_DISTILLSRE else "none",
                }

                records.append(record)
                write_record(writer, record)

                if args.log_every > 0 and step_i % args.log_every == 0:
                    method_rows = [r for r in records if r["method"] == method]
                    em_avg = safe_mean([r["em"] for r in method_rows])
                    pbar.set_postfix(
                        {
                            "avg_em": f"{(em_avg or 0.0):.4f}",
                            "avg_gen_s": f"{(safe_mean([r['generation_time_sec'] for r in method_rows]) or 0.0):.3f}",
                            "avg_filter_s": f"{(safe_mean([r['filter_time_sec'] for r in method_rows]) or 0.0):.3f}",
                        }
                    )

            pbar.close()

    config = {
        "dataset": dataset_name,
        "split": split_name,
        "method": args.method,
        "methods_executed": methods,
        "num_total_rows_in_split": len(ds),
        "num_sampled_rows": len(selected_indices),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "base_retr_col": args.base_retr_col,
        "sre_retr_col": args.sre_retr_col,
        "top_k_passages": args.top_k_passages,
        "clip_chars": args.clip_chars,
        "distillsre_mode": args.distillsre_mode,
        "max_selected_sentences": args.max_selected_sentences,
        "distillsre_score_th": args.distillsre_score_th,
        "vllm_base_url": base_url_v1,
        "vllm_model_name": args.vllm_model_name,
        "vllm_model_name_final": client.model,
        "model_fallback_enabled": not args.disable_model_fallback,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    summary_payload = build_summary_payload(records=records, config=config)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    csv_path, md_path = write_table_files(summary_payload, output_dir)

    print("\nStudy complete.")
    print(f"Per-example metrics: {per_example_path}")
    print(f"Summary JSON:        {summary_path}")
    print(f"Summary table CSV:   {csv_path}")
    print(f"Summary table MD:    {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
