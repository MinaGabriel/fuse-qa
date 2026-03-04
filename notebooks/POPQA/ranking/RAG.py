#!/usr/bin/env python
# coding: utf-8

# ## NO SRE (Study R@K)

# In[ ]:


import os
import subprocess
import json
import numpy as np
import faiss
import torch
from huggingface_hub import snapshot_download
from pyserini.search.lucene import LuceneSearcher
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ============================================================
# CONFIG
# ============================================================
INDEX_PATH = "../../wiki_dpr.index"
META_PATH  = "../../wiki_dpr_meta.npz"

BM25_REPO  = "MinaGabriel/wiki18-bm25-index"
DPR_MODEL  = "facebook/dpr-question_encoder-single-nq-base"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

K_BM25  = 100
K_DENSE = 30
FUSE_K  = 200
TOP_K   = 30

DPR_TEXT_CHARS  = 2000
BM25_TEXT_CHARS = 2000

DENSE_GPU  = 0   # DPR encoder + FAISS GPU
RERANK_GPU = 2   # BGE reranker GPU

# ============================================================
# JAVA (Pyserini)
# ============================================================
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")
print(subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode())

# ============================================================
# DEVICES
# ============================================================
use_cuda = torch.cuda.is_available()
device_dense  = torch.device(f"cuda:{DENSE_GPU}" if use_cuda else "cpu")
device_rerank = torch.device(f"cuda:{RERANK_GPU}" if use_cuda and torch.cuda.device_count() > RERANK_GPU else "cpu")
print(f"Dense retrieval device:  {device_dense}")
print(f"Reranker device:         {device_rerank}")

# ============================================================
# LOAD BM25
# ============================================================
print("Loading BM25 index...")
local_dir = snapshot_download(repo_id=BM25_REPO, repo_type="dataset")
index_dir = os.path.join(local_dir, "bm25_index")
searcher = LuceneSearcher(index_dir)

# ============================================================
# LOAD DPR FAISS + META
# ============================================================
print("Loading FAISS index...")
index_cpu = faiss.read_index(INDEX_PATH)

if use_cuda:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, DENSE_GPU, index_cpu)
    del index_cpu
else:
    index = index_cpu

meta = np.load(META_PATH, allow_pickle=True, mmap_mode="r")
doc_ids = meta["doc_ids"]
titles  = meta["titles"]
texts   = meta["texts"]

# ============================================================
# LOAD DPR QUESTION ENCODER
# ============================================================
print("Loading DPR model...")
q_tok   = DPRQuestionEncoderTokenizerFast.from_pretrained(DPR_MODEL)
q_model = DPRQuestionEncoder.from_pretrained(DPR_MODEL).to(device_dense).eval()

# ============================================================
# LOAD BGE RERANKER (cross-encoder)
# ============================================================
print("Loading BGE reranker...")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    RERANK_MODEL_NAME,
    torch_dtype=torch.float16 if device_rerank.type == "cuda" else torch.float32,
    attn_implementation="eager",
).to(device_rerank).eval()

# ============================================================
# HELPERS
# ============================================================
def _norm_key(title: str, text: str) -> str:
    t = (title or "").strip().lower()
    x = (text or "").strip().lower()
    return t + "||" + x[:200]

# ============================================================
# DPR DENSE SEARCH
# ============================================================
@torch.no_grad()
def dense_search(question: str, top_k: int = 30):
    enc = q_tok(question, truncation=True, padding=True, max_length=64, return_tensors="pt")
    enc = {k: v.to(device_dense) for k, v in enc.items()}
    emb = q_model(**enc).pooler_output  # [1, d]

    vec = np.ascontiguousarray(emb.detach().cpu().numpy().astype(np.float32))

    # NOTE: Keep this if your FAISS index was built for cosine/IP with normalized vectors.
    # If your index was built for L2 without normalization, remove this line.
    faiss.normalize_L2(vec)

    scores, idxs = index.search(vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        idx = int(idx)
        results.append({
            "source": "dpr",
            "rank": rank,
            "score": float(score),
            "doc_id": str(doc_ids[idx]),
            "title": str(titles[idx]),
            "text": str(texts[idx]),
        })
    return results

# ============================================================
# BM25 SEARCH
# ============================================================
def bm25_search(question: str, top_k: int = 100):
    hits = searcher.search(question, top_k)
    results = []
    for rank, h in enumerate(hits, start=1):
        raw_json = searcher.doc(h.docid).raw()
        record = json.loads(raw_json)
        contents = record.get("contents", "") or ""
        results.append({
            "source": "bm25",
            "rank": rank,
            "score": float(h.score),
            "doc_id": str(h.docid),
            "title": record.get("title", ""),
            "text": contents,
        })
    return results

# ============================================================
# RRF FUSION + DEDUPE
# ============================================================
def fuse_candidates(bm25_res, dpr_res, fuse_k: int = 200, rrf_k: int = 60):
    pool = {}
    for r in bm25_res + dpr_res:
        key = _norm_key(r.get("title", ""), r.get("text", ""))
        if key not in pool:
            pool[key] = {**r, "rrf": 0.0}
        pool[key]["rrf"] += 1.0 / (rrf_k + r["rank"])

    fused = sorted(pool.values(), key=lambda x: x["rrf"], reverse=True)
    return fused[:fuse_k]

# ============================================================
# BGE RERANK (CROSS-ENCODER)
# ============================================================
@torch.no_grad()
def rerank_with_bge_reranker(
    question: str,
    candidates: list[dict],
    top_k: int = 25,
    batch_size: int = 16,
    max_length: int = 384,
    text_key: str = "text",
):
    if not candidates:
        return []

    passages = [(c.get(text_key) or "").strip() for c in candidates]
    pairs = [(question, p) for p in passages]

    scores = np.empty(len(pairs), dtype=np.float32)

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        enc = rerank_tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device_rerank) for k, v in enc.items()}

        out = rerank_model(**enc)
        batch_scores = out.logits.squeeze(-1)  # [B]
        scores[start:start + len(batch_pairs)] = batch_scores.detach().float().cpu().numpy()

    order = np.argsort(-scores)[:top_k]
    reranked = []
    for i in order:
        item = dict(candidates[int(i)])
        item["rerank_score"] = float(scores[int(i)])
        reranked.append(item)

    return reranked

# ============================================================
# MAIN PIPELINE
# ============================================================
def retrieve_and_rerank(
    question: str,
    k_bm25: int = K_BM25,
    k_dense: int = K_DENSE,
    fuse_k: int = FUSE_K,
    top_k: int = TOP_K,
):
    bm25_res = bm25_search(question, top_k=k_bm25)
    dpr_res  = dense_search(question, top_k=k_dense)
    fused    = fuse_candidates(bm25_res, dpr_res, fuse_k=fuse_k)

    reranked = rerank_with_bge_reranker(
        question=question,
        candidates=fused,
        top_k=top_k,
        batch_size=16,
        max_length=384,
        text_key="text",
    )
    return reranked




# In[ ]:


from datasets import load_dataset, DatasetDict
raw = load_dataset("akariasai/PopQA")['test']


# In[ ]:


# from prettytable import PrettyTable

# # Create a PrettyTable object
# table = PrettyTable()

# # Define column headers
# table.field_names = ["#", "rrf", "rerank_score", "Rank", "Score", "Source", "Doc ID", "Title", "Text"]

# # Align text columns to the left
# table.align["Text"] = "l"
# table.align["Title"] = "l"

# # Optional: Set max width for Text to avoid overly wide tables
# table.max_width["Text"] = 500  # adjust as needed
# counter = 0
# # Fetch and process results
# key = ['Jerrold J. Katz', 'Jerrold Jacob Katz', 'Jerrold Jay Katz', 'J. Katz', 'Jerrold Katz (philosopher)']
# for r in retrieve_and_rerank("In what city was Bill Fellowes born?",k_bm25=500, k_dense=500, fuse_k=500, top_k=20):
#     # Remove newline characters from text and collapse extra spaces
#     clean_text = r['text'].replace('\n', ' ').replace('\r', ' ')
#     # Optional: Clean up extra spaces
#     import re
#     clean_text = re.sub(r'\s+', ' ', clean_text).strip()
#     counter+= 1
#     table.add_row([
#         counter, 
#         f"{r['rrf']:.4f}",
#         f"{r['rerank_score']:.4f}",
#         r['rank'],
#         f"{r['score']:.4f}",
#         r['source'],
#         r['doc_id'],
#         r['title'],
#         clean_text  # now guaranteed to be on one line
#     ])

# print(table)


# In[ ]:


import numpy as np

s_pops = np.array([int(ex["s_pop"]) for ex in raw], dtype=np.int64)
T = int(np.percentile(s_pops, 80))   # top 20% is "frequent"
# frequent if s_pop >= T
print("80th percentile threshold =", T)


# In[ ]:


# popqa_retrieval_rank_report_simple.py
# (DROP-IN) Only change: add TOTAL/ALL report (no other logic changed)

import json, re
import numpy as np
from tqdm import tqdm

# -------------------------
# HARD-CODED SETTINGS
# -------------------------
OUT_FILE = "popqa_retrieval_rank_report.txt"

K_BM25 = 500
K_DENSE = 500
FUSE_K = 500
TOP_K = 20
K_LIST = [1, 5, 10, 20]

FREQ_THRESHOLD = T  # Frequent if s_pop >= this

TEXT_KEY = "text"        # retrieve_and_rerank docs use 'text'
ANS_KEY = "possible_answers"  # PopQA gold list stored here

# -------------------------
# helpers
# -------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

def norm(s: str) -> str:
    s = "" if s is None else str(s).lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def parse_answers(x):
    # possible_answers may be list[str] or a JSON string like '["Lubango"]'
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
            except:
                pass
        return [s]
    return [str(x)]

def first_hit_rank_top20(retrieved_docs, gold_aliases):
    # return first rank (1..20) where any gold alias appears in doc text; else None
    if not retrieved_docs or not gold_aliases:
        return None
    gold_norm = [norm(g) for g in gold_aliases if norm(g)]
    if not gold_norm:
        return None

    for i, d in enumerate(retrieved_docs[:TOP_K], start=1):
        txt = d.get(TEXT_KEY, "") if isinstance(d, dict) else str(d)
        txt = norm(txt)
        if not txt:
            continue
        for g in gold_norm:
            if g and (g in txt):
                return i
    return None

def mean01(xs):
    return float(np.mean(xs)) if xs else 0.0

# -------------------------
# data
# -------------------------
ds = raw 
n = len(ds)
print("n =", n)

# store hits per group
freq_hits = {k: [] for k in K_LIST}
nonf_hits = {k: [] for k in K_LIST}
freq_n = 0
nonf_n = 0

# (NEW) store hits for ALL
all_hits = {k: [] for k in K_LIST}

for ex in tqdm(ds, total=n, desc="Evaluating R@k (gold in top-k docs)"):
    q = ex["question"]
    gold = parse_answers(ex.get(ANS_KEY))
    s_pop = int(ex.get("s_pop", 0))

    retrieved = retrieve_and_rerank(
        q,
        k_bm25=K_BM25,
        k_dense=K_DENSE,
        fuse_k=FUSE_K,
        top_k=TOP_K,
    )
    hit_rank = first_hit_rank_top20(retrieved, gold)

    # (NEW) update ALL hits
    for k in K_LIST:
        all_hits[k].append(1 if (hit_rank is not None and hit_rank <= k) else 0)

    is_freq = (s_pop >= FREQ_THRESHOLD)
    if is_freq:
        freq_n += 1
        for k in K_LIST:
            freq_hits[k].append(1 if (hit_rank is not None and hit_rank <= k) else 0)
    else:
        nonf_n += 1
        for k in K_LIST:
            nonf_hits[k].append(1 if (hit_rank is not None and hit_rank <= k) else 0)

# compute metrics
all_scores  = {k: mean01(all_hits[k]) for k in K_LIST}
freq_scores = {k: mean01(freq_hits[k]) for k in K_LIST}
nonf_scores = {k: mean01(nonf_hits[k]) for k in K_LIST}

# write report
lines = []
lines.append("=" * 80)
lines.append("PopQA Retrieval-Only Report (R@k) — Frequent vs Non-Frequent")
lines.append("=" * 80)
lines.append(f"Split: test | n={n}")
lines.append(f"Retrieval: retrieve_and_rerank(k_bm25={K_BM25}, k_dense={K_DENSE}, fuse_k={FUSE_K}, top_k={TOP_K})")
lines.append("")
lines.append("Metric definition:")
lines.append("- R@k = fraction of questions where ANY gold alias appears (substring match) in ANY of the top-k retrieved docs.")
lines.append("")
lines.append(f"Frequent definition: s_pop >= {FREQ_THRESHOLD}")
lines.append("")

# (NEW) TOTAL / ALL section
lines.append(f"ALL:        n={n}")
lines.append(f"  R@1  = {all_scores[1]:.4f}")
lines.append(f"  R@5  = {all_scores[5]:.4f}")
lines.append(f"  R@10 = {all_scores[10]:.4f}")
lines.append(f"  R@20 = {all_scores[20]:.4f}")
lines.append("")

lines.append(f"FREQUENT:   n={freq_n}")
lines.append(f"  R@1  = {freq_scores[1]:.4f}")
lines.append(f"  R@5  = {freq_scores[5]:.4f}")
lines.append(f"  R@10 = {freq_scores[10]:.4f}")
lines.append(f"  R@20 = {freq_scores[20]:.4f}")
lines.append("")
lines.append(f"NON-FREQ:   n={nonf_n}")
lines.append(f"  R@1  = {nonf_scores[1]:.4f}")
lines.append(f"  R@5  = {nonf_scores[5]:.4f}")
lines.append(f"  R@10 = {nonf_scores[10]:.4f}")
lines.append(f"  R@20 = {nonf_scores[20]:.4f}")
lines.append("")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Saved:", OUT_FILE)

