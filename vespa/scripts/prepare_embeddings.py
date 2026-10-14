"""Embeds SciFact once and caches everything the Vespa runs need.

Shares the cache with ../../scripts: the arrays are written under the same names
stage 3 uses, so whichever side runs first pays the embedding cost and the other
reads it. Also writes ids.json, so the feed and evaluation scripts do not need
the datasets library.

Document order is list(doc_text.keys()) over BeIR/scifact's corpus split and the
query filter is the same as stage 3's, so row i of every array refers to the same
document or query in both halves of the repo.

Set EMB_CACHE to point the cache somewhere other than ../../scripts.
"""
import json, os, time
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OLD = "sentence-transformers/all-MiniLM-L6-v2"
NEW = "sentence-transformers/paraphrase-MiniLM-L6-v2"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
IDS = os.path.join(HERE, "ids.json")

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels = load_dataset("BeIR/scifact-qrels")["test"]

doc_text = {d["_id"]: (d.get("title", "") + " " + d["text"]).strip() for d in corpus}
query_text = {q["_id"]: q["text"] for q in queries}
relevant = defaultdict(set)
for r in qrels:
    if r["score"] > 0:
        relevant[str(r["query-id"])].add(str(r["corpus-id"]))

doc_ids = list(doc_text.keys())
doc_index = {d: i for i, d in enumerate(doc_ids)}
q_ids = [q for q in relevant if q in query_text and any(d in doc_index for d in relevant[q])]
docs = [doc_text[d] for d in doc_ids]
qtexts = [query_text[q] for q in q_ids]
print(f"{len(doc_ids)} documents, {len(q_ids)} evaluation queries", flush=True)


def embed(model_name, texts, tag):
    path = os.path.join(CACHE, f"emb_{tag}.npy")
    if os.path.exists(path):
        print(f"  {tag}: cached", flush=True)
        return np.load(path)
    t0 = time.time()
    print(f"  {tag}: embedding {len(texts)} texts with {model_name} ...", flush=True)
    v = np.asarray(SentenceTransformer(model_name).encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32)
    np.save(path, v)
    print(f"  {tag}: {time.time()-t0:.0f}s", flush=True)
    return v


os.makedirs(CACHE, exist_ok=True)
embed(OLD, docs, "docs_old")
embed(NEW, docs, "docs_new")
embed(OLD, qtexts, "q_old")
embed(NEW, qtexts, "q_new")

json.dump({"doc_ids": doc_ids, "q_ids": q_ids,
           "relevant": {q: sorted(relevant[q]) for q in q_ids},
           "doc_text": doc_text},
          open(IDS, "w"), indent=1)
print(f"saved {IDS}")
