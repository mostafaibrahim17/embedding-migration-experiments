"""Stage 2: why the low point comes early.

Against a new-model query, do already-migrated documents score higher than the
ones still in the old space? This measures the mean cosine similarity of each
group to the new-model queries, at each migration fraction, and the share of
top-10 slots the migrated group takes.

Same SciFact data, same close pair, same five seeds as stage 3. Reuses the
cached embeddings stage 3 writes; set EMB_CACHE to point at them.
"""
import json, os
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OLD = "sentence-transformers/all-MiniLM-L6-v2"
NEW = "sentence-transformers/paraphrase-MiniLM-L6-v2"
K = 10
SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = [round(f, 1) for f in np.linspace(0, 1, 11)]
CACHE = os.environ.get("EMB_CACHE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_cosine_results.json")

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
N = len(doc_ids)
docs = [doc_text[d] for d in doc_ids]
qtexts = [query_text[q] for q in q_ids]
print(f"{N} documents, {len(q_ids)} evaluation queries", flush=True)


def embed(model_name, texts, tag):
    path = os.path.join(CACHE, f"emb_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    print(f"  embedding {len(texts)} texts with {model_name} ...", flush=True)
    v = np.asarray(SentenceTransformer(model_name).encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32)
    np.save(path, v)
    return v


docs_old = embed(OLD, docs, "docs_old")
docs_new = embed(NEW, docs, "docs_new")
q_new = embed(NEW, qtexts, "q_new")

# Mean cosine of each group to the new-model queries, averaged over queries.
# Vectors are L2-normalized, so the dot product is the cosine.
sims_old = q_new @ docs_old.T          # queries x docs, old-space documents
sims_new = q_new @ docs_new.T          # queries x docs, new-space documents

# How far apart the two groups really are. The means alone invite a picture of two
# separated clusters, which is wrong: the distributions overlap heavily and
# migrated documents win most top slots rather than all of them, which is why the
# top-10 share below is 84% and 92% rather than 100%.
share_old_above = float((sims_old > np.median(sims_new)).mean())

res = {"n_docs": N, "n_queries": len(q_ids), "seeds": SEEDS, "fractions": FRACTIONS,
       "whole_corpus": {
           "mean_cosine_old_space": float(sims_old.mean()),
           "mean_cosine_new_space": float(sims_new.mean()),
           "std_cosine_old_space": float(sims_old.std()),
           "std_cosine_new_space": float(sims_new.std()),
           "share_old_space_above_median_new_space": share_old_above},
       "by_fraction": {}}
print(f"  share of old-space scores above the median migrated score: {share_old_above:.4f}")
print(f"\nWhole corpus, mean cosine to a new-model query:")
print(f"  documents in the old space: {sims_old.mean():.3f}")
print(f"  documents in the new space: {sims_new.mean():.3f}")

# Per fraction: split the corpus by what has actually migrated under each seed.
print("\nfrac   migrated  un-migrated   top10 share migrated")
for f in FRACTIONS:
    mig, unmig, share = [], [], []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(N)
        n = int(f * N)
        mask = np.zeros(N, bool)
        mask[order[:n]] = True
        mixed = docs_old.copy()
        mixed[mask] = docs_new[mask]
        sims = q_new @ mixed.T
        if mask.any():
            mig.append(float(sims[:, mask].mean()))
        if (~mask).any():
            unmig.append(float(sims[:, ~mask].mean()))
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        share.append(float(mask[top].mean()))
    row = {"mean_cosine_migrated": (sum(mig) / len(mig)) if mig else None,
           "mean_cosine_unmigrated": (sum(unmig) / len(unmig)) if unmig else None,
           "top10_share_migrated": sum(share) / len(share)}
    res["by_fraction"][str(f)] = row
    m = f"{row['mean_cosine_migrated']:.3f}" if row["mean_cosine_migrated"] is not None else "   - "
    u = f"{row['mean_cosine_unmigrated']:.3f}" if row["mean_cosine_unmigrated"] is not None else "   - "
    print(f" {f:.1f}    {m}       {u}            {row['top10_share_migrated']:.2f}")

json.dump(res, open(OUT, "w"), indent=1, allow_nan=False)
print(f"\nsaved {OUT}")
