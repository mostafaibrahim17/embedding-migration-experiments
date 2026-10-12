"""Stage 3: the main SciFact run.

Extends the first exploratory run with the three things it was missing:

  RUN 1  HNSW instead of exact search. Two build protocols:
           A  in-place update: graph built on OLD vectors, migrated documents
              re-added under the same label (models a partial-update migration).
           B  rebuild on the mix: fresh graph over the mixed vectors at each step
              (models a full reindex over a half-migrated index).
         For each: raw recall@10, per-space normalized recall computed only over
         the candidates the graph actually returned, and reachability (share of a
         query's relevant old-space documents that appear anywhere in the
         candidate set).
  RUN 2  A distant model pair (all-MiniLM-L6-v2 -> bge-small-en-v1.5), exact
         search. Neighbour overlap and the migration curve, five seeds.
  RUN 3  At each fraction, the share of top-10 slots taken by migrated documents
         against their share of the index (exact search, MiniLM pair).

BeIR/scifact, seeds 0-4, eleven fractions from 0% to 100%. Writes whatever comes
out, including nulls.

Recall@10 here is pooled across queries: total relevant documents retrieved in
the top 10 divided by total relevant documents, not the per-query mean. Every
script in this repo uses the same definition.
"""

import json, os, sys, time
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import hnswlib

OLD_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
NEW_MODEL     = "sentence-transformers/paraphrase-MiniLM-L6-v2"
DISTANT_MODEL = "BAAI/bge-small-en-v1.5"          # 384-d, different training objective
K = 10
SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = np.round(np.linspace(0, 1, 11), 1)
CACHE = os.environ.get("EMB_CACHE", ".")
OUT = "stage3_results.json"

def log(*a):
    print(*a, flush=True)

# ------------------------------------------------------------------ data
log("Loading BeIR/scifact ...")
corpus  = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels   = load_dataset("BeIR/scifact-qrels")["test"]

doc_text   = {d["_id"]: (d.get("title", "") + " " + d["text"]).strip() for d in corpus}
query_text = {q["_id"]: q["text"] for q in queries}
relevant = defaultdict(set)
for r in qrels:
    if r["score"] > 0:
        relevant[str(r["query-id"])].add(str(r["corpus-id"]))

doc_ids   = list(doc_text.keys())
doc_index = {d: i for i, d in enumerate(doc_ids)}
q_ids = [q for q in relevant if q in query_text and any(d in doc_index for d in relevant[q])]
N = len(doc_ids)
log(f"{N} documents, {len(q_ids)} evaluation queries")
docs   = [doc_text[d] for d in doc_ids]
qtexts = [query_text[q] for q in q_ids]

# relevant doc row indices per query (for reachability)
rel_rows = [np.array([doc_index[d] for d in relevant[q] if d in doc_index]) for q in q_ids]

# ------------------------------------------------------------------ embeddings (cached)
def embed(model_name, texts, tag):
    path = os.path.join(CACHE, f"emb_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    log(f"  embedding {len(texts)} texts with {model_name} ...")
    m = SentenceTransformer(model_name)
    v = np.asarray(m.encode(texts, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True), dtype=np.float32)
    np.save(path, v)
    return v

t0 = time.time()
docs_old = embed(OLD_MODEL, docs, "docs_old")
docs_new = embed(NEW_MODEL, docs, "docs_new")
q_new    = embed(NEW_MODEL, qtexts, "q_new")
docs_far = embed(DISTANT_MODEL, docs, "docs_far")
q_far    = embed(DISTANT_MODEL, qtexts, "q_far")
log(f"embeddings ready ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ metrics
def recall_from_topk(topk_rows):
    hits = total = 0
    for i, qid in enumerate(q_ids):
        rel = relevant[qid]
        got = {doc_ids[j] for j in topk_rows[i]}
        hits  += len(got & rel)
        total += len(rel)
    return hits / total

def exact_topk(qv, dv, k=K):
    sims = qv @ dv.T
    return np.argpartition(-sims, k, axis=1)[:, :k], sims

def exact_recall(qv, dv):
    top, _ = exact_topk(qv, dv)
    return recall_from_topk(top)

def normalized_topk_from_sims(sims, migrated_mask, k=K):
    """z-score scores per space over ALL docs, then top-k (exact-search version)."""
    out = np.empty((sims.shape[0], k), dtype=int)
    m = migrated_mask
    for i in range(sims.shape[0]):
        s = sims[i].copy()
        for grp in (m, ~m):
            if grp.sum() >= 2:
                g = s[grp]; s[grp] = (g - g.mean()) / (g.std() + 1e-9)
        out[i] = np.argpartition(-s, k)[:k]
    return out

def neighbours(vecs, k=10):
    s = vecs @ vecs.T
    np.fill_diagonal(s, -1)
    return np.argpartition(-s, k, axis=1)[:, :k]

def overlap(a, b):
    na, nb = neighbours(a), neighbours(b)
    return float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(na, nb)]))

results = {"n_docs": N, "n_queries": len(q_ids), "seeds": SEEDS,
           "fractions": [float(f) for f in FRACTIONS]}

# ------------------------------------------------------------------ baseline (exact, MiniLM pair)
base_new = exact_recall(q_new, docs_new)
crossed  = exact_recall(q_new, docs_old)
results["exact"] = {"new_vs_new": base_new, "new_vs_old": crossed,
                    "overlap_minilm_pair": overlap(docs_old, docs_new)}
log(f"\n=== exact baseline: new/new {base_new:.3f}  new/old {crossed:.3f}  "
    f"overlap {results['exact']['overlap_minilm_pair']:.2f}")

# ------------------------------------------------------------------ RUN 3 + exact curves (MiniLM)
log("\n=== RUN 3 / exact curves (MiniLM pair): raw, normalized, boundary share")
exact_curves = {"raw": [], "norm": [], "top10_migrated_share": []}
for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(N)
    raw_c, norm_c, share_c = [], [], []
    for f in FRACTIONS:
        n = int(f * N)
        mixed = docs_old.copy(); mixed[order[:n]] = docs_new[order[:n]]
        mask = np.zeros(N, bool); mask[order[:n]] = True
        top, sims = exact_topk(q_new, mixed)
        raw_c.append(recall_from_topk(top))
        norm_c.append(recall_from_topk(normalized_topk_from_sims(sims, mask)))
        share_c.append(float(mask[top].mean()))
    exact_curves["raw"].append(raw_c); exact_curves["norm"].append(norm_c)
    exact_curves["top10_migrated_share"].append(share_c)
    log(f"  seed {seed}: raw trough {min(raw_c):.3f} @ {FRACTIONS[int(np.argmin(raw_c))]:.1f}")
results["exact_curves_minilm"] = exact_curves
mean_raw = np.mean(exact_curves["raw"], axis=0)
mean_share = np.mean(exact_curves["top10_migrated_share"], axis=0)
log("  frac   raw    norm   top10-share-migrated  corpus-share")
for f, r, nm, sh in zip(FRACTIONS, mean_raw, np.mean(exact_curves["norm"], axis=0), mean_share):
    log(f"  {f:.1f}   {r:.3f}  {nm:.3f}     {sh:.2f}                 {f:.2f}")

# ------------------------------------------------------------------ RUN 1: HNSW
log("\n=== RUN 1: HNSW")
M, EFC = 16, 200

def build(vecs, ef):
    idx = hnswlib.Index(space="cosine", dim=vecs.shape[1])
    idx.init_index(max_elements=N, ef_construction=EFC, M=M, random_seed=100)
    idx.add_items(vecs, np.arange(N))
    idx.set_ef(ef)
    return idx

# tune ef_search so pure-new HNSW reaches >= 0.98 of exact recall
ef_used = None
for ef in [50, 100, 200, 400, 800]:
    idx = build(docs_new, ef)
    lab, _ = idx.knn_query(q_new, k=K)
    r = recall_from_topk(lab)
    log(f"  ef={ef}: pure-new HNSW recall {r:.3f} (exact {base_new:.3f}, ratio {r/base_new:.3f})")
    if r >= 0.98 * base_new:
        ef_used = ef; break
if ef_used is None:
    ef_used = 800
results["hnsw"] = {"M": M, "ef_construction": EFC, "ef_search": ef_used, "protocols": {}}
log(f"  using ef_search={ef_used}")

def hnsw_eval(idx, mask, ef):
    """raw recall, candidate-normalized recall, reachability of relevant OLD-space docs."""
    kc = min(ef, N)
    labs, dists = idx.knn_query(q_new, k=kc)          # candidates the graph returns
    sims = 1.0 - dists
    raw_top = labs[:, :K]
    raw = recall_from_topk(raw_top)
    # per-space z-score over candidates only, then top-K
    norm_top = np.empty((len(q_ids), K), dtype=int)
    for i in range(len(q_ids)):
        s = sims[i].copy(); mm = mask[labs[i]]
        for grp in (mm, ~mm):
            if grp.sum() >= 2:
                g = s[grp]; s[grp] = (g - g.mean()) / (g.std() + 1e-9)
        norm_top[i] = labs[i][np.argsort(-s)[:K]]
    norm = recall_from_topk(norm_top)
    # reachability: of each query's relevant docs still in OLD space, share present in candidates
    reach, cnt = 0.0, 0
    for i in range(len(q_ids)):
        rel_old = [d for d in rel_rows[i] if not mask[d]]
        if not rel_old: continue
        cand = set(labs[i].tolist())
        reach += sum(1 for d in rel_old if d in cand) / len(rel_old); cnt += 1
    return raw, norm, (reach / cnt if cnt else float("nan"))

for proto in ("A_inplace_update", "B_rebuild_on_mix"):
    log(f"\n  protocol {proto}")
    curves = {"raw": [], "norm": [], "reach": []}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(N)
        raw_c, norm_c, reach_c = [], [], []
        idx = build(docs_old, ef_used) if proto.startswith("A") else None
        prev = 0
        for f in FRACTIONS:
            n = int(f * N)
            mask = np.zeros(N, bool); mask[order[:n]] = True
            if proto.startswith("A"):
                if n > prev:
                    idx.add_items(docs_new[order[prev:n]], order[prev:n])   # in-place update, same labels
                    prev = n
            else:
                mixed = docs_old.copy(); mixed[order[:n]] = docs_new[order[:n]]
                idx = build(mixed, ef_used)
            raw, norm, reach = hnsw_eval(idx, mask, ef_used)
            raw_c.append(raw); norm_c.append(norm); reach_c.append(reach)
        curves["raw"].append(raw_c); curves["norm"].append(norm_c); curves["reach"].append(reach_c)
        log(f"    seed {seed}: raw trough {min(raw_c):.3f} @ {FRACTIONS[int(np.argmin(raw_c))]:.1f}   "
            f"norm min {min(norm_c):.3f}   reach@0.2 {reach_c[2]:.2f}")
    results["hnsw"]["protocols"][proto] = curves
    mr, mn, mre = (np.mean(curves[k], axis=0) for k in ("raw", "norm", "reach"))
    log("    frac   raw    norm   reach(relevant old docs in candidates)")
    for f, r, nm, re_ in zip(FRACTIONS, mr, mn, mre):
        log(f"    {f:.1f}   {r:.3f}  {nm:.3f}  {re_:.2f}")

# ------------------------------------------------------------------ RUN 2: distant pair
log("\n=== RUN 2: distant pair  all-MiniLM-L6-v2 -> bge-small-en-v1.5 (exact)")
far_base = exact_recall(q_far, docs_far)
far_cross = exact_recall(q_far, docs_old)
far_ov = overlap(docs_old, docs_far)
log(f"  far/far {far_base:.3f}   far-query/old-docs {far_cross:.3f}   overlap {far_ov:.2f}")
far_curves = {"raw": [], "norm": []}
for seed in SEEDS:
    order = np.random.default_rng(seed).permutation(N)
    raw_c, norm_c = [], []
    for f in FRACTIONS:
        n = int(f * N)
        mixed = docs_old.copy(); mixed[order[:n]] = docs_far[order[:n]]
        mask = np.zeros(N, bool); mask[order[:n]] = True
        top, sims = exact_topk(q_far, mixed)
        raw_c.append(recall_from_topk(top))
        norm_c.append(recall_from_topk(normalized_topk_from_sims(sims, mask)))
    far_curves["raw"].append(raw_c); far_curves["norm"].append(norm_c)
    log(f"  seed {seed}: raw trough {min(raw_c):.3f} @ {FRACTIONS[int(np.argmin(raw_c))]:.1f}   start {raw_c[0]:.3f}  end {raw_c[-1]:.3f}")
results["distant_pair"] = {"model": DISTANT_MODEL, "far_vs_far": far_base,
                           "farq_vs_olddocs": far_cross, "overlap": far_ov, "curves": far_curves}
mr = np.mean(far_curves["raw"], axis=0); mn = np.mean(far_curves["norm"], axis=0)
log("  frac   raw    norm")
for f, r, nm in zip(FRACTIONS, mr, mn):
    log(f"  {f:.1f}   {r:.3f}  {nm:.3f}")

json.dump(results, open(OUT, "w"), indent=1)
log(f"\nsaved {OUT}  (total {time.time()-t0:.0f}s)")
