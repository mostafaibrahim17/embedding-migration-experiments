"""stage7: does a wider HNSW search (ef_search 200) recover more of the trough than ef 50?
Same SciFact data, same MiniLM pair, same seeds/protocols as stage3. Fractions 0-0.3 only."""
import json, os, time
from collections import defaultdict
import numpy as np
from datasets import load_dataset
import hnswlib

CACHE = os.path.dirname(os.path.abspath(__file__))
K = 10; SEEDS = [0, 1, 2, 3, 4]; FRACTIONS = [0.0, 0.1, 0.2, 0.3]
EFS = [50, 200]
OUT = "/Users/ahmedfaramawy/Desktop/Agolia/stage7_ef_results.json"

corpus  = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels   = load_dataset("BeIR/scifact-qrels")["test"]
doc_text   = {d["_id"]: (d.get("title", "") + " " + d["text"]).strip() for d in corpus}
query_text = {q["_id"]: q["text"] for q in queries}
relevant = defaultdict(set)
for r in qrels:
    if r["score"] > 0: relevant[str(r["query-id"])].add(str(r["corpus-id"]))
doc_ids = list(doc_text.keys()); doc_index = {d: i for i, d in enumerate(doc_ids)}
q_ids = [q for q in relevant if q in query_text and any(d in doc_index for d in relevant[q])]
N = len(doc_ids)
rel_rows = [np.array([doc_index[d] for d in relevant[q] if d in doc_index]) for q in q_ids]
docs_old = np.load(f"{CACHE}/emb_docs_old.npy"); docs_new = np.load(f"{CACHE}/emb_docs_new.npy")
q_new = np.load(f"{CACHE}/emb_q_new.npy")
assert docs_old.shape[0] == N and q_new.shape[0] == len(q_ids)
print(N, "docs", len(q_ids), "queries", flush=True)

def recall_from_topk(topk_rows):
    hits = total = 0
    for i, qid in enumerate(q_ids):
        rel = relevant[qid]; got = {doc_ids[j] for j in topk_rows[i]}
        hits += len(rel & got); total += len(rel)
    return hits / total

def exact_recall(qv, dv):
    sims = qv @ dv.T
    return recall_from_topk(np.argsort(-sims, axis=1)[:, :K])

def build(vecs, ef):
    idx = hnswlib.Index(space="cosine", dim=vecs.shape[1])
    idx.init_index(max_elements=N, ef_construction=200, M=16, random_seed=100)
    idx.add_items(vecs, np.arange(N)); idx.set_ef(ef); return idx

def hnsw_eval(idx, mask, ef):
    kc = min(ef, N)
    labs, dists = idx.knn_query(q_new, k=kc); sims = 1.0 - dists
    raw = recall_from_topk(labs[:, :K])
    norm_top = np.empty((len(q_ids), K), dtype=int)
    for i in range(len(q_ids)):
        s = sims[i].copy(); mm = mask[labs[i]]
        for grp in (mm, ~mm):
            if grp.sum() >= 2:
                g = s[grp]; s[grp] = (g - g.mean()) / (g.std() + 1e-9)
        norm_top[i] = labs[i][np.argsort(-s)[:K]]
    norm = recall_from_topk(norm_top)
    reach, cnt = 0.0, 0
    for i in range(len(q_ids)):
        rel_old = [d for d in rel_rows[i] if not mask[d]]
        if not rel_old: continue
        cand = set(labs[i].tolist()); reach += sum(1 for d in rel_old if d in cand) / len(rel_old); cnt += 1
    return raw, norm, reach / cnt

res = {"exact_new_vs_new": exact_recall(q_new, docs_new), "exact_new_vs_old": exact_recall(q_new, docs_old),
       "fractions": FRACTIONS, "seeds": SEEDS, "runs": {}}
# exact normalized at these fractions for reference
t0 = time.time()
for ef in EFS:
    for proto in ("A_inplace_update", "B_rebuild_on_mix"):
        curves = {"raw": [], "norm": [], "reach": []}
        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(N)
            idx = build(docs_old, ef) if proto.startswith("A") else None
            prev = 0; rc, nc, rec = [], [], []
            for f in FRACTIONS:
                n = int(f * N); mask = np.zeros(N, bool); mask[order[:n]] = True
                if proto.startswith("A"):
                    if n > prev: idx.add_items(docs_new[order[prev:n]], order[prev:n]); prev = n
                else:
                    mixed = docs_old.copy(); mixed[order[:n]] = docs_new[order[:n]]; idx = build(mixed, ef)
                r, nm, re_ = hnsw_eval(idx, mask, ef); rc.append(r); nc.append(nm); rec.append(re_)
            curves["raw"].append(rc); curves["norm"].append(nc); curves["reach"].append(rec)
        res["runs"][f"ef{ef}_{proto}"] = curves
        mr, mn, mre = (np.mean(curves[k], axis=0) for k in ("raw", "norm", "reach"))
        print(f"\nef={ef} {proto}   ({time.time()-t0:.0f}s)")
        for f, a, b, c in zip(FRACTIONS, mr, mn, mre):
            print(f"  {f:.1f}  raw {a:.3f}  norm {b:.3f}  reach {c:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
