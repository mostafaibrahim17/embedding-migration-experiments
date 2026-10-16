"""Stage 8: where do the normalization statistics have to come from?

Two questions the earlier stages left open, both about the per-space z-score
rather than about the migration.

ONE. stage 3 and the Vespa runs do not normalize the same way. stage 3 scores a
candidate list and can only compute each space's mean and standard deviation
from the candidates it got back. The Vespa rank profile receives them as query
inputs, computed over every document in each space. So the simulation's 0.38 and
Vespa's 0.48 for the same single-field migration differ by protocol as well as by
search width, and anyone comparing them needs to know by how much. This measures
it at a fixed candidate budget, so the only thing that changes is where the
statistics came from.

TWO. Both READMEs call the whole-space statistics an oracle and treat the
recovery as a ceiling, on the grounds that computing them means the exhaustive
scan a vector index exists to avoid. That turns out to be too pessimistic. The
statistics are two numbers per space; a sample estimates them. This measures how
large the sample has to be.

Exact search throughout, so no graph noise enters. Reuses the cached embeddings
stage 3 writes.

    python3 stage8_normalization_statistics.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", HERE)
OUT = os.path.join(HERE, "stage8_normalization_statistics.json")
IDS = os.path.join(HERE, "..", "vespa", "scripts", "ids.json")
K = 10
SEEDS = [0, 1, 2, 3, 4]
FRACTION = 0.2            # the low point
CANDIDATES = 50           # stage 3's ef_search 50
SAMPLES = [32, 64, 128, 256, 512, 1024]

meta = json.load(open(IDS))
doc_ids, q_ids, relevant = meta["doc_ids"], meta["q_ids"], meta["relevant"]
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
q_new = np.load(os.path.join(CACHE, "emb_q_new.npy"))
N = len(doc_ids)
print(f"{N} documents, {len(q_ids)} queries, {FRACTION:.0%} migrated, exact search", flush=True)


def pooled(top):
    hits = tot = 0
    for i, qid in enumerate(q_ids):
        got = {doc_ids[j] for j in top[i]}
        rel = set(relevant[qid])
        hits += len(got & rel)
        tot += len(rel)
    return hits / tot


def mixed_state(seed):
    order = np.random.default_rng(seed).permutation(N)
    mask = np.zeros(N, dtype=bool)
    mask[order[:int(FRACTION * N)]] = True
    mixed = docs_old.copy()
    mixed[mask] = docs_new[mask]
    return mask, mixed, q_new @ mixed.T


def z_from_candidates(mask, S):
    """What stage 3 does: statistics from the returned candidates only."""
    top = []
    for i in range(len(q_ids)):
        cand = np.argpartition(-S[i], CANDIDATES)[:CANDIDATES]
        sc, m = S[i, cand], mask[cand]
        z = np.zeros_like(sc)
        for grp in (m, ~m):
            if grp.any():
                z[grp] = (sc[grp] - sc[grp].mean()) / (sc[grp].std() or 1.0)
        top.append(cand[np.argsort(-z)[:K]])
    return pooled(top)


def z_from_space(mask, mixed, S, sample=None, rng=None, rerank=True):
    """What Vespa does: statistics over each space, optionally from a sample."""
    mig, unm = mixed[mask], mixed[~mask]
    if sample:
        mig = mig[rng.choice(len(mig), min(sample, len(mig)), replace=False)]
        unm = unm[rng.choice(len(unm), min(sample, len(unm)), replace=False)]
    top = []
    for i in range(len(q_ids)):
        sn, so = q_new[i] @ mig.T, q_new[i] @ unm.T
        zz = np.where(mask, (S[i] - sn.mean()) / (sn.std() or 1.0),
                            (S[i] - so.mean()) / (so.std() or 1.0))
        if rerank:                                   # rerank a candidate list
            cand = np.argpartition(-S[i], CANDIDATES)[:CANDIDATES]
            top.append(cand[np.argsort(-zz[cand])[:K]])
        else:                                        # exact search, every document
            top.append(np.argpartition(-zz, K)[:K])
    return pooled(top)


raw, cand_stats, space_stats_rerank, full = [], [], [], []
for seed in SEEDS:
    mask, mixed, S = mixed_state(seed)
    raw.append(pooled([np.argpartition(-S[i], K)[:K] for i in range(len(q_ids))]))
    cand_stats.append(z_from_candidates(mask, S))
    space_stats_rerank.append(z_from_space(mask, mixed, S))
    full.append(z_from_space(mask, mixed, S, rerank=False))

print(f"\nat {CANDIDATES} candidates, the only difference is where the statistics came from:")
print(f"  no normalization                       {np.mean(raw):.4f}")
print(f"  statistics from the candidates only    {np.mean(cand_stats):.4f}   <- what stage 3 reports")
print(f"  statistics over each whole space       {np.mean(space_stats_rerank):.4f}   <- what Vespa is given")
print(f"  difference                             {np.mean(space_stats_rerank)-np.mean(cand_stats):+.4f}")
print(f"\nexact search over every document, whole-space statistics  {np.mean(full):.4f}")

print("\nhow much of each space do those statistics actually need?")
sampled = {}
for n in SAMPLES:
    vals = []
    for seed in SEEDS:
        mask, mixed, S = mixed_state(seed)
        vals.append(z_from_space(mask, mixed, S, sample=n,
                                 rng=np.random.default_rng(1000 + seed), rerank=False))
    sampled[n] = float(np.mean(vals))
    print(f"  {n:5d} documents sampled per space   {sampled[n]:.4f}"
          f"   ({sampled[n]-np.mean(full):+.4f} against every document)")

res = {"stage": 8, "fraction": FRACTION, "candidates": CANDIDATES, "seeds": SEEDS,
       "k": K, "n_docs": N, "n_queries": len(q_ids),
       "what": "where the per-space normalization statistics come from, at the low point",
       "raw_no_normalization": float(np.mean(raw)),
       "statistics_from_candidates_only": float(np.mean(cand_stats)),
       "statistics_from_whole_space_reranking_candidates": float(np.mean(space_stats_rerank)),
       "statistics_from_whole_space_exact_search": float(np.mean(full)),
       "protocol_difference_at_equal_budget":
           float(np.mean(space_stats_rerank) - np.mean(cand_stats)),
       "sampled_statistics_exact_search": sampled,
       "note": "stage 3 normalizes the candidates it returned; the Vespa rank profile is "
               "given statistics over each whole space. At a fixed candidate budget that "
               "protocol difference is worth about 0.10, which is why the simulation "
               "reports 0.38 for the single-field case and Vespa reports 0.48. The "
               "whole-space statistics are not an expensive oracle: a few hundred sampled "
               "documents per space land within 0.001 of using every one."}
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")
