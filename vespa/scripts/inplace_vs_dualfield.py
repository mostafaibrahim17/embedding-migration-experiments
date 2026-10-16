"""Step 9b: is the second field what makes normalization work on a graph?

Step 9 found that per-space normalization repairs the low point on Vespa's graph,
recovering 0.3156 to 0.5929 at 20% migrated, where the simulation's graph only
reached 0.3746. The difference is not that Vespa normalizes better. It is that a
mixed state in Vespa is necessarily two fields and therefore two HNSW graphs,
each nearestNeighbor operator holding its own candidate budget, so documents in
the new space cannot crowd documents in the old space out of retrieval.

If that is the cause, migrating the other way should reproduce the simulation's
failure in Vespa: overwrite the single vector field in place, so one graph holds
vectors from both models, and normalization should stop working.

This runs that migration. emb_old is overwritten with new-model vectors for the
documents that have migrated, which is what an in-place model change looks like
when you do not add a second field. One operator, one graph, one candidate budget.

Destructive: it overwrites emb_old. Re-run feed_old_space.py afterwards, which
this script does for you at the end.

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 inplace_vs_dualfield.py
"""
import json, os, subprocess, sys, time
import numpy as np
import vespa_eval as V

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
TAG = os.environ.get("OUT_TAG", "")
OUT = os.path.join(HERE, "..", "results", f"step9b_inplace_vs_dualfield{TAG}.json")
JSONL = os.path.join(HERE, "inplace_slice.jsonl")
ENDPOINT = V.endpoint()
SEED = int(os.environ.get("SEED", "0"))
FRACTIONS = [round(float(x), 2) for x in
             os.environ.get("FRACTIONS", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0").split(",")]
EXPLORE = 40
# Candidate budget, and here there is ONE operator, so this is also the total
# candidate list. The dual-field run in normalization.py has two operators and
# therefore roughly twice its TARGET_HITS in total, which is a confound in any
# comparison at equal TARGET_HITS. Run this at twice the dual-field setting to
# compare at matched or better total budget.
TARGET = int(os.environ.get("TARGET_HITS", "50"))

meta = V.load_meta(HERE)
doc_ids, doc_text, q_ids, relevant = meta["doc_ids"], meta["doc_text"], meta["q_ids"], meta["relevant"]
N = len(doc_ids)
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
q_new = np.load(os.path.join(CACHE, "emb_q_new.npy"))
index_of = {d: i for i, d in enumerate(doc_ids)}
order = np.random.default_rng(SEED).permutation(N)

YQL = ("select doc_id, space from scifact where "
       f"{V.annotation(target_hits=TARGET, explore=EXPLORE)}nearestNeighbor(emb_old, q)")
YQL_EXACT = ("select doc_id, space from scifact where "
             f"{V.annotation(target_hits=TARGET, exact=True)}nearestNeighbor(emb_old, q)")


def feed(ops, tag):
    if not ops:
        return 0.0
    with open(JSONL, "w") as f:
        for o in ops:
            f.write(json.dumps(o) + "\n")
    t0 = time.time()
    p = subprocess.run(["vespa", "feed", "-t", ENDPOINT, JSONL], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[:900], file=sys.stderr)
        raise SystemExit(f"{tag}: feed failed")
    return round(time.time() - t0, 2)


def ask(yql, ranking, i, hits, stats=None):
    import urllib.request
    body = {"yql": yql, "ranking": ranking, "hits": hits, "timeout": "20s",
            "input.query(q)": [float(x) for x in q_new[i]]}
    if stats:
        body.update({f"input.query({k})": v for k, v in stats.items()})
    req = urllib.request.Request(ENDPOINT + "/search/", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        root = json.load(r)["root"]
    if "errors" in root:
        raise SystemExit(f"{ranking}: " + json.dumps(root["errors"])[:400])
    return root.get("children", [])


def recall(yql, ranking, allstats):
    hits = tot = 0
    for i, qid in enumerate(q_ids):
        got = {h["fields"].get("doc_id")
               for h in ask(yql, ranking, i, V.K, allstats[i] if allstats else None)}
        rel = set(relevant[qid])
        hits += len(got & rel)
        tot += len(rel)
    return round(hits / tot, 4)


def stats_for(mask):
    """Per-query mean and std of cosine within each space. The field now holds a
    mix, so the old space is the un-migrated documents' old vectors and the new
    space is the migrated documents' new vectors."""
    mig, unmig = docs_new[mask], docs_old[~mask]
    out = []
    for i in range(len(q_ids)):
        sn = q_new[i] @ mig.T if mig.shape[0] else np.zeros(1, dtype=np.float32)
        so = q_new[i] @ unmig.T if unmig.shape[0] else np.zeros(1, dtype=np.float32)
        out.append({"mu_n": float(sn.mean()), "sd_n": float(sn.std()) or 1.0,
                    "mu_o": float(so.mean()), "sd_o": float(so.std()) or 1.0})
    return out


def candidates(target=None):
    """Total candidates the single operator exposes, averaged over queries. One
    graph and one operator, so this is the whole candidate list, and it is the
    number to match against the dual-field run's two operators combined."""
    target = target or TARGET
    n = lists = 0
    for i in range(len(q_ids)):
        ch = ask(YQL, "inplace_cosine", i, 4 * target)
        if ch:
            n += len(ch)
            lists += 1
    return round(n / lists, 2) if lists else None


def reach(mask, yql=None):
    """Share of relevant old-space documents in the single operator's candidates.

    Run against YQL this is the graph's reach. Run against YQL_EXACT it is the
    reach of a perfect search given the same budget, which is the decomposition
    that matters: exact search at targetHits 50 returns the true top 50 by score,
    so whatever it misses was lost to slot competition alone, with no graph
    involved. The gap between the two is what the graph itself costs.
    """
    yql = yql or YQL
    found = total = 0
    for i, qid in enumerate(q_ids):
        rel_old = {d for d in relevant[qid] if d in index_of and not mask[index_of[d]]}
        if not rel_old:
            continue
        got = {h["fields"].get("doc_id") for h in ask(yql, "inplace_cosine", i, TARGET)}
        found += len(got & rel_old)
        total += len(rel_old)
    return round(found / total, 4) if total else None


print(f"{N} documents, seed {SEED}, targetHits {TARGET}, one field overwritten in place\n", flush=True)
runs = []
prev = 0
for frac in FRACTIONS:
    n = int(frac * N)
    secs = feed([{"update": f"id:scifact:scifact::{doc_ids[j]}",
                  "fields": {"emb_old": {"assign": {"values": [float(x) for x in docs_new[j]]}},
                             "space": {"assign": 1}}} for j in order[prev:n]], "overwrite")
    prev = n
    mask = np.zeros(N, dtype=bool)
    mask[order[:n]] = True
    st = stats_for(mask)

    rec = {"fraction": frac, "migrated_docs": n, "overwrite_seconds": secs,
           "exact_raw": recall(YQL_EXACT, "inplace_cosine", None),
           "exact_normalized": recall(YQL_EXACT, "inplace_normalized", st),
           "graph_raw": recall(YQL, "inplace_cosine", None),
           "graph_normalized": recall(YQL, "inplace_normalized", st),
           "graph_reach_old_space_relevant": reach(mask),
           "exact_reach_old_space_relevant": reach(mask, YQL_EXACT),
           "mean_total_candidates": candidates(),
           "target_hits_single_operator": TARGET}
    runs.append(rec)
    print(f"  {frac:4.1f}  exact raw {rec['exact_raw']:.4f} -> norm {rec['exact_normalized']:.4f}"
          f"    graph raw {rec['graph_raw']:.4f} -> norm {rec['graph_normalized']:.4f}"
          f"    reach graph {rec['graph_reach_old_space_relevant']}"
          f" / exact {rec['exact_reach_old_space_relevant']}", flush=True)

res = {"step": "9b", "seed": SEED, "explore": EXPLORE, "target_hits": TARGET, "fractions": FRACTIONS,
       "migration_style": "in place: one tensor field overwritten, one HNSW graph holding both spaces",
       "compare_with": "results/step9_normalization.json, the same migration done as a second field",
       "stage3_seed0_hnsw_inplace": {"raw": [0.59, 0.3569, 0.3068, 0.3274],
                                     "norm": [0.59, 0.4425, 0.3746, 0.3894],
                                     "reach": [0.794, 0.5273, 0.3986, 0.3528]},
       "runs": runs}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")

print("restoring emb_old ...", flush=True)
t = feed([{"update": f"id:scifact:scifact::{d}",
           "fields": {"emb_old": {"assign": {"values": [float(x) for x in docs_old[i]]}},
                      "space": {"assign": 0}}} for i, d in enumerate(doc_ids)], "restore")
print(f"restored in {t}s")
