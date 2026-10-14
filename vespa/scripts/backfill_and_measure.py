"""Steps 5 and 6: backfill emb_new a slice at a time and query the mixed state.

Starts from the index feed_old_space.py leaves behind: every document carrying
only the old model's vector, space=0. Then, following the same migration order
stage 3 uses for the given seed, partially updates documents in increments of
10%, assigning emb_new and flipping space to 1.

After each increment it asks the same 300 new-model queries against the mixed
index, scoring each document in whichever space it currently occupies. The
resulting curve is directly comparable to exact_curves_minilm in
../../results/stage3_results.json, because the document order, the queries, the
relevance judgements and the metric are the same.

It also records what the backfill costs: update throughput per increment against
the 794 documents/second the initial load managed, and query latency measured
while the index is half in one space and half in the other.

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 backfill_and_measure.py
"""
import json, os, subprocess, sys, time
import numpy as np
import vespa_eval as V

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
SEED = int(os.environ.get("SEED", "0"))
FRACTIONS = [round(f, 1) for f in np.linspace(0, 1, 11)]
SLICE = os.path.join(HERE, "backfill_slice.jsonl")
OUT = os.path.join(HERE, "..", "results", "step5_backfill.json")
EXPLORE = 40          # ~ef_search 50, the setting stage 3 uses
ENDPOINT = V.endpoint()

meta = V.load_meta(HERE)
doc_ids, q_ids, relevant = meta["doc_ids"], meta["q_ids"], meta["relevant"]
N = len(doc_ids)
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
q_new = np.load(os.path.join(CACHE, "emb_q_new.npy"))
assert docs_new.shape[0] == N and q_new.shape[0] == len(q_ids)

order = np.random.default_rng(SEED).permutation(N)   # same order as stage 3, same seed
print(f"{N} documents, seed {SEED}, explore {EXPLORE}, endpoint {ENDPOINT}", flush=True)


def update_slice(idxs):
    """Partial update: assign emb_new and flip space. Nothing else is touched."""
    with open(SLICE, "w") as f:
        for j in idxs:
            f.write(json.dumps({
                "update": f"id:scifact:scifact::{doc_ids[j]}",
                "fields": {"emb_new": {"assign": {"values": [float(x) for x in docs_new[j]]}},
                           "space": {"assign": 1}},
            }) + "\n")
    t0 = time.time()
    p = subprocess.run(["vespa", "feed", "-t", ENDPOINT, SLICE], capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        print(p.stderr[:1500], file=sys.stderr)
        raise SystemExit(f"vespa feed failed ({p.returncode})")
    summary = {}
    try:
        summary = json.loads(p.stdout[p.stdout.index("{"):p.stdout.rindex("}") + 1])
    except ValueError:
        pass
    return wall, summary


runs = []
prev = 0
for frac in FRACTIONS:
    n = int(frac * N)
    wall, summary = (0.0, {}) if n == prev else update_slice(order[prev:n])
    updated = n - prev
    prev = n

    graph = V.pooled_recall(V.yql_mixed(explore=EXPLORE), "mixed", q_new, q_ids, relevant)
    exact = V.pooled_recall(V.yql_mixed(exact=True), "mixed", q_new, q_ids, relevant)

    rec = {"fraction": frac, "migrated_docs": n,
           "updated_this_step": updated,
           "update_wall_seconds": round(wall, 2),
           "update_docs_per_second": round(updated / wall, 1) if wall > 0 else None,
           "feed_client_summary": summary,
           "graph_recall": graph["recall_at_10_pooled"],
           "exact_recall": exact["recall_at_10_pooled"],
           "graph_latency_p50_ms": graph["latency_ms_p50"],
           "graph_latency_p95_ms": graph["latency_ms_p95"],
           "graph_queries_with_no_hits": graph["queries_with_no_hits"],
           "graph_detail": graph, "exact_detail": exact}
    runs.append(rec)
    print(f"  {frac:4.1f}  migrated {n:5d}  +{updated:5d} in {wall:5.2f}s"
          f"  ({rec['update_docs_per_second'] or '-':>7}/s)"
          f"   exact {exact['recall_at_10_pooled']:.4f}"
          f"   graph {graph['recall_at_10_pooled']:.4f}"
          f"   p50 {graph['latency_ms_p50']:.1f}ms", flush=True)

res = {"step": "5+6", "seed": SEED, "explore": EXPLORE, "n_docs": N,
       "queries": len(q_ids), "fractions": FRACTIONS, "endpoint": ENDPOINT,
       "initial_load_docs_per_second_cold": 793.9,
       "initial_load_docs_per_second_warm": 1658.5,
       "runs": runs}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)

ex = [r["exact_recall"] for r in runs]
gr = [r["graph_recall"] for r in runs]
print(f"\nexact: {' '.join(f'{v:.3f}' for v in ex)}")
print(f"graph: {' '.join(f'{v:.3f}' for v in gr)}")
print(f"exact low point {min(ex):.3f} at {FRACTIONS[ex.index(min(ex))]:.0%} migrated")
print(f"graph low point {min(gr):.3f} at {FRACTIONS[gr.index(min(gr))]:.0%} migrated")
print(f"saved {OUT}")
