"""Pooled recall@10 against a live Vespa cluster.

The metric is the one every script in ../../scripts uses: total relevant
documents retrieved in the top 10 divided by total relevant documents, not the
per-query mean. Same queries, same relevance judgements, same document order, so
a number from here is directly comparable to the hnswlib runs.

Three query shapes, matching the three rank profiles:

  old     old-model queries against emb_old      the pre-migration baseline
  new     new-model queries against emb_new      the finished index
  mixed   new-model queries against both spaces, each document scored in the one
          it currently occupies, through two labelled nearestNeighbor operators

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 eval_recall.py --profile old
    ... --profile mixed --exact        # approximate:false, for a reference number
    ... --profile old --explore 150    # hnsw.exploreAdditionalHits

Prints recall and latency. With --out it appends a labelled record to a JSON file.
"""
import argparse, json, os, statistics, time, urllib.error, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
ENDPOINT = os.environ.get("VESPA_ENDPOINT", "http://127.0.0.1:8080")
K = 10

ap = argparse.ArgumentParser()
ap.add_argument("--profile", choices=["old", "new", "mixed"], required=True)
ap.add_argument("--exact", action="store_true", help="approximate:false")
ap.add_argument("--explore", type=int, default=None,
                help="hnsw.exploreAdditionalHits, as a YQL annotation")
ap.add_argument("--targethits", type=int, default=K)
ap.add_argument("--out", default=None, help="append the result to this JSON file")
ap.add_argument("--label", default=None, help="name this run inside --out")
args = ap.parse_args()

meta = json.load(open(os.path.join(HERE, "ids.json")))
q_ids, relevant = meta["q_ids"], meta["relevant"]

# The mixed state asks a new-model query; the pre-migration baseline asks an old one.
qfile = "emb_q_old.npy" if args.profile == "old" else "emb_q_new.npy"
qv = np.load(os.path.join(CACHE, qfile))
assert qv.shape[0] == len(q_ids), (qv.shape, len(q_ids))

ann = [f"targetHits:{args.targethits}"]
if args.exact:
    ann.append("approximate:false")
if args.explore is not None:
    # This has to be a YQL annotation on the operator. Passing
    # hnsw.exploreAdditionalHits as a query property is accepted and then
    # silently ignored: recall does not move at any value. Measured, not assumed.
    ann.append(f"hnsw.exploreAdditionalHits:{args.explore}")
A = "{" + ", ".join(ann) + "}"

if args.profile == "mixed":
    yql = ("select doc_id from scifact where "
           f'({A[:-1]}, label:"n"}}nearestNeighbor(emb_new, q) and space = 1) or '
           f'({A[:-1]}, label:"o"}}nearestNeighbor(emb_old, q) and space = 0)')
else:
    field = "emb_old" if args.profile == "old" else "emb_new"
    yql = f"select doc_id from scifact where {A}nearestNeighbor({field}, q)"


def search(vec):
    body = {"yql": yql, "ranking": args.profile, "hits": K,
            "input.query(q)": [float(x) for x in vec], "timeout": "10s"}
    req = urllib.request.Request(ENDPOINT + "/search/", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)


hits = total = 0
latencies = []
empty = 0
print(f"{args.profile} profile, {len(q_ids)} queries, "
      f"{'exact' if args.exact else 'approximate'}"
      f"{f', explore {args.explore}' if args.explore is not None else ''}", flush=True)
t_start = time.time()
for i, qid in enumerate(q_ids):
    t0 = time.time()
    r = search(qv[i])
    latencies.append((time.time() - t0) * 1000)
    root = r.get("root", {})
    if "errors" in root:
        raise SystemExit("query error: " + json.dumps(root["errors"])[:600])
    got = {h.get("fields", {}).get("doc_id") for h in root.get("children", [])}
    got.discard(None)
    if not got:
        empty += 1
    rel = set(relevant[qid])
    hits += len(got & rel)
    total += len(rel)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(q_ids)} queries, recall so far {hits/total:.4f}", flush=True)

recall = hits / total
res = {"label": args.label or args.profile, "profile": args.profile,
       "exact": args.exact, "explore": args.explore, "targethits": args.targethits,
       "queries": len(q_ids), "relevant_total": total, "retrieved_relevant": hits,
       "recall_at_10_pooled": round(recall, 4), "queries_with_no_hits": empty,
       "latency_ms_p50": round(statistics.median(latencies), 1),
       "latency_ms_p95": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
       "wall_seconds": round(time.time() - t_start, 1), "endpoint": ENDPOINT}
print(json.dumps(res, indent=1))

if args.out:
    path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    runs = json.load(open(path)) if os.path.exists(path) else []
    runs.append(res)
    json.dump(runs, open(path, "w"), indent=1)
    print(f"appended to {path}")
