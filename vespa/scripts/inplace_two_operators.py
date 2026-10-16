"""Step 9c: is it the second FIELD, or is it the second OPERATOR?

Step 9 found that per-space normalization repairs the low point on a graph when
the two embedding spaces sit in two fields, and step 9b found that it does not
when one field is overwritten in place. The explanation first given for that was
that two fields mean two nearestNeighbor operators, each with a candidate budget
of its own, so migrated documents cannot take slots away from the old space.

That explanation does not survive its own data. mixed_list_split in
results/step9_normalization.json records how many old-space documents the
dual-field query actually returns, and it is targetHits * (1 - migrated) at
every fraction: 50.0, 45.2, 40.0, 35.3, 30.5, 25.5, 20.4, 15.3, 10.5, 5.0. The
old-space operator retrieves targetHits candidates from the emb_old graph and
then loses the migrated ones to the `and space = 0` filter, so migrated
documents do spend its budget. Nothing is reserved.

Which leaves two candidate explanations that step 9 and 9b cannot tell apart,
because they differ in two ways at once:

  budget   the old space is protected because its operator has its own targetHits
  purity   the old space is protected because its FIELD holds only old-model
           vectors, so ranking inside it has no cross-space score competition
           and migrated documents can only dilute the list in proportion to
           their number, never outscore their way into it

This run separates them. One field, overwritten in place, exactly as in step 9b
so the mix is identical, but queried with TWO operators over that same field,
each filtered to its own space:

    ({targetHits:N}nearestNeighbor(emb_old, q) and space = 1) or
    ({targetHits:N}nearestNeighbor(emb_old, q) and space = 0)

That gives the old space an operator and a targetHits of its own while leaving
the field mixed. So:

  if reach returns to the dual-field level (~0.80)  -> budget
  if reach stays at the step 9b level (~0.44)       -> purity

Destructive: it overwrites emb_old. Re-run feed_old_space.py afterwards, which
this script does for you at the end.

    VESPA_ENDPOINT=http://127.0.0.1:8090 TARGET_HITS=50 python3 inplace_two_operators.py
"""
import json, os, subprocess, sys, time
import numpy as np
import vespa_eval as V

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
TAG = os.environ.get("OUT_TAG", "")
OUT = os.path.join(HERE, "..", "results", f"step9c_two_operators_one_field{TAG}.json")
JSONL = os.path.join(HERE, "twoop_slice.jsonl")
ENDPOINT = V.endpoint()
SEED = int(os.environ.get("SEED", "0"))
FRACTIONS = [round(float(x), 2) for x in
             os.environ.get("FRACTIONS", "0,0.1,0.2,0.3,0.4,0.5").split(",")]
EXPLORE = 40
TARGET = int(os.environ.get("TARGET_HITS", "50"))

meta = V.load_meta(HERE)
doc_ids, q_ids, relevant = meta["doc_ids"], meta["q_ids"], meta["relevant"]
N = len(doc_ids)
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
q_new = np.load(os.path.join(CACHE, "emb_q_new.npy"))
index_of = {d: i for i, d in enumerate(doc_ids)}
order = np.random.default_rng(SEED).permutation(N)

A = V.annotation(target_hits=TARGET, explore=EXPLORE)
A_EXACT = V.annotation(target_hits=TARGET, exact=True)
# two operators, one field, each filtered to a space
YQL = ("select doc_id, space from scifact where "
       f"({A}nearestNeighbor(emb_old, q) and space = 1) or "
       f"({A}nearestNeighbor(emb_old, q) and space = 0)")
YQL_EXACT = ("select doc_id, space from scifact where "
             f"({A_EXACT}nearestNeighbor(emb_old, q) and space = 1) or "
             f"({A_EXACT}nearestNeighbor(emb_old, q) and space = 0)")
# the old-space operator on its own, which is what decides reach
YQL_OLD_ONLY = ("select doc_id, space from scifact where "
                f"{A}nearestNeighbor(emb_old, q) and space = 0")
YQL_OLD_ONLY_EXACT = ("select doc_id, space from scifact where "
                      f"{A_EXACT}nearestNeighbor(emb_old, q) and space = 0")


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
    """Per-query mean and std of cosine within each space. One field, so the old
    space is the un-migrated documents' old vectors and the new space is the
    migrated documents' new vectors. Same oracle as step 9 and 9b."""
    mig, unmig = docs_new[mask], docs_old[~mask]
    out = []
    for i in range(len(q_ids)):
        sn = q_new[i] @ mig.T if mig.shape[0] else np.zeros(1, dtype=np.float32)
        so = q_new[i] @ unmig.T if unmig.shape[0] else np.zeros(1, dtype=np.float32)
        out.append({"mu_n": float(sn.mean()), "sd_n": float(sn.std()) or 1.0,
                    "mu_o": float(so.mean()), "sd_o": float(so.std()) or 1.0})
    return out


def reach(mask, yql):
    """Share of each query's relevant old-space documents that the old-space
    operator returns, pooled over queries (relevant found / relevant total)."""
    found = total = 0
    for i, qid in enumerate(q_ids):
        rel_old = {d for d in relevant[qid] if d in index_of and not mask[index_of[d]]}
        if not rel_old:
            continue
        got = {h["fields"].get("doc_id") for h in ask(yql, "inplace_cosine", i, TARGET)}
        found += len(got & rel_old)
        total += len(rel_old)
    return round(found / total, 4) if total else None


def split():
    """How the two-operator list divides by space, and how big it is. This is the
    number that says whether the old-space operator keeps its own budget or
    spends it on migrated documents it then filters away."""
    new_n = old_n = lists = 0
    for i in range(len(q_ids)):
        ch = ask(YQL, "inplace_cosine", i, 4 * TARGET)
        if not ch:
            continue
        new_n += sum(1 for h in ch if h["fields"].get("space") == 1)
        old_n += sum(1 for h in ch if h["fields"].get("space") == 0)
        lists += 1
    return {"mean_new_space_hits": round(new_n / lists, 2),
            "mean_old_space_hits": round(old_n / lists, 2),
            "mean_total_candidates": round((new_n + old_n) / lists, 2),
            "target_hits_per_operator": TARGET} if lists else None


print(f"{N} documents, seed {SEED}, targetHits {TARGET} per operator", flush=True)
print("one field overwritten in place, queried with two filtered operators\n", flush=True)
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
           "graph_reach_old_space_relevant": reach(mask, YQL_OLD_ONLY),
           "exact_reach_old_space_relevant": reach(mask, YQL_OLD_ONLY_EXACT),
           "mixed_list_split": split()}
    runs.append(rec)
    s = rec["mixed_list_split"]
    print(f"  {frac:4.1f}  graph raw {rec['graph_raw']:.4f} -> norm {rec['graph_normalized']:.4f}"
          f"    reach graph {rec['graph_reach_old_space_relevant']}"
          f" / exact {rec['exact_reach_old_space_relevant']}"
          f"    list {s['mean_new_space_hits']}n/{s['mean_old_space_hits']}o", flush=True)

res = {"step": "9c", "seed": SEED, "explore": EXPLORE, "target_hits": TARGET,
       "fractions": FRACTIONS, "endpoint": ENDPOINT,
       "question": "two operators over ONE mixed field: does giving the old space its "
                   "own operator and targetHits restore its reach, or does protection "
                   "require a field that holds only old-space vectors?",
       "compare_with": {
           "two fields, two operators": "results/step9_normalization.json",
           "one field, one operator": "results/step9b_inplace_vs_dualfield.json"},
       "runs": runs}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")

print("restoring emb_old ...", flush=True)
t = feed([{"update": f"id:scifact:scifact::{d}",
           "fields": {"emb_old": {"assign": {"values": [float(x) for x in docs_old[i]]}},
                      "space": {"assign": 0}}} for i, d in enumerate(doc_ids)], "restore")
print(f"restored in {t}s")
