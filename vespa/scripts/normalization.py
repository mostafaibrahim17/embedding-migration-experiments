"""Step 9: per-space normalization as a Vespa rank expression.

The simulation repairs the low point by z-scoring each space's scores separately
before combining the rankings, and finds that this works in exact search and
mostly fails on a graph, because rescoring cannot recover a document the graph
never returned. This runs the same fix inside Vespa.

The mechanics are the interesting part. A first-phase expression is evaluated one
document at a time and cannot see the distribution it belongs to, so the per-space
mean and standard deviation are computed per query by the caller and passed as
query inputs. The expression then z-scores each document inside its own space:

    if (attribute(space) == 1,
        (cos_new() - query(mu_n)) / query(sd_n),
        (cos_old() - query(mu_o)) / query(sd_o))

It ranks on a dot product rather than closeness() because the vectors are unit
length, so the dot product is the cosine, which is the quantity the numpy runs
normalize. Comparing anything else would not be comparing the same fix.

The statistics are computed over every document in each space, which is the same
oracle the simulation's exact-search normalization uses. That is generous to the
graph case, where the simulation only normalizes the candidates it got back, and
it is the right way round: if normalization fails on the graph even with perfect
distribution knowledge, the failure is about reach rather than about scaling.

Starts from a pre-migration index and migrates to each fraction in turn.

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 normalization.py
"""
import json, os, subprocess, sys, time
import numpy as np
import vespa_eval as V

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
TAG = os.environ.get("OUT_TAG", "")
OUT = os.path.join(HERE, "..", "results", f"step9_normalization{TAG}.json")
JSONL = os.path.join(HERE, "norm_slice.jsonl")
ENDPOINT = V.endpoint()
SEED = int(os.environ.get("SEED", "0"))
# All eleven fractions by default, so the curve covers the whole migration rather
# than the first third. FRACTIONS=0,0.1,0.2,0.3 reproduces the shorter early run.
FRACTIONS = [round(float(x), 2) for x in
             os.environ.get("FRACTIONS", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0").split(",")]
EXPLORE = 40
# Candidate budget PER OPERATOR, and the mixed query has two of them, so the total
# candidate list here is about twice TARGET. That matters when comparing against
# the single-operator in-place run, which gets TARGET candidates in total: see
# inplace_vs_dualfield.py and the budget-matched controls in ../README.md.
# Normalization can only reorder what came back, so with TARGET equal to k it
# cannot change recall@10 at all. stage 3 reranks ef_search 50 candidates.
TARGET = int(os.environ.get("TARGET_HITS", "50"))
# targetHits is what widens an ANN candidate list, so reach is measured with the
# same budget the old-space operator gets inside the mixed query.

meta = V.load_meta(HERE)
doc_ids, q_ids, relevant = meta["doc_ids"], meta["q_ids"], meta["relevant"]
N = len(doc_ids)
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
q_new = np.load(os.path.join(CACHE, "emb_q_new.npy"))
index_of = {d: i for i, d in enumerate(doc_ids)}
order = np.random.default_rng(SEED).permutation(N)


def feed_slice(idxs):
    if len(idxs) == 0:
        return 0.0
    with open(JSONL, "w") as f:
        for j in idxs:
            f.write(json.dumps({
                "update": f"id:scifact:scifact::{doc_ids[j]}",
                "fields": {"emb_new": {"assign": {"values": [float(x) for x in docs_new[j]]}},
                           "space": {"assign": 1}}}) + "\n")
    t0 = time.time()
    p = subprocess.run(["vespa", "feed", "-t", ENDPOINT, JSONL], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[:900], file=sys.stderr)
        raise SystemExit("feed failed")
    return round(time.time() - t0, 2)


def space_stats(mask):
    """Per-query mean and std of cosine within each space, over all documents."""
    mig, unmig = docs_new[mask], docs_old[~mask]
    out = []
    for i in range(len(q_ids)):
        sn = q_new[i] @ mig.T if mig.shape[0] else np.zeros(1, dtype=np.float32)
        so = q_new[i] @ unmig.T if unmig.shape[0] else np.zeros(1, dtype=np.float32)
        out.append({"mu_n": float(sn.mean()), "sd_n": float(sn.std()) or 1.0,
                    "mu_o": float(so.mean()), "sd_o": float(so.std()) or 1.0})
    return out


def run(profile, stats, exact):
    yql = V.yql_mixed(target_hits=TARGET, exact=exact, explore=None if exact else EXPLORE)
    hits = tot = 0
    lat = []
    for i, qid in enumerate(q_ids):
        body = {"yql": yql, "ranking": profile, "hits": V.K, "timeout": "20s",
                "input.query(q)": [float(x) for x in q_new[i]]}
        if profile == "mixed_normalized":
            body.update({f"input.query({k})": v for k, v in stats[i].items()})
        import urllib.request
        req = urllib.request.Request(ENDPOINT + "/search/", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req) as r:
            d = json.load(r)
        lat.append((time.time() - t0) * 1000)
        root = d.get("root", {})
        if "errors" in root:
            raise SystemExit(f"{profile}: " + json.dumps(root["errors"])[:500])
        got = {h.get("fields", {}).get("doc_id") for h in root.get("children", [])}
        rel = set(relevant[qid])
        hits += len(got & rel)
        tot += len(rel)
    return round(hits / tot, 4), round(float(np.median(lat)), 1)


def _hits(yql, ranking, i, hits, stats=None):
    import urllib.request
    body = {"yql": yql, "ranking": ranking, "hits": hits, "timeout": "20s",
            "input.query(q)": [float(x) for x in q_new[i]]}
    if stats:
        body.update({f"input.query({k})": v for k, v in stats.items()})
    req = urllib.request.Request(ENDPOINT + "/search/", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)["root"].get("children", [])


def reach(mask, target=None):
    """Share of each query's relevant OLD-space documents that the old-space
    operator returns in its own candidate list.

    This is the quantity that decides whether rescoring can help: normalization
    can only promote a document that came back. It is measured on the old-space
    operator alone, with the same targetHits budget it gets inside the mixed
    query, because that is the budget it actually has. targetHits, not hits, is
    what widens an ANN candidate list; asking for more hits than targetHits
    returns fewer results, not a wider search.
    """
    target = target or TARGET
    yql = ("select doc_id from scifact where "
           f'{V.annotation(target_hits=target, explore=EXPLORE, label="o")}'
           "nearestNeighbor(emb_old, q) and space = 0")
    found = total = 0
    for i, qid in enumerate(q_ids):
        rel_old = {d for d in relevant[qid] if d in index_of and not mask[index_of[d]]}
        if not rel_old:
            continue
        got = {h.get("fields", {}).get("doc_id") for h in _hits(yql, "mixed_cosine", i, target)}
        found += len(got & rel_old)
        total += len(rel_old)
    return round(found / total, 4) if total else None


def space_split(mask):
    """How a mixed candidate list divides between the two spaces, averaged over
    queries. Each operator has its own targetHits budget, so the new space cannot
    crowd the old one out of retrieval however much better it scores."""
    yql = V.yql_mixed(target_hits=TARGET, explore=EXPLORE)
    new_n = old_n = lists = 0
    for i in range(len(q_ids)):
        ch = _hits(yql, "mixed_cosine", i, 4 * TARGET)
        if not ch:
            continue
        new_n += sum(1 for h in ch if h["fields"].get("space") == 1)
        old_n += sum(1 for h in ch if h["fields"].get("space") == 0)
        lists += 1
    return {"mean_new_space_hits": round(new_n / lists, 2),
            "mean_old_space_hits": round(old_n / lists, 2),
            "mean_total_candidates": round((new_n + old_n) / lists, 2),
            "target_hits_per_operator": TARGET} if lists else None


print(f"{N} documents, seed {SEED}, explore {EXPLORE}, targetHits {TARGET}, endpoint {ENDPOINT}", flush=True)
print("expects a pre-migration index; run feed_old_space.py first\n", flush=True)
runs = []
prev = 0
for frac in FRACTIONS:
    n = int(frac * N)
    secs = feed_slice(order[prev:n])
    prev = n
    mask = np.zeros(N, dtype=bool)
    mask[order[:n]] = True
    stats = space_stats(mask)

    raw_e, lat_re = run("mixed_cosine", stats, exact=True)
    nrm_e, lat_ne = run("mixed_normalized", stats, exact=True)
    raw_g, lat_rg = run("mixed_cosine", stats, exact=False)
    nrm_g, lat_ng = run("mixed_normalized", stats, exact=False)
    rch = reach(mask)
    split = space_split(mask)

    rec = {"fraction": frac, "migrated_docs": n, "update_seconds": secs,
           "exact_raw": raw_e, "exact_normalized": nrm_e,
           "graph_raw": raw_g, "graph_normalized": nrm_g,
           "graph_reach_old_space_relevant": rch,
           "mixed_list_split": split,
           "latency_p50_ms": {"exact_raw": lat_re, "exact_norm": lat_ne,
                              "graph_raw": lat_rg, "graph_norm": lat_ng}}
    runs.append(rec)
    print(f"  {frac:4.1f}  exact raw {raw_e:.4f} -> norm {nrm_e:.4f}    "
          f"graph raw {raw_g:.4f} -> norm {nrm_g:.4f}    reach {rch}"
          f"    list {split['mean_new_space_hits']}n/{split['mean_old_space_hits']}o"
          f" = {split['mean_total_candidates']}", flush=True)

res = {"step": 9, "seed": SEED, "explore": EXPLORE, "target_hits": TARGET, "fractions": FRACTIONS,
       "endpoint": ENDPOINT,
       "normalization": "per-query z-score within each space, statistics over all documents, "
                        "passed as query inputs and applied in first-phase",
       "stage3_seed0_reference": {
           "exact_raw": [0.59, 0.3658, 0.3156, 0.3481],
           "exact_norm": [0.59, 0.6047, 0.587, 0.5959],
           "hnsw_inplace_raw": [0.59, 0.3569, 0.3068, 0.3274],
           "hnsw_inplace_norm": [0.59, 0.4425, 0.3746, 0.3894],
           "hnsw_inplace_reach": [0.794, 0.5273, 0.3986, 0.3528]},
       "runs": runs}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")
