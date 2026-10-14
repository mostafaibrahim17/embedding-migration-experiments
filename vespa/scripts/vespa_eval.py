"""Shared query and scoring helpers for the Vespa runs.

Pooled recall@10, defined exactly as in ../../scripts: total relevant documents
retrieved in the top k divided by total relevant documents, not the per-query
mean. Keeping it in one place means the backfill curve and the one-off checks
cannot drift apart.
"""
import json, os, statistics, time, urllib.request

K = 10


def endpoint():
    return os.environ.get("VESPA_ENDPOINT", "http://127.0.0.1:8080")


def annotation(target_hits=K, exact=False, explore=None, label=None):
    """Build a nearestNeighbor annotation.

    explore has to go here rather than into the query properties: as a property
    hnsw.exploreAdditionalHits is accepted and silently ignored. Vespa validates
    properties but not annotations, so neither form ever errors.
    """
    parts = [f"targetHits:{target_hits}"]
    if exact:
        parts.append("approximate:false")
    if explore is not None:
        parts.append(f"hnsw.exploreAdditionalHits:{explore}")
    if label:
        parts.append(f'label:"{label}"')
    return "{" + ", ".join(parts) + "}"


def yql_single(field, **kw):
    return f"select doc_id from scifact where {annotation(**kw)}nearestNeighbor({field}, q)"


def yql_mixed(**kw):
    """The mixed state: one query vector, two spaces, each document scored in the
    space it currently occupies."""
    n = annotation(label="n", **kw)
    o = annotation(label="o", **kw)
    return ("select doc_id from scifact where "
            f"({n}nearestNeighbor(emb_new, q) and space = 1) or "
            f"({o}nearestNeighbor(emb_old, q) and space = 0)")


def search(yql, ranking, vec, hits=K, base=None):
    body = {"yql": yql, "ranking": ranking, "hits": hits,
            "input.query(q)": [float(x) for x in vec], "timeout": "20s"}
    req = urllib.request.Request((base or endpoint()) + "/search/",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def pooled_recall(yql, ranking, qvecs, q_ids, relevant, k=K, base=None):
    hits = total = empty = 0
    lat = []
    for i, qid in enumerate(q_ids):
        t0 = time.time()
        r = search(yql, ranking, qvecs[i], hits=k, base=base)
        lat.append((time.time() - t0) * 1000)
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
    return {"recall_at_10_pooled": round(hits / total, 4),
            "retrieved_relevant": hits, "relevant_total": total,
            "queries_with_no_hits": empty,
            "latency_ms_p50": round(statistics.median(lat), 1),
            "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat))], 1)}


def load_meta(here):
    path = os.path.join(here, "ids.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}. Run prepare_embeddings.py first.")
    return json.load(open(path))
