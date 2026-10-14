"""Step 5b: what does the backfill actually cost?

The article claims Vespa's attribute updates can avoid the document store
read-modify-write, which is the reason it gives for preferring dual-write on
Vespa over a full reindex. Vespa's own documentation says most attribute fields
avoid that read and that index fields require it, but it does not say which
applies to a tensor attribute that also carries an HNSW index. That is exactly
the field this migration updates, so the claim needs measuring.

Five write shapes, each one batch of all 5,183 documents through the same feed
client, so batch size and client warm-up are equal and the throughputs are
comparable:

  put_full          full document PUT, old vector only        (what the load does)
  update_tensor     assign emb_new + space, first time        (the backfill: inserts
                                                               5,183 vectors into the
                                                               second HNSW graph)
  update_int        assign space only                         (pure small attribute)
  update_text       assign text, unchanged value              (indexed field, the
                                                               documented read-modify-write)
  update_tensor_2   assign emb_new a second time              (graph update, not insert)

Document store cache lookups are read before and after each shape. A
read-modify-write has to fetch the document from the store, so it shows up there.

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 update_cost.py
"""
import json, os, subprocess, sys, time
import numpy as np
import vespa_eval as V

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
OUT = os.path.join(HERE, "..", "results", "step5b_update_cost.json")
JSONL = os.path.join(HERE, "cost_batch.jsonl")
ENDPOINT = V.endpoint()
CONTAINER = os.environ.get("VESPA_CONTAINER", "vespa")

meta = V.load_meta(HERE)
doc_ids, doc_text = meta["doc_ids"], meta["doc_text"]
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
N = len(doc_ids)

METRICS = ("content.proton.documentdb.ready.document_store.cache.lookups.rate",
           "content.proton.documentdb.ready.document_store.cache.hit_rate.average",
           "content.proton.documentdb.ready.document_store.disk_usage.average",
           "content.proton.documentdb.ready.document_store.memory_usage.allocated_bytes.average",
           "content.proton.documentdb.ready.attribute.memory_usage.allocated_bytes.average",
           "content.proton.documentdb.ready.lid_space.lid_limit.last")


def metrics():
    """Pull the proton metrics that say whether the document store was touched."""
    try:
        p = subprocess.run(["docker", "exec", CONTAINER, "curl", "-s",
                            "http://localhost:19092/metrics/v2/values?consumer=Vespa"],
                           capture_output=True, text=True, timeout=60)
        d = json.loads(p.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}
    out = {}
    for node in d.get("nodes", []):
        for svc in node.get("services", []):
            for m in svc.get("metrics", []):
                for k, v in m.get("values", {}).items():
                    if k in METRICS:
                        out[k] = v
    return out


def feed(ops, tag):
    with open(JSONL, "w") as f:
        for o in ops:
            f.write(json.dumps(o) + "\n")
    size = os.path.getsize(JSONL)
    before = metrics()
    t0 = time.time()
    p = subprocess.run(["vespa", "feed", "-t", ENDPOINT, JSONL], capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        print(p.stderr[:1200], file=sys.stderr)
        raise SystemExit(f"{tag}: vespa feed failed ({p.returncode})")
    summary = {}
    try:
        summary = json.loads(p.stdout[p.stdout.index("{"):p.stdout.rindex("}") + 1])
    except ValueError:
        pass
    after = metrics()
    rec = {"shape": tag, "ops": len(ops), "jsonl_bytes": size,
           "wall_seconds": round(wall, 2), "docs_per_second": round(len(ops) / wall, 1),
           "request_mbps": summary.get("http.request.MBps"),
           "latency_avg_ms": summary.get("http.response.latency.millis.avg"),
           "latency_max_ms": summary.get("http.response.latency.millis.max"),
           "errors": summary.get("feeder.error.count"),
           "metrics_before": before, "metrics_after": after}
    print(f"  {tag:17s} {len(ops):5d} ops  {wall:6.2f}s  {rec['docs_per_second']:8.1f}/s  "
          f"{size/1e6:5.1f} MB  avg {rec['latency_avg_ms']}ms  err {rec['errors']}", flush=True)
    return rec


def put_full():
    return [{"put": f"id:scifact:scifact::{d}",
             "fields": {"doc_id": d, "text": doc_text[d], "space": 0,
                        "emb_old": {"values": [float(x) for x in docs_old[i]]}}}
            for i, d in enumerate(doc_ids)]


def upd_tensor():
    return [{"update": f"id:scifact:scifact::{d}",
             "fields": {"emb_new": {"assign": {"values": [float(x) for x in docs_new[i]]}},
                        "space": {"assign": 1}}}
            for i, d in enumerate(doc_ids)]


def upd_int():
    return [{"update": f"id:scifact:scifact::{d}", "fields": {"space": {"assign": 1}}}
            for d in doc_ids]


def upd_text():
    return [{"update": f"id:scifact:scifact::{d}", "fields": {"text": {"assign": doc_text[d]}}}
            for d in doc_ids]


print(f"{N} documents, endpoint {ENDPOINT}\n", flush=True)
runs = [feed(put_full(), "put_full"),
        feed(upd_tensor(), "update_tensor"),
        feed(upd_int(), "update_int"),
        feed(upd_text(), "update_text"),
        feed(upd_tensor(), "update_tensor_2")]

res = {"step": "5b", "n_docs": N, "endpoint": ENDPOINT,
       "what": "cost of each write shape, one batch of all documents each",
       "metric_names": list(METRICS), "runs": runs}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)

print("\nrelative to a full document PUT of the same documents:")
base = runs[0]["docs_per_second"]
for r in runs:
    print(f"  {r['shape']:17s} {r['docs_per_second']:8.1f}/s  "
          f"{r['docs_per_second']/base:5.2f}x  payload {r['jsonl_bytes']/1e6:5.1f} MB")
print(f"\nsaved {OUT}")
