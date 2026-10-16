"""Step 5b: what does the backfill actually cost, and does it touch the document store?

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

HOW THE DOCUMENT STORE IS MEASURED, and why the first version of this script was
wrong. The obvious approach is to read the proton metric
document_store.disk_usage.average before and after each batch. That does not
work: those values come from a snapshot interval, so two reads a few seconds
apart return the same snapshot and every shape looks like it changed nothing.
Measured that way, put_full - 5,183 full document PUTs, 52 MB of payload, which
unquestionably rewrites the store - reported byte-identical usage before and
after. A measurement that cannot see that write cannot be trusted to prove the
absence of one.

So the store is measured directly instead. Proton keeps the document store as an
append-only log under 0.ready/summary, so a write that touches the store makes
those files grow and a write that does not leaves them alone. Each batch is
followed by a forced flush and then a wait until the byte count stops moving, so
nothing is read mid-write.

The script then checks its own instrument. put_full and update_text both have to
grow the store, the first because a PUT writes a whole new document and the
second because Vespa documents an indexed field update as a read-modify-write.
If either comes back as no growth, the measurement is broken rather than
interesting, and the script says so in the output and in positive_controls_pass.

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
STORE = os.environ.get(
    "VESPA_STORE",
    "/opt/vespa/var/db/vespa/search/cluster.scifact/n0/documents/scifact/0.ready/summary")

meta = V.load_meta(HERE)
doc_ids, doc_text = meta["doc_ids"], meta["doc_text"]
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
N = len(doc_ids)

METRICS = ("content.proton.documentdb.ready.document_store.cache.lookups.rate",
           "content.proton.documentdb.ready.document_store.cache.hit_rate.average",
           "content.proton.documentdb.ready.document_store.disk_usage.average",
           "content.proton.documentdb.ready.document_store.disk_bloat.average",
           "content.proton.documentdb.ready.document_store.memory_usage.allocated_bytes.average",
           "content.proton.documentdb.ready.attribute.memory_usage.allocated_bytes.average",
           "content.proton.documentdb.ready.lid_space.lid_limit.last")


def dexec(args, timeout=120):
    return subprocess.run(["docker", "exec", CONTAINER] + args,
                          capture_output=True, text=True, timeout=timeout)


def store_bytes():
    """Total size of the document store files. One `ls -l` so the whole listing
    comes from a single point in time."""
    p = dexec(["sh", "-c", f"ls -l {STORE}/ | awk '{{print $5, $9}}'"])
    total, files = 0, {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            total += int(parts[0])
            files[parts[1]] = int(parts[0])
    return total, files


def flush_and_settle(label, quiet_reads=2, interval=4.0, limit=90.0):
    """Force a flush, then wait for the store size to stop moving.

    Writes are acknowledged before the store has necessarily been flushed, so
    reading the byte count straight after a feed can catch it mid-write. This
    triggers a flush and then requires the same byte count on consecutive reads
    before believing it.
    """
    dexec(["vespa-proton-cmd", "--local", "triggerFlush"], timeout=90)
    t0 = time.time()
    last, stable = None, 0
    while time.time() - t0 < limit:
        time.sleep(interval)
        total, files = store_bytes()
        if total == last:
            stable += 1
            if stable >= quiet_reads:
                return total, files, round(time.time() - t0, 1)
        else:
            stable = 0
        last = total
    print(f"    warning: {label} store size still moving after {limit:.0f}s", flush=True)
    total, files = store_bytes()
    return total, files, round(time.time() - t0, 1)


def metrics():
    """Secondary evidence only. These are snapshot aggregates, so a value here
    can lag a write by a whole snapshot interval; the disk measurement above is
    what the conclusions rest on."""
    try:
        p = dexec(["curl", "-s", "http://localhost:19092/metrics/v2/values?consumer=Vespa"], 60)
        d = json.loads(p.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}
    out = {}
    for node in d.get("nodes", []):
        for svc in node.get("services", []):
            for m in svc.get("metrics", []):
                for k, v in m.get("values", {}).items():
                    if k in METRICS:
                        out[k] = max(v, out[k]) if k in out and isinstance(v, (int, float)) else v
    return out


def feed(ops, tag):
    with open(JSONL, "w") as f:
        for o in ops:
            f.write(json.dumps(o) + "\n")
    size = os.path.getsize(JSONL)
    before_bytes, before_files, _ = flush_and_settle(tag + " before")
    before_metrics = metrics()
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
    after_bytes, after_files, settle = flush_and_settle(tag + " after")
    after_metrics = metrics()
    grew = after_bytes - before_bytes
    rec = {"shape": tag, "ops": len(ops), "jsonl_bytes": size,
           "wall_seconds": round(wall, 2), "docs_per_second": round(len(ops) / wall, 1),
           "request_mbps": summary.get("http.request.MBps"),
           "latency_avg_ms": summary.get("http.response.latency.millis.avg"),
           "latency_max_ms": summary.get("http.response.latency.millis.max"),
           "errors": summary.get("feeder.error.count"),
           "document_store_bytes_before": before_bytes,
           "document_store_bytes_after": after_bytes,
           "document_store_growth_bytes": grew,
           "document_store_growth_bytes_per_doc": round(grew / len(ops), 1),
           "document_store_touched": grew > 0,
           "settle_seconds_after": settle,
           "document_store_files_after": after_files,
           "metrics_before": before_metrics, "metrics_after": after_metrics}
    print(f"  {tag:17s} {len(ops):5d} ops  {wall:6.2f}s  {rec['docs_per_second']:8.1f}/s  "
          f"{size/1e6:5.1f} MB payload  store {grew/1e6:+8.2f} MB "
          f"({rec['document_store_growth_bytes_per_doc']:+8.1f} B/doc)  err {rec['errors']}",
          flush=True)
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


print(f"{N} documents, endpoint {ENDPOINT}", flush=True)
print(f"document store: {STORE}\n", flush=True)
runs = [feed(put_full(), "put_full"),
        feed(upd_tensor(), "update_tensor"),
        feed(upd_int(), "update_int"),
        feed(upd_text(), "update_text"),
        feed(upd_tensor(), "update_tensor_2")]

by = {r["shape"]: r for r in runs}
controls = {"put_full_grew": by["put_full"]["document_store_touched"],
            "update_text_grew": by["update_text"]["document_store_touched"]}
controls["pass"] = all(controls.values())

res = {"step": "5b", "n_docs": N, "endpoint": ENDPOINT,
       "what": "cost of each write shape, one batch of all documents each",
       "document_store_path": STORE,
       "how_measured": "append-only document store files sized directly inside the "
                       "container, after a forced flush and a wait for the byte count "
                       "to stop moving. The proton disk_usage metric is a snapshot "
                       "aggregate and cannot see a write that just happened, so it is "
                       "recorded as secondary evidence only.",
       "positive_controls": controls,
       "positive_controls_pass": controls["pass"],
       "metric_names": list(METRICS), "runs": runs}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)

print("\nrelative to a full document PUT of the same documents:")
base = runs[0]["docs_per_second"]
for r in runs:
    print(f"  {r['shape']:17s} {r['docs_per_second']:8.1f}/s  "
          f"{r['docs_per_second']/base:5.2f}x  payload {r['jsonl_bytes']/1e6:5.1f} MB  "
          f"store {r['document_store_growth_bytes']/1e6:+7.2f} MB")

print("\npositive controls, the writes that MUST grow the document store:")
print(f"  put_full grew the store:    {controls['put_full_grew']}")
print(f"  update_text grew the store: {controls['update_text_grew']}")
if not controls["pass"]:
    print("  FAILED. The instrument missed a write that certainly happened, so no "
          "conclusion about an untouched document store is supported by this run.")
else:
    print("  passed, so a shape reported as no growth is evidence and not a blind spot.")
print(f"\nsaved {OUT}")
