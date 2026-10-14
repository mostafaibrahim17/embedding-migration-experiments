"""Step 3: load the pre-migration index.

Feeds all 5,183 SciFact documents with only emb_old populated and space=0, which
is the state a production index is in before anyone decides to change models.
Records feed throughput so the backfill later has something to be compared to.

Uses the Vespa CLI feed client rather than one HTTP request per document, because
that is what a real backfill would use and it is the number worth reporting.

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 feed_old_space.py
"""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
ENDPOINT = os.environ.get("VESPA_ENDPOINT", "http://127.0.0.1:8080")
IDS = os.path.join(HERE, "ids.json")
JSONL = os.path.join(HERE, "feed_old.jsonl")
OUT = os.path.join(HERE, "..", "results", "step3_feed.json")

if not os.path.exists(IDS):
    raise SystemExit(f"missing {IDS}. Run prepare_embeddings.py first.")
meta = json.load(open(IDS))
doc_ids, doc_text = meta["doc_ids"], meta["doc_text"]

path = os.path.join(CACHE, "emb_docs_old.npy")
if not os.path.exists(path):
    raise SystemExit(f"missing emb_docs_old.npy in {CACHE}. Run prepare_embeddings.py first.")
docs_old = np.load(path)
assert docs_old.shape[0] == len(doc_ids), (docs_old.shape, len(doc_ids))
print(f"{docs_old.shape[0]} documents, {docs_old.shape[1]} dimensions", flush=True)

t0 = time.time()
with open(JSONL, "w") as f:
    for i, did in enumerate(doc_ids):
        f.write(json.dumps({
            "put": f"id:scifact:scifact::{did}",
            "fields": {"doc_id": did, "text": doc_text[did], "space": 0,
                       "emb_old": {"values": [float(x) for x in docs_old[i]]}},
        }) + "\n")
size_mb = os.path.getsize(JSONL) / 1e6
print(f"wrote {JSONL} ({size_mb:.0f} MB) in {time.time()-t0:.0f}s", flush=True)

print(f"feeding to {ENDPOINT} ...", flush=True)
t0 = time.time()
proc = subprocess.run(["vespa", "feed", "-t", ENDPOINT, JSONL],
                      capture_output=True, text=True)
wall = time.time() - t0
print(proc.stdout.strip() or "(no stdout)")
if proc.returncode != 0:
    print(proc.stderr.strip()[:2000], file=sys.stderr)
    raise SystemExit(f"vespa feed failed with code {proc.returncode}")

summary = {}
try:
    summary = json.loads(proc.stdout[proc.stdout.index("{"):proc.stdout.rindex("}") + 1])
except ValueError:
    print("could not parse the feed client summary; wall time is still recorded")

res = {"step": 3, "what": "initial load, emb_old only, space=0",
       "n_docs": int(docs_old.shape[0]), "dim": int(docs_old.shape[1]),
       "jsonl_mb": round(size_mb, 1), "wall_seconds": round(wall, 1),
       "docs_per_second": round(docs_old.shape[0] / wall, 1),
       "feed_client_summary": summary, "endpoint": ENDPOINT}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\n{docs_old.shape[0]} documents in {wall:.0f}s = {res['docs_per_second']} docs/s")
print(f"saved {OUT}")
