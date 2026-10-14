"""Deploy check: feeds two documents, one migrated and one not, and runs the four
query shapes the migration run depends on. Expected output, with the query vector
pointing along the new-model axis:

    mixed:  B space=1 relevance=1.0   (scored in the new space)
            A space=0 relevance=0.5   (scored in the old space, orthogonal)
    old:    both documents, 1.0 each
    new:    B only, 1.0               (A has no new vector to match)
    exact:  same as old

The 1.0-against-0.5 split in the mixed profile is the point: a migrated document
and an un-migrated one are being ranked against each other on closeness values
computed in different embedding spaces.

Set VESPA_ENDPOINT if Vespa is not on the default port, e.g. when 8080 is taken.
Run it before the backfill, and again after, to confirm the cluster still answers.
"""
import json, os, urllib.request

BASE = os.environ.get("VESPA_ENDPOINT", "http://127.0.0.1:8080")

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def put(docid, fields):
    req = urllib.request.Request(f"{BASE}/document/v1/scifact/scifact/docid/{docid}",
                                 data=json.dumps({"fields": fields}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status

def unit(i):
    v = [0.0] * 384
    v[i] = 1.0
    return v

# A: never migrated, old vector only, space 0
print("feed A:", put("A", {"doc_id": "A", "text": "doc A unmigrated",
                           "space": 0, "emb_old": {"values": unit(0)}}))
# B: migrated, carries BOTH vectors, space 1
print("feed B:", put("B", {"doc_id": "B", "text": "doc B migrated",
                           "space": 1, "emb_old": {"values": unit(0)},
                           "emb_new": {"values": unit(1)}}))

YQL_MIXED = ('select doc_id, space from scifact where '
             '({targetHits:10, label:"n"}nearestNeighbor(emb_new, q) and space = 1) or '
             '({targetHits:10, label:"o"}nearestNeighbor(emb_old, q) and space = 0)')

def show(tag, body):
    r = post("/search/", body)
    root = r.get("root", {})
    if "errors" in root:
        print(f"\n{tag}: ERRORS"); print(json.dumps(root["errors"], indent=1)[:900]); return
    hits = root.get("children", [])
    print(f"\n{tag}: {root.get('fields',{}).get('totalCount')} hits")
    for h in hits:
        f = h.get("fields", {})
        print(f"   {f.get('doc_id')}  space={f.get('space')}  relevance={h.get('relevance')}")

# query vector = the NEW-model direction (unit(1)); B matches in new space, A only in old
show("mixed, q=new-space direction", {"yql": YQL_MIXED, "ranking": "mixed",
                                      "input.query(q)": unit(1), "hits": 10})
show("profile 'old' over emb_old",
     {"yql": 'select doc_id, space from scifact where {targetHits:10}nearestNeighbor(emb_old, q)',
      "ranking": "old", "input.query(q)": unit(0), "hits": 10})
show("profile 'new' over emb_new (only B has one)",
     {"yql": 'select doc_id, space from scifact where {targetHits:10}nearestNeighbor(emb_new, q)',
      "ranking": "new", "input.query(q)": unit(1), "hits": 10})
show("exact search, approximate:false",
     {"yql": 'select doc_id, space from scifact where {targetHits:10, approximate:false}nearestNeighbor(emb_old, q)',
      "ranking": "old", "input.query(q)": unit(0), "hits": 10})

for d in ("A", "B"):
    req = urllib.request.Request(f"{BASE}/document/v1/scifact/scifact/docid/{d}", method="DELETE")
    with urllib.request.urlopen(req) as r:
        print(f"cleanup {d}: {r.status}")
