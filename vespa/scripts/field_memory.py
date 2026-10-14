"""What the second embedding field costs, per field, from proton's own metrics.

The article says a dual-write migration on Vespa avoids copying the document
payload but still pays for the second vector and its index. This prints that
bill. Run it once before the backfill and once after to see emb_new appear.

    VESPA_CONTAINER=vespa python3 field_memory.py
"""
import json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "step5c_field_memory.json")
CONTAINER = os.environ.get("VESPA_CONTAINER", "vespa")
URL = "http://localhost:19092/metrics/v2/values?consumer=Vespa"

p = subprocess.run(["docker", "exec", CONTAINER, "curl", "-s", URL],
                   capture_output=True, text=True, timeout=60)
if p.returncode != 0 or not p.stdout.strip():
    raise SystemExit(f"could not read metrics from container {CONTAINER!r}: "
                     f"{(p.stderr or 'empty response')[:300]}")
d = json.loads(p.stdout)

fields, store = {}, {}
for node in d.get("nodes", []):
    for svc in node.get("services", []):
        for m in svc.get("metrics", []):
            fld = m.get("dimensions", {}).get("field")
            for k, v in m.get("values", {}).items():
                if "ready.attribute.memory_usage" in k and fld:
                    which = k.rsplit(".", 2)[-2]          # allocated_bytes / used_bytes / dead_bytes
                    fields.setdefault(fld, {})[which] = v
                elif "ready.document_store" in k and ("disk_usage" in k or "memory_usage" in k):
                    # The same metric arrives from several services, some reporting
                    # zero, so keep the largest rather than whichever lands last.
                    key = k.split("ready.document_store.")[-1]
                    store[key] = max(store.get(key, 0), v or 0)

res = {"step": "5c", "attribute_memory_bytes_per_field": fields,
       "document_store": store,
       "note": "Proton metrics are snapshot aggregates, so read them after the writes settle."}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)

print("attribute memory per field:")
for f, v in sorted(fields.items(), key=lambda x: -x[1].get("used_bytes", 0)):
    used = v.get("used_bytes", 0) / 1e6
    alloc = v.get("allocated_bytes", 0) / 1e6
    print(f"  {f:20s} used {used:7.2f} MB   allocated {alloc:7.2f} MB")
ds = store.get("disk_usage.average")
if ds is not None:
    print(f"\ndocument store on disk: {ds/1e6:.2f} MB")
print(f"saved {OUT}")
