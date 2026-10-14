"""Steps 7 and 8: the coordinated switch, and what rollback actually costs.

Runs against a fully migrated index, the state backfill_and_measure.py leaves
behind. Four questions:

  1. At 100% migrated, does the mixed profile agree with the new one? If it does,
     the switch is a no-op for quality and can be made whenever it is convenient.
  2. What does switching reads cost when both rank profiles are already deployed?
     The claim is that it is a query parameter rather than a deploy.
  3. What would it cost if the profile were not already deployed? Timed by adding
     one to the schema and measuring how long until a query can use it.
  4. What does rollback cost, with the old vectors still present and then without
     them? The second case is the one that decides whether you may delete the old
     field as soon as the backfill finishes.

Leaves the index as it found it: fully migrated, both vectors present.

    VESPA_ENDPOINT=http://127.0.0.1:8090 python3 switch_and_rollback.py
"""
import json, os, shutil, subprocess, sys, time
import numpy as np
import vespa_eval as V

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("EMB_CACHE", os.path.join(HERE, "..", "..", "scripts"))
APP = os.path.join(HERE, "..", "app")
OUT = os.path.join(HERE, "..", "results", "step78_switch_rollback.json")
TMP = os.path.join(HERE, "_deploy_probe")
JSONL = os.path.join(HERE, "rollback_batch.jsonl")
ENDPOINT = V.endpoint()
EXPLORE = 40

meta = V.load_meta(HERE)
doc_ids, doc_text, q_ids, relevant = meta["doc_ids"], meta["doc_text"], meta["q_ids"], meta["relevant"]
docs_old = np.load(os.path.join(CACHE, "emb_docs_old.npy"))
q_old = np.load(os.path.join(CACHE, "emb_q_old.npy"))
q_new = np.load(os.path.join(CACHE, "emb_q_new.npy"))
N = len(doc_ids)
res = {"step": "7+8", "endpoint": ENDPOINT, "n_docs": N, "explore": EXPLORE}


def count(where):
    r = V.search(f"select * from scifact where {where}", "old", docs_old[0], hits=0)
    return r["root"]["fields"]["totalCount"]


def recall(profile, qv, exact=False):
    yql = V.yql_mixed(exact=exact, explore=None if exact else EXPLORE) if profile == "mixed" \
        else V.yql_single("emb_old" if profile == "old" else "emb_new",
                          exact=exact, explore=None if exact else EXPLORE)
    return V.pooled_recall(yql, profile, qv, q_ids, relevant)


def feed(ops, tag):
    with open(JSONL, "w") as f:
        for o in ops:
            f.write(json.dumps(o) + "\n")
    t0 = time.time()
    p = subprocess.run(["vespa", "feed", "-t", ENDPOINT, JSONL], capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        print(p.stderr[:1000], file=sys.stderr)
        raise SystemExit(f"{tag}: feed failed")
    return round(wall, 2)


migrated = count("space = 1")
print(f"{migrated}/{N} documents in the new space", flush=True)
if migrated != N:
    raise SystemExit("this script expects a fully migrated index; run backfill_and_measure.py first")

# ---------------------------------------------------------------- 7: the switch
print("\n=== step 7: the coordinated switch ===", flush=True)
mixed = recall("mixed", q_new)
new = recall("new", q_new)
old = recall("old", q_old)
print(f"  mixed profile, new-model queries : {mixed['recall_at_10_pooled']}  p50 {mixed['latency_ms_p50']}ms")
print(f"  new profile,   new-model queries : {new['recall_at_10_pooled']}  p50 {new['latency_ms_p50']}ms")
print(f"  old profile,   old-model queries : {old['recall_at_10_pooled']}  p50 {old['latency_ms_p50']}ms")
res["at_full_migration"] = {"mixed": mixed, "new": new, "old_rollback_target": old}

# Switching reads with both profiles deployed: the next query simply names the
# other profile. Timed as the wall time of one query each way.
t0 = time.time(); V.search(V.yql_single("emb_new", explore=EXPLORE), "new", q_new[0]); t_new = time.time() - t0
t0 = time.time(); V.search(V.yql_single("emb_old", explore=EXPLORE), "old", q_old[0]); t_old = time.time() - t0
res["switch_preloaded_profile_ms"] = {"to_new": round(t_new * 1000, 1), "to_old": round(t_old * 1000, 1),
                                      "deploys_required": 0, "restarts_required": 0}
print(f"  switching with both profiles deployed: {t_new*1000:.1f} ms to new, "
      f"{t_old*1000:.1f} ms back, no deploy")

# ---------------------------------------------- what a deploy would have cost
print("\n=== step 7b: the same switch if the profile were not deployed ===", flush=True)
shutil.rmtree(TMP, ignore_errors=True)
shutil.copytree(APP, TMP)
sd = os.path.join(TMP, "schemas", "scifact.sd")
text = open(sd).read()
probe = '''
    rank-profile added_later {
        inputs {
            query(q) tensor<float>(x[384])
        }
        first-phase {
            expression: closeness(field, emb_new)
        }
    }
'''
text = text.rstrip()[:-1] + probe + "}\n"
open(sd, "w").write(text)

t0 = time.time()
dp = subprocess.run(["vespa", "deploy", TMP], capture_output=True, text=True)
deploy_returned = time.time() - t0
if dp.returncode != 0:
    print(dp.stdout[-800:], dp.stderr[-800:], file=sys.stderr)
    raise SystemExit("probe deploy failed")

# Poll until a query can actually use the new profile, and keep asking the live
# profile meanwhile to see whether serving is interrupted.
usable_after = None
served = failed = 0
t0 = time.time()
while time.time() - t0 < 180:
    try:
        V.search(V.yql_single("emb_new", explore=EXPLORE), "old", q_new[0])
        served += 1
    except Exception:
        failed += 1
    try:
        r = V.search(V.yql_single("emb_new", explore=EXPLORE), "added_later", q_new[0])
        if "errors" not in r.get("root", {}):
            usable_after = time.time() - t0
            break
    except Exception:
        pass
    time.sleep(1)

res["switch_via_deploy"] = {
    "deploy_command_seconds": round(deploy_returned, 1),
    "profile_usable_after_seconds": None if usable_after is None else round(usable_after, 1),
    "total_seconds": None if usable_after is None else round(deploy_returned + usable_after, 1),
    "queries_served_during_convergence": served,
    "queries_failed_during_convergence": failed}
print(f"  vespa deploy returned in {deploy_returned:.1f}s; new profile answered "
      f"{'never within 180s' if usable_after is None else f'{usable_after:.1f}s later'}")
print(f"  during convergence: {served} queries served, {failed} failed")

print("  restoring the committed application package ...", flush=True)
subprocess.run(["vespa", "deploy", APP], capture_output=True, text=True)
shutil.rmtree(TMP, ignore_errors=True)

# ---------------------------------------------------------------- 8: rollback
print("\n=== step 8: rollback ===", flush=True)
t0 = time.time()
back = recall("old", q_old)
res["rollback_with_old_vectors_present"] = {
    "recall": back["recall_at_10_pooled"], "seconds": round(time.time() - t0, 1),
    "mechanism": "name the old rank profile and the old query model; nothing is written"}
print(f"  old vectors still present: recall {back['recall_at_10_pooled']} "
      f"restored by naming the other profile, nothing written")

# How do you unset a tensor field? Two candidate forms, checked against the
# document afterwards rather than trusting the status code.
import urllib.error, urllib.request

def unset_probe(name, body):
    req = urllib.request.Request(
        f"{ENDPOINT}/document/v1/scifact/scifact/docid/{doc_ids[0]}",
        data=json.dumps({"update": f"id:scifact:scifact::{doc_ids[0]}", **body}).encode(),
        headers={"Content-Type": "application/json"}, method="PUT")
    try:
        with urllib.request.urlopen(req) as r:
            status = f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return {"status": f"HTTP {e.code}", "detail": (e.read()[:160] or b"").decode(errors="replace"),
                "field_removed": False}
    g = urllib.request.Request(f"{ENDPOINT}/document/v1/scifact/scifact/docid/{doc_ids[0]}")
    with urllib.request.urlopen(g) as r:
        gone = json.load(r).get("fields", {}).get("emb_old") is None
    return {"status": status, "field_removed": gone}

attempts = {"assign_null": unset_probe("assign_null", {"fields": {"emb_old": {"assign": None}}})}
# restore that one document before trying the other form
feed([{"put": f"id:scifact:scifact::{doc_ids[0]}",
       "fields": {"doc_id": doc_ids[0], "text": doc_text[doc_ids[0]], "space": 1,
                  "emb_old": {"values": [float(x) for x in docs_old[0]]}}}], "restore probe doc")
attempts["remove_zero"] = unset_probe("remove_zero", {"fields": {"emb_old": {"remove": 0}}})
res["unset_tensor_via_partial_update"] = attempts
for k, v in attempts.items():
    print(f"  partial update {k}: {v['status']}, field removed = {v['field_removed']}"
          + (f" ({v.get('detail','')[:90]})" if v.get("detail") else ""))

# Dropping the old vector across the index. A partial update can do it, so this
# costs no document rewrite, which is what makes dropping it early tempting.
t_drop = feed([{"update": f"id:scifact:scifact::{d}",
                "fields": {"emb_old": {"assign": None}}} for d in doc_ids], "drop emb_old")
after_drop = recall("old", q_old)
res["after_dropping_old_vectors"] = {
    "mechanism": 'partial update, {"emb_old": {"assign": null}}; no document rewrite needed',
    "drop_seconds": t_drop, "docs": N,
    "recall_under_old_profile": after_drop["recall_at_10_pooled"],
    "queries_with_no_hits": after_drop["queries_with_no_hits"]}
print(f"  dropped emb_old from all {N} documents by partial update in {t_drop}s")
print(f"  rollback now impossible: old profile recall {after_drop['recall_at_10_pooled']}, "
      f"{after_drop['queries_with_no_hits']}/{len(q_ids)} queries return nothing")

# Restore, also by partial update, with the vectors still on disk here.
t_restore = feed([{"update": f"id:scifact:scifact::{d}",
                   "fields": {"emb_old": {"assign": {"values": [float(x) for x in docs_old[i]]}}}}
                  for i, d in enumerate(doc_ids)], "restore emb_old")
restored = recall("old", q_old)
res["restoring_old_vectors"] = {
    "refeed_seconds": t_restore,
    "recall_under_old_profile": restored["recall_at_10_pooled"],
    "note": ("these vectors were still on local disk. In a real rollback they would have to be "
             "recomputed, which is the cost of the original embedding run, not of this feed.")}
print(f"  restored by re-feeding in {t_restore}s, recall back to {restored['recall_at_10_pooled']}")

# Put emb_new back so the index ends fully migrated, as found.
docs_new = np.load(os.path.join(CACHE, "emb_docs_new.npy"))
t_renew = feed([{"update": f"id:scifact:scifact::{d}",
                 "fields": {"emb_new": {"assign": {"values": [float(x) for x in docs_new[i]]}},
                            "space": {"assign": 1}}} for i, d in enumerate(doc_ids)], "restore emb_new")
final = recall("new", q_new)
res["left_as_found"] = {"refeed_seconds": t_renew, "new_profile_recall": final["recall_at_10_pooled"]}
print(f"  restored emb_new in {t_renew}s, new profile recall {final['recall_at_10_pooled']}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT}")
