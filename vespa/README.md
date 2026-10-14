# The same migration, in Vespa

The runs in `../scripts/` simulate a mixed-space index with numpy and hnswlib.
This directory does it in a real engine: one index carrying two embedding fields,
a backfill that fills the second one a slice at a time, a coordinated switch of
query model and target field, and a rollback.

Status: the index is loaded and the pre-migration baseline reproduces the numpy
one exactly. The backfill, switch and rollback runs are not written yet.

## What the schema does

`app/schemas/scifact.sd` gives every document two 384-dimensional tensor fields,
`emb_old` and `emb_new`, each with its own HNSW index at M 16 and
ef_construction 200 — the same settings as
`../scripts/stage3_hnsw_and_distant_pair.py`, so the two sets of numbers are
comparable. `distance-metric` is `prenormalized-angular` because the embedder
returns unit vectors. An `int` attribute, `space`, records which model's vector a
document is currently served from.

Three rank profiles, all deployed at once:

| Profile | What it scores | When it is used |
|---|---|---|
| `old` | `closeness(field, emb_old)` | before the migration |
| `mixed` | `max(closeness(label, "n"), closeness(label, "o"))` | during the backfill |
| `new` | `closeness(field, emb_new)` | after the switch |

The `mixed` profile is the interesting one. A single new-model query vector
reaches both graphs through two labelled `nearestNeighbor` operators, and each
document is scored in whichever space it currently lives in:

```
select doc_id from scifact where
  ({targetHits:10, label:"n"}nearestNeighbor(emb_new, q) and space = 1) or
  ({targetHits:10, label:"o"}nearestNeighbor(emb_old, q) and space = 0)
```

Taking the `max` of two closeness values computed in different embedding spaces
is exactly the comparison the article argues you cannot make. Writing it as a
rank expression makes it explicit rather than accidental.

Because the switch is a choice of rank profile and both profiles are deployed
from the start, moving the whole index from the old space to the new one is a
query parameter rather than a redeploy. The run will measure whether that holds.

## Running it

Verified on Vespa **8.763.13**, Vespa CLI 8.699.8, Docker 29.2.0 on macOS
(Apple M3 Pro, 8.2 GB given to Docker; Vespa's documented minimum is 4 GB).

```bash
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa:8.763.13

vespa config set target local
vespa deploy --wait 300 ./app

python3 scripts/smoke_test.py          # feeds 2 documents, runs 4 query shapes, cleans up
```

If something else on the machine already holds port 8080, publish Vespa
somewhere else and tell the smoke test where it went:

```bash
docker run --detach --name vespa --hostname vespa-container \
  --publish 8090:8080 --publish 19071:19071 vespaengine/vespa:8.763.13
VESPA_ENDPOINT=http://127.0.0.1:8090 python3 scripts/smoke_test.py
```

`vespa deploy --wait` probes the query port directly and will report an unhealthy
container in that case even though the deployment succeeded. Check
`curl http://127.0.0.1:8090/status.html` instead.

## Results so far

**Step 3, the initial load.** All 5,183 documents with `emb_old` only and
`space=0`, through the Vespa CLI feed client: 6.5 seconds, 794 documents/second,
zero errors, 52 MB of JSONL. That is the number the backfill in step 5 has to be
compared against, since a backfill of the same index writes the same number of
documents again. `results/step3_feed.json`.

**Step 4, the pre-migration baseline.** Old-model queries against `emb_old`,
pooled recall@10 over the same 300 queries and 339 relevance judgements as the
hnswlib runs. `results/step4_baseline.json`:

| Search | recall@10 | p50 latency | hnswlib equivalent |
|---|---|---|---|
| exact (`approximate:false`) | **0.7817** | 2.3 ms | 0.7817, identical |
| graph, `exploreAdditionalHits` 0 | 0.7463 | 1.8 ms | — |
| graph, `exploreAdditionalHits` 40 | 0.7788 | 1.9 ms | ef_search 50 |
| graph, `exploreAdditionalHits` 190 | 0.7817 | 2.2 ms | ef_search 200 |

Vespa's exact search returns the same pooled recall as the numpy run to four
decimals, 265 of 339 relevant documents either way. The embeddings, the feed, the
`prenormalized-angular` metric, the query path and the metric definition all
agree with the simulation, so a number from Vespa can be set beside a number from
`../scripts/` without an asterisk.

Exploring about 50 candidates lands within 0.4% of exact, which is the same
relationship `../scripts/stage3_hnsw_and_distant_pair.py` reports for ef_search
50. The two graph configurations are comparable.

### One thing worth knowing before tuning an ANN query

`hnsw.exploreAdditionalHits` only works as a YQL annotation on the operator:

```
{targetHits:10, hnsw.exploreAdditionalHits:190}nearestNeighbor(emb_old, q)
```

Passing it as a query property instead is accepted and then silently ignored.
Recall sat at 0.7463 for property values of 0, 40, 90, 190 and 490 before the
annotation moved it to 0.7817. Vespa validates query properties and rejects
unknown ones with a 400 — `ranking.matching.hnsw.exploreAdditionalHits` fails
that way — but it does not validate YQL annotations, so a misspelled one tunes
nothing and reports no error. Raising `targetHits` to 100 reaches exact recall
too, at the cost of a larger result set to rank.

## What is verified so far

- The schema deploys: two HNSW-indexed tensor fields on one document type,
  `prenormalized-angular`, `attribute: fast-search`, and `closeness(label, ...)`
  over two labelled `nearestNeighbor` operators are all accepted.
- Two `nearestNeighbor` operators in one query, OR'd and filtered by `space`,
  return documents from both spaces in one result set, each scored in its own.
- A document with no `emb_new` never appears in the new-space graph, so
  un-migrated documents drop out of new-space retrieval rather than scoring badly.
- `approximate:false` gives exact search for a reference number, on the same
  index, without a second deployment.
- The whole pipeline agrees with the simulation: Vespa exact search reproduces
  the numpy baseline at 0.7817.

## Running the migration

```bash
python3 scripts/prepare_embeddings.py                    # embeds once, ~1 min, shares ../scripts cache
python3 scripts/feed_old_space.py                        # step 3, the pre-migration index
python3 scripts/eval_recall.py --profile old --exact      # step 4, expect 0.7817
python3 scripts/eval_recall.py --profile old --explore 40 # the graph at ~ef 50
```

`prepare_embeddings.py` writes the four embedding arrays into `../scripts/` under
the names stage 3 uses, so the two halves of the repo share one cache and one
document ordering. It also writes `scripts/ids.json` so nothing downstream needs
the datasets library.

## What is not verified

- **Whether a partial update to an HNSW-indexed tensor attribute avoids the
  document store read-modify-write.** Vespa's
  [partial updates](https://docs.vespa.ai/en/writing/partial-updates.html)
  documentation says most attribute fields avoid it and that index fields
  require it, but it does not say which applies to a tensor attribute that also
  carries an HNSW index. The article asserts the cheap path. The backfill run has
  to measure it rather than repeat it.
- Feed throughput, query latency during the backfill, and the memory and disk
  cost of the second vector. All unmeasured.
- Whether the rank-profile switch is really instant at the scale of a full index.
