# The same migration, in Vespa

The runs in `../scripts/` simulate a mixed-space index with numpy and hnswlib.
This directory does it in a real engine: one index carrying two embedding fields,
a backfill that fills the second one a slice at a time, a coordinated switch of
query model and target field, and a rollback.

Status: the index is loaded, the baseline reproduces the numpy one exactly, and
the backfill reproduces the article's low point at every migration fraction. The
coordinated switch and the rollback are not written yet.

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

**Steps 5 and 6, the backfill and the mixed state.** `emb_new` filled in
increments of 10% through partial updates, following the same migration order
stage 3 uses for seed 0, with the 300 new-model queries asked against the mixed
index after each increment. `results/step5_backfill.json`:

| Migrated | Vespa exact | stage 3 seed 0 | Vespa graph | p50 latency |
|---|---|---|---|---|
| 0% | 0.5900 | 0.5900 | 0.5900 | 2.3 ms |
| 10% | 0.3658 | 0.3658 | 0.3658 | 2.5 ms |
| 20% | **0.3156** | **0.3156** | **0.3156** | 2.3 ms |
| 30% | 0.3481 | 0.3481 | 0.3481 | 2.4 ms |
| 40% | 0.3658 | 0.3658 | 0.3658 | 2.5 ms |
| 50% | 0.4100 | 0.4100 | 0.4100 | 2.6 ms |
| 60% | 0.4631 | 0.4631 | 0.4572 | 2.5 ms |
| 70% | 0.5133 | 0.5133 | 0.5074 | 2.5 ms |
| 80% | 0.5575 | 0.5575 | 0.5575 | 2.3 ms |
| 90% | 0.5959 | 0.5959 | 0.5959 | 2.2 ms |
| 100% | 0.6460 | 0.6460 | 0.6490 | 2.5 ms |

Every fraction agrees with the simulation to four decimals. Recall bottoms out at
0.3156 with a fifth of the documents moved, against 0.5900 before the backfill
started and 0.6460 when it finished: below both ends, in a real engine, on the
same data.

Nothing looked wrong from the outside while this happened. No query returned an
error, none returned zero hits, and p50 latency stayed between 2.2 and 2.6 ms at
every fraction, including the worst one. A mixed index serves fast and answers
everything. It just answers worse.

**Step 5b, what the backfill costs.** Five write shapes, each one batch of all
5,183 documents through the same feed client. `results/step5b_update_cost.json`:

| Write shape | Throughput | Payload | Document store reads | Document store on disk |
|---|---|---|---|---|
| full document PUT | 1142/s | 52.1 MB | 0.00 | 42.6 → 42.6 MB |
| assign `emb_new` + `space` | 908/s | 44.3 MB | 0.00 | 42.6 → 42.6 MB |
| assign `space` only | 852/s | 0.4 MB | 0.00 | 42.6 → 42.6 MB |
| assign `text`, same value | 1118/s | 8.2 MB | 9.30 | 42.6 → **49.6 MB** |
| assign `emb_new` again | 1005/s | 44.3 MB | 0.00 | 49.6 → 49.6 MB |

**Updating an HNSW-indexed tensor attribute does not touch the document store.**
No reads, no growth, whether the vector is being inserted into the second graph
for the first time or replaced. Updating the indexed `text` field to the value it
already held does both: it reads the document and rewrites the store, which grows
by roughly the payload. That is the distinction Vespa's documentation draws
between attribute and index fields, and the answer to the question it leaves
open, which is that the HNSW index on a tensor attribute does not drag it onto
the expensive path. The article's claim holds.

Throughput is the wrong place to look for it. Everything sits between 850 and
1,140 documents per second, and a 0.4 MB update of a single integer is the
slowest of the five, so on one node at this size the feed rate is bound by
per-operation overhead rather than by what the write touches. The saving is in
storage and I/O, not in documents per second. A practitioner expecting attribute
updates to feed visibly faster will not see it here.

**Step 5c, the bill for the second vector.** `results/step5c_field_memory.json`:

| Field | Memory used | Allocated |
|---|---|---|
| `emb_old` | 8.78 MB | 9.43 MB |
| `emb_new` | 8.80 MB | 9.47 MB |
| `space` | 0.17 MB | 0.36 MB |
| `doc_id` | 0.19 MB | 0.26 MB |

5,183 vectors of 384 float32 values are 7.96 MB of raw numbers, so each field
costs about 0.8 MB over its vectors for the graph and the attribute structures.
Carrying both spaces doubles vector memory for the length of the migration and
leaves the document payload alone, which is the trade the article describes,
priced.

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
python3 scripts/prepare_embeddings.py                     # embeds once, ~1 min, shares ../scripts cache
python3 scripts/feed_old_space.py                         # step 3, the pre-migration index
python3 scripts/eval_recall.py --profile old --exact       # step 4, expect 0.7817
python3 scripts/eval_recall.py --profile old --explore 40  # the graph at ~ef 50
python3 scripts/backfill_and_measure.py                   # steps 5 and 6, the whole curve
python3 scripts/update_cost.py                            # step 5b, what each write shape costs
python3 scripts/field_memory.py                           # step 5c, per-field memory
```

`backfill_and_measure.py` expects the index in the state `feed_old_space.py`
leaves it in, so re-run that first to start over. `SEED` picks the migration
order and defaults to 0, matching stage 3's first seed. `update_cost.py` rewrites
every document several times and leaves the index fully migrated, so run it after
the backfill, not before.

`prepare_embeddings.py` writes the four embedding arrays into `../scripts/` under
the names stage 3 uses, so the two halves of the repo share one cache and one
document ordering. It also writes `scripts/ids.json` so nothing downstream needs
the datasets library.

## What is not verified

- Whether the rank-profile switch is really instant at the scale of a full index,
  and what rollback costs. Steps 7 and 8, not yet run.
- Per-space normalization as a Vespa rank expression. Step 9, not yet written.
- Everything here is one content node, 5,183 documents and 384 dimensions. The
  write-cost result in particular is about what each write shape *touches*, which
  should carry; the throughput numbers are small-single-node numbers and should
  not be read as capacity planning.
- Proton metrics are snapshot aggregates rather than per-request counters, so
  document store reads are attributed at the granularity of a metrics window.
  Document store disk usage is cumulative and unambiguous, and it is the figure
  the conclusion above rests on; the read counter agrees with it.
- Five seeds in the simulation, one seed here. The Vespa curve matches stage 3's
  seed 0 exactly, but a single order cannot show the spread the simulation reports.
