# The same migration, in Vespa

The runs in `../scripts/` simulate a mixed-space index with numpy and hnswlib.
This directory does it in a real engine: one index carrying two embedding fields,
a backfill that fills the second one a slice at a time, a coordinated switch of
query model and target field, and a rollback.

Status: all of it runs. The baseline reproduces the numpy one exactly, the
backfill reproduces the article's low point at every migration fraction, the
switch and rollback are measured, and per-space normalization runs as a rank
expression — where it turns out to behave differently from the simulation, for a
reason worth the article's attention.

## What the schema does

`app/schemas/scifact.sd` gives every document two 384-dimensional tensor fields,
`emb_old` and `emb_new`, each with its own HNSW index at M 16 and
ef_construction 200 — the same settings as
`../scripts/stage3_hnsw_and_distant_pair.py`, so the two sets of numbers are
comparable. `distance-metric` is `prenormalized-angular` because the embedder
returns unit vectors. An `int` attribute, `space`, records which model's vector a
document is currently served from.

Seven rank profiles, all deployed at once:

| Profile | What it scores | When it is used |
|---|---|---|
| `old` | `closeness(field, emb_old)` | before the migration |
| `mixed` | `max(closeness(label, "n"), closeness(label, "o"))` | during the backfill |
| `mixed_cosine` | the same, on raw cosine | the raw baseline for step 9 |
| `mixed_normalized` | per-space z-score of cosine | the normalization fix |
| `inplace_cosine`, `inplace_normalized` | one field holding both spaces | the step 9b control |
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

Everything below assumes Vespa on 8080. **The committed results were produced on
8090**, because port 8080 was taken on the machine they were run on, which is why
every results JSON records `"endpoint": "http://127.0.0.1:8090"`. Nothing depends
on the number. If 8080 is taken for you too, publish somewhere else and tell the
scripts where it went:

```bash
docker run --detach --name vespa --hostname vespa-container \
  --publish 8090:8080 --publish 19071:19071 vespaengine/vespa:8.763.13
export VESPA_ENDPOINT=http://127.0.0.1:8090
python3 scripts/smoke_test.py
```

One symptom of moving it: `vespa deploy --wait` probes the query port directly on
8080 and will report an unhealthy container even though the deployment succeeded.
Check `curl http://127.0.0.1:8090/status.html` instead.

## Results so far

**Step 3, the initial load.** All 5,183 documents with `emb_old` only and
`space=0`, through the Vespa CLI feed client: 52 MB of JSONL, zero errors, and
the same load run three times took 4.2 s, 2.5 s and 3.9 s, so 1,235 to 2,083
documents per second on this machine. `REPEATS=3` records every figure, because
one number from one run only invites a reader to rerun it and find a different
one. That range is what the backfill in step 5 has to be compared against, since
a backfill of the same index writes the same number of documents again.
`results/step3_feed.json`.

An earlier version of this file quoted 6.5 seconds and 794 documents/second here,
and `backfill_and_measure.py` carried the same pair hardcoded, while the committed
JSON held a figure from a later run. Nothing reads a throughput from a literal any
more; step 5 reads step 3's output.

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

| Migrated | Vespa exact | stage 3 seed 0 | Vespa graph | graph p50 | exact p50 |
|---|---|---|---|---|---|
| 0% | 0.5900 | 0.5900 | 0.5900 | 2.2 ms | 2.4 ms |
| 10% | 0.3658 | 0.3658 | 0.3658 | 2.3 ms | 2.3 ms |
| 20% | **0.3156** | **0.3156** | **0.3156** | 2.3 ms | 2.4 ms |
| 30% | 0.3481 | 0.3481 | 0.3481 | 2.4 ms | 2.5 ms |
| 40% | 0.3658 | 0.3658 | 0.3658 | 2.0 ms | 2.6 ms |
| 50% | 0.4100 | 0.4100 | 0.4100 | 2.3 ms | 2.5 ms |
| 60% | 0.4631 | 0.4631 | 0.4572 | 2.5 ms | 2.7 ms |
| 70% | 0.5133 | 0.5133 | 0.5074 | 2.5 ms | 2.7 ms |
| 80% | 0.5575 | 0.5575 | 0.5575 | 2.3 ms | 2.5 ms |
| 90% | 0.5959 | 0.5959 | 0.5959 | 2.6 ms | 2.3 ms |
| 100% | 0.6460 | 0.6460 | 0.6490 | 2.3 ms | 2.2 ms |

Every fraction agrees with the simulation to four decimals. Recall bottoms out at
0.3156 with a fifth of the documents moved, against 0.5900 before the backfill
started and 0.6460 when it finished: below both ends, in a real engine, on the
same data.

Nothing looked wrong from the outside while this happened. No query returned an
error, none returned zero hits, and graph p50 latency stayed between 2.0 and
2.6 ms at every fraction, including the worst one. A mixed index serves fast and answers
everything. It just answers worse.

**Step 5b, what the backfill costs.** Five write shapes, each one batch of all
5,183 documents through the same feed client. `results/step5b_update_cost.json`:

| Write shape | Throughput | Payload | Document store growth | Per document |
|---|---|---|---|---|
| full document PUT | 1780/s | 52.1 MB | **+21,379,904 B** | +4,125 B |
| assign `emb_new` + `space` | 1254/s | 44.3 MB | +4,120 B | at the floor |
| assign `space` only | 953/s | 0.4 MB | +4,120 B | at the floor |
| assign `text`, same value | 801/s | 8.2 MB | **+21,306,128 B** | +4,111 B |
| assign `emb_new` again | 1246/s | 44.3 MB | +4,120 B | at the floor |

**Updating an HNSW-indexed tensor attribute does not rewrite the document
store.** Each of the three attribute shapes moves it by exactly 4,120 bytes,
which is one 4 KiB `.dat` page plus a few dozen bytes of `.idx`, and the gap
between one run's "after" and the next run's "before" is the same magnitude, so
4 KiB is this instrument's floor rather than a measurement of anything. Dividing
it by 5,183 documents would produce a per-document figure out of noise, so the
right statement is no measurable growth, whether the vector is being inserted
into the second graph for the first time or replaced.

Assigning the indexed `text` field the value it already held rewrites 4,111 bytes
per document, within 0.4% of what a full PUT of the same document
writes, so an indexed-field update really does rewrite the whole document and not
just the field. That is the distinction Vespa's documentation draws between
attribute and index fields, and the answer to the question it leaves open, which
is that the HNSW index on a tensor attribute does not drag it onto the expensive
path.

**How this is measured, and how the first version of it was wrong.** The obvious
instrument is the proton metric `document_store.disk_usage.average`, read before
and after each batch. It does not work. Those values come from a metrics snapshot
interval, so two reads a few seconds apart return the same snapshot. Measured
that way, `put_full` — 5,183 full document PUTs, 52 MB of payload, a write that
cannot possibly leave the store untouched — reported **42550910 bytes before and
42550910 bytes after, byte-identical**. An instrument that cannot see that write
cannot be trusted to prove the absence of one, and the earlier table above was
read off it.

The store is now measured directly. Proton keeps the document store as an
append-only log under `0.ready/summary`, so a write that touches it makes those
files grow and a write that does not leaves them alone. Each batch is followed by
a forced flush and then a wait until the byte count stops moving. The script then
checks its own instrument: `put_full` and `update_text` both *have* to grow the
store, and `positive_controls_pass` in the JSON records whether they did. They
do, which is what makes the three zeros evidence rather than a blind spot.

One part of the earlier claim is gone rather than fixed. It said the tensor
update performed no document store *reads*, which came from the same unreliable
metric. What is established is that it performs no rewrite, and since a
read-modify-write has to write, no rewrite rules out a read-modify-write. The
read counter itself is not evidence here.

Throughput is the wrong place to look for it. Everything sits between 800 and
1,780 documents per second, a 0.4 MB update of a single integer is the second
slowest of the five, and the full document PUT that writes the most is the
fastest, so on one node at this size the feed rate is bound by per-operation
overhead rather than by what the write touches. The saving is in storage and I/O,
not in documents per second. A practitioner expecting attribute updates to feed
visibly faster will not see it here.

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

**Step 7, the coordinated switch.** With every document migrated,
`results/step78_switch_rollback.json`:

| Profile | Query model | recall@10 |
|---|---|---|
| `mixed` | new | 0.6460 |
| `new` | new | 0.6460 |
| `old` | old | 0.7788 |

The mixed profile and the new one agree once nothing is left in the old space, so
the switch costs nothing in quality and can be made whenever it is convenient
rather than in a window. Switching is one query naming the other profile: 3.0 ms
to the new space, 2.7 ms back, no deploy and no restart.

Had the profile not been deployed in advance, adding one took 0.7 s for
`vespa deploy` to return and about 1.0 s more before a query could use it, with
two queries served in the gap and neither failing, which is too few to call it
uninterrupted serving. So pre-deploying both profiles turns
a roughly two-second change into a three-millisecond one. That is a real
advantage and a small one: the honest version is that on a single node the deploy
route is also cheap and does not drop traffic. The reason to pre-deploy is that
the switch becomes a routing decision you can make and unmake per query, not that
a deploy would have hurt.

**Step 8, rollback.** While both vectors are present, rollback is the same
operation as the switch: name the old profile and the old query model, write
nothing, and recall returns to 0.78 immediately.

That changes completely once the old vectors are gone, and getting rid of them is
easier than it should be:

| Step | Mechanism | Cost | recall@10, old profile |
|---|---|---|---|
| drop `emb_old`, all 5,183 documents | partial update, `{"assign": null}` | 5.9 s | **0.0000**, 300/300 queries empty |
| restore from vectors on local disk | partial update, `assign` | 5.5 s | 0.7817 |

A tensor attribute can be unset with a partial update: `{"emb_old": {"assign":
null}}` returns 200 and the field really is gone, confirmed by reading the
document back rather than by trusting the status code. (`{"remove": 0}` is a 400:
*Expected start of JSON object, got VALUE_NUMBER_INT*.) No document rewrite is
needed, so reclaiming 8.8 MB per million-ish vectors takes six seconds and one
feed.

Which is the trap. Dropping the old space is fast, cheap and reversible-looking,
and the moment it is done the old profile returns nothing at all — not degraded
results, zero hits on every query. Recovery needs the old vectors back, and the
5.5 s above is only the feed: those vectors were still sitting on local disk. In
a real rollback they would have to be recomputed with the old model first, which
is the original embedding run again.

So the thing to keep is not a plan for rollback, it is the old field. Dropping it
is the step that ends the migration, and it belongs after the evaluation
threshold has been met on live traffic, not after the backfill finishes.

**Step 9, normalization as a rank expression, and the result that changes the
story.** The simulation repairs the low point by z-scoring each space separately,
and finds this works in exact search but mostly fails on a graph, because
rescoring cannot promote a document the graph never returned. Running the same
fix in Vespa reproduces the first half and contradicts the second.

A first-phase expression sees one document at a time and cannot know the
distribution it belongs to, so the per-space mean and standard deviation are
computed per query by the caller and passed as query inputs:

```
if (attribute(space) == 1,
    (cos_new() - query(mu_n)) / query(sd_n),
    (cos_old() - query(mu_o)) / query(sd_o))
```

It ranks on a dot product rather than `closeness()`, because the vectors are unit
length so the dot product is the cosine, which is the quantity the numpy runs
normalize. The statistics come from every document in each space, the same oracle
the simulation's exact normalization uses.

With two fields, normalization recovers the low point completely. At 20%
migrated, with each operator given `targetHits` 50 and
`hnsw.exploreAdditionalHits` 40, which matches stage 3's returned list length of
50 but searches wider than its ef_search 50 (see the caveat at the end of this
section),
the graph gets 0.5870, which is exactly what exact search gets. Reach of relevant
old-space documents does not fall over the migration at all. It rises as the old
space empties, for a reason that took two more runs to pin down and is not the
one written here first; see "Nothing is reserved" below
(`results/step9_normalization.json`):

| Migrated | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| two fields, reach | 0.811 | 0.803 | 0.801 | 0.824 | 0.863 | 0.861 | 0.891 | 0.908 | 0.914 | 0.971 |
| one field in place, reach | 0.811 | 0.587 | 0.442 | 0.393 | 0.356 | 0.347 | 0.348 | 0.347 | 0.343 | 0.229 |
| stage 3 hnswlib in place, reach | 0.794 | 0.527 | 0.399 | 0.353 | — | — | — | — | — | — |

Overwriting the single field in place puts both spaces back in one graph and the
failure returns, exactly as the simulation has it: reach collapses to 0.442 at
20% migrated and normalization recovers only 0.4808 of the 0.5811 exact search
reaches. That run is `inplace_vs_dualfield.py`.

### It is not the graph, and that took a second measurement to establish

The natural reading of the in-place collapse is that the graph is at fault: links
built from distances across two different spaces lead a new-space query away from
the old-space documents it needs. Both this repo and the simulation report the
failure as a graph result, so that reading is easy to reach for and nothing here
tested it.

It is wrong. `inplace_vs_dualfield.py` now measures the same reach twice, once
through the graph and once with `approximate:false` at the same `targetHits`.
Exact search at `targetHits` 50 returns the true top 50 documents by score, so
whatever it misses was lost to score competition alone, with no graph in the
picture:

| Migrated | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|
| one field, graph reach, `targetHits` 50 | 0.587 | 0.442 | 0.393 | 0.356 | 0.347 |
| one field, **exact** reach, `targetHits` 50 | 0.613 | **0.475** | 0.435 | 0.400 | 0.387 |
| one field, graph reach, `targetHits` 100 | 0.731 | 0.594 | 0.515 | 0.463 | 0.445 |
| one field, **exact** reach, `targetHits` 100 | 0.741 | **0.627** | 0.548 | 0.502 | 0.497 |

At 20% migrated a perfect search into a 50-document candidate list still reaches
only 0.475 of the relevant old-space documents, against the graph's 0.442. Reach
starts at 0.805 before anything migrates, so of the 0.363 it loses by 20%
migrated, score competition accounts for 0.331 and the graph for 0.033, nine
tenths against one tenth.

So the in-place failure is not an approximate-search artefact. It is the same
crowding-out that creates the low point in the first place, happening one stage
earlier: when a single candidate list is filled by raw score, migrated documents
take most of it, and the relevant old-space documents are gone before anything
gets rescored. Brute-force retrieval into a fixed-size candidate list would lose
them too. That is why widening the list helps (stage 7, and the `targetHits` 100
rows above) and why giving the old space a list of its own helps far more.

### The control that comparison needs, because the first version was confounded

The two runs above were first compared at the same `targetHits`, and that is not
the same candidate budget. A mixed query over two fields has **two**
`nearestNeighbor` operators and each gets its own `targetHits`, so at 50 each the
dual-field run was scoring about 90 candidates where the single-field run scored
50. Measured, not assumed: `mixed_list_split` records 50.0 from the new space and
40.0 from the old at 20% migrated. Stage 7 shows candidate width matters a great
deal in this regime, so most of a 0.5870-against-0.4808 gap could have been
budget alone.

So the comparison is run at matched and at reversed total budgets. At 20%
migrated:

| Migration style | `targetHits` | Candidates actually scored | graph normalized | old-space reach |
|---|---|---|---|---|
| two fields | 25 per operator | **44.9** | **0.5900** | 0.728 |
| two fields | 50 per operator | 90.0 | 0.5870 | 0.801 |
| one field in place | 50 | 50.0 | 0.4808 | 0.442 |
| one field in place | 100 | 100.0 | 0.5516 | 0.594 |
| exact search, two fields | — | all 5,183 | 0.5870 | — |

The confound was real and the effect survives it. Give the in-place graph twice
the candidates the dual-field query gets and it still recovers less, 0.5516
against 0.5870, with reach at 0.594 against 0.801. Cut the dual-field query to
**fewer** candidates than the in-place run — 44.9 against 50.0 — and it still
wins, 0.5900 against 0.4808. Two fields at 45 candidates beat one field at 100.

Budget does account for part of what the original comparison showed. Doubling the
in-place budget from 50 to 100 lifts normalized recall from 0.4808 to 0.5516,
which is most of the way to the dual-field result, so roughly two thirds of the
gap as first reported was candidate width rather than structure.

### Nothing is reserved, which the second version of this explanation got wrong

The explanation first written here was that each operator holds a candidate
budget of its own, so migrated documents cannot spend the old space's. That is
also wrong, and the data refuting it was already committed in this repo.
`mixed_list_split.mean_old_space_hits` in `results/step9_normalization.json` is
`targetHits × (1 − migrated)` at every fraction:

| Migrated | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| old-space documents returned | 50.0 | 45.17 | 40.0 | 35.25 | 30.52 | 25.48 | 20.4 | 15.26 | 10.54 | 5.01 |

The `_t25` run reproduces the same line at 25. The old-space operator retrieves
`targetHits` candidates from the `emb_old` graph, which holds a vector for every
document including the migrated ones, and then loses the migrated ones to the
`and space = 0` filter. Vespa is post-filtering here, so migrated documents *do*
spend that operator's budget, in exact proportion to their share. Nothing is
reserved and the budget is not fixed.

### So is it the second field or the second operator? (step 9c)

Those two had never been separated, because step 9 changes both at once. So
`inplace_two_operators.py` runs the in-place mix — one field, identical to step
9b — but queries it with **two** operators over that one field, each filtered to
a space:

```
({targetHits:50}nearestNeighbor(emb_old, q) and space = 1) or
({targetHits:50}nearestNeighbor(emb_old, q) and space = 0)
```

That gives the old space an operator and a `targetHits` of its own while leaving
the field mixed. At 20% migrated (`results/step9c_two_operators_one_field.json`
and `..._t100.json`):

| Arrangement | `targetHits` | Candidates scored | Old-space slots in them | graph normalized |
|---|---|---|---|---|
| **two fields**, two operators | 50 each | **90.0** | **40.0** | **0.5870** |
| one field, two operators | 50 each | 50.0 | 6.5 | 0.4838 |
| one field, one operator | 50 | 50.0 | — | 0.4808 |
| one field, two operators | 100 each | 100.0 | 18.0 | 0.5516 |
| one field, one operator | 100 | 100.0 | — | 0.5516 |

**The second operator buys nothing. The second field buys everything.** Two
operators over one field return one list of 50, not two — the number is exactly
50.0 at every fraction — because both operators search the same graph, and the
new space takes 43.46 of those 50 slots on raw score. Recovery lands at 0.4838,
which is the single-operator number (0.4808) to within run-to-run noise. At
`targetHits` 100 the two arrangements are identical to four decimals, 0.5516
both ways.

Two *fields* are two graphs, and two graphs are genuinely two candidate lists,
90 documents against 50. So what the second field buys is a second list, whose
ranking contains only old-space vectors and therefore has no cross-space score
competition in it. Migrated documents still shrink that list in proportion to
their number, as the post-filter table above shows, but they cannot outscore
their way into it. That is the difference, and it is now separated from both
candidate width and operator count.

### A caveat about how reach is measured here

The reach column above and in the earlier tables is measured by a **standalone
single-operator query**, not by the mixed query whose recall sits beside it. Step
9c shows why that matters: one field with two operators reports reach 0.804, as
high as the dual-field case, while recovering only 0.4838, because the
old-space documents are reachable by that operator *alone* and still never make
the combined query's 50-slot list. Reach measured this way is an upper bound on
what the scored query sees. **The quantity that actually predicts recovery is
the old-space share of the scored candidate list** — 40.0 against 6.5 against
18.0 in the table above — not the standalone probe.

That also disposes of the alternative explanation for step 9b. If Vespa's graph
were simply better than hnswlib's, the in-place run in Vespa would not collapse,
and it collapses to 0.442 reach at 20% migrated, close to the simulation's 0.399.

So the article's warning holds for in-place model changes and not for dual-write,
and the recommendation it already makes for Vespa turns out to be protected by a
mechanism it does not mention. The second embedding field is not only a way to
keep both versions queryable. It is what keeps the old space in the candidate
list at all while the index is mixed.

One practitioner note from getting this wrong first. With `targetHits` equal to
the number of hits returned, normalization changed nothing whatsoever in the
in-place case, 0.3068 before and after, because rescoring ten candidates to
return ten documents cannot alter the set. Rescoring needs a candidate list
wider than the result, and how much wider is the whole lever. `TARGET_HITS` and
`FRACTIONS` reproduce any of the settings above:

```bash
TARGET_HITS=50  python3 normalization.py            # two fields, the main run
TARGET_HITS=25  OUT_TAG=_t25  python3 normalization.py    # matched total budget
TARGET_HITS=50  python3 inplace_vs_dualfield.py     # one field, same per operator
TARGET_HITS=100 OUT_TAG=_t100 python3 inplace_vs_dualfield.py   # matched total
TARGET_HITS=50  python3 inplace_two_operators.py    # one field, two operators
```

A caveat that belongs with this result rather than at the end. On 5,183
documents, Vespa's graph at `targetHits` 50 plus `exploreAdditionalHits` 40 is
effectively exact: `graph_raw` equals `exact_raw` to four decimals at ten of the
eleven fractions, and at the eleventh (0% migrated) the graph is marginally
*higher*, 0.5929 against 0.5900. The dual-field numbers therefore show what a
second candidate list does when the graph itself is barely lossy. On an index
large enough for ANN to lose real recall, a second field should still give the
old space a list of its own, but how much of the recovery survives is not
something 5,183 documents can tell you.

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
- Per-space normalization recovers the low point on a graph when the two spaces
  sit in separate fields, and does not when one field holds both, at matched,
  halved and doubled candidate budgets.
- What does the work is the second field, not the second operator. Two operators
  over one mixed field return one candidate list rather than two, and recover
  what one operator recovers.
- A partial update to an HNSW-indexed tensor attribute does not rewrite the
  document store, on a measurement whose positive controls pass.

## Running the migration

First, the Python dependencies. The root `requirements.txt` covers both halves of
the repo, and `prepare_embeddings.py` needs `datasets` and
`sentence-transformers` from it:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r ../requirements.txt
```

Then the runs, in this order. Every line is here, resets included, so the block
can be pasted top to bottom:

```bash
export VESPA_ENDPOINT=http://127.0.0.1:8080          # 8090 if you moved it above
R="NO_RECORD=1 python3 scripts/feed_old_space.py"         # the reset, used below

python3 scripts/prepare_embeddings.py                     # embeds once, ~1 min, shares ../scripts cache
REPEATS=3 python3 scripts/feed_old_space.py               # step 3, the pre-migration index
python3 scripts/eval_recall.py --profile old --exact      # step 4, expect 0.7817
python3 scripts/eval_recall.py --profile old --explore 40  # the graph at ~ef 50
python3 scripts/backfill_and_measure.py                   # steps 5 and 6, the whole curve
python3 scripts/update_cost.py                            # step 5b, what each write shape costs
python3 scripts/field_memory.py                           # step 5c, per-field memory
python3 scripts/switch_and_rollback.py                    # steps 7 and 8

eval $R; TARGET_HITS=50  python3 scripts/normalization.py         # step 9, the dual-field fix
eval $R; TARGET_HITS=25  OUT_TAG=_t25 python3 scripts/normalization.py    # matched total budget
eval $R; TARGET_HITS=50  python3 scripts/inplace_vs_dualfield.py         # step 9b, one field
eval $R; TARGET_HITS=100 OUT_TAG=_t100 python3 scripts/inplace_vs_dualfield.py
eval $R; TARGET_HITS=50  python3 scripts/inplace_two_operators.py        # step 9c, the separator
eval $R; TARGET_HITS=100 OUT_TAG=_t100 python3 scripts/inplace_two_operators.py
eval $R                                                   # leave the index pre-migration
```

Skipping a reset is the one mistake that produces wrong numbers silently, on the
repo's most contested result, which is why the resets are in the block rather
than in prose below it. `NO_RECORD=1` keeps a reset from overwriting the recorded
load throughput. `inplace_vs_dualfield.py` and `inplace_two_operators.py`
overwrite `emb_old` deliberately and restore it when they finish.

Steps 9 and 9b default to all eleven fractions and 9c to six;
`FRACTIONS=0,0.1,0.2,0.3` runs any of them short. `OUT_TAG` suffixes the output
file so a control run does not overwrite the main one.

`switch_and_rollback.py` expects a fully migrated index, deploys a throwaway
schema variant and restores the committed one, and puts the index back as it
found it.

`backfill_and_measure.py` expects the index in the state `feed_old_space.py`
leaves it in, so re-run that first to start over. `SEED` picks the migration
order and defaults to 0, matching stage 3's first seed. `update_cost.py` rewrites
every document several times and leaves the index fully migrated, so run it after
the backfill, not before. It also needs `docker exec` into the container to size
the document store files, so `VESPA_CONTAINER` and `VESPA_STORE` are there for a
deployment that is not the Docker one in this README.

`prepare_embeddings.py` writes the four embedding arrays into `../scripts/` under
the names stage 3 uses, so the two halves of the repo share one cache and one
document ordering. It also writes `scripts/ids.json` so nothing downstream needs
the datasets library.

## What is not verified

- The normalization statistics are an oracle: computed over every document in
  each space by the caller. A real implementation would estimate them, and the
  estimate's error is not measured here.
- The switch and rollback timings are single-node and 5,183 documents. What
  should carry is the shape: a pre-deployed profile makes the switch a per-query
  decision, and unsetting a tensor attribute needs no document rewrite. The
  seconds should not.
- Recall under the old profile came back as 0.7788 in one run of step 7 and
  0.7817 in another, both at `exploreAdditionalHits` 40. That is graph
  non-determinism of the size ../README documents, not an effect of the
  migration.
- Everything here is one content node, 5,183 documents and 384 dimensions. The
  write-cost result in particular is about what each write shape *touches*, which
  should carry; the throughput numbers are small-single-node numbers and should
  not be read as capacity planning.
- Proton metrics are snapshot aggregates rather than per-request counters, and
  reading one before and after a write that takes a few seconds returns the same
  snapshot twice. They are recorded in the step 5b JSON for completeness and no
  conclusion rests on them. The document store result is measured from the store
  files on disk, with `put_full` and `update_text` as positive controls that have
  to show growth for the run to mean anything.
- The document store result says a tensor attribute update performs no rewrite.
  Whether it performs a *read* is not measured; it is inferred, since a
  read-modify-write has to write.
- The dual-field normalization result is measured against the in-place
  alternative at matched, halved and doubled candidate budgets, and against a
  one-field two-operator arrangement, and survives all of them. It is one seed,
  one index of 5,183 documents, and a graph that is near-exact at this size. The
  direction should carry further than the magnitudes.
- Reach in every table here is measured by a standalone single-operator query,
  which step 9c shows is an upper bound on what the mixed query actually scores.
  The old-space share of the scored candidate list is the better quantity and is
  reported beside it.
- Reach is pooled in these runs (relevant found over relevant total) and a
  per-query mean in `../scripts/stage3_hnsw_and_distant_pair.py`. Where the two
  appear in one table the estimators differ, which is worth a few hundredths.
- No query here writes while reading. Write catch-up, which the article's
  strategy hierarchy turns on, is not measured anywhere in this repo.
- There is no embedder component in the application package. The "model switch"
  is a different query vector computed outside Vespa, so what is measured is the
  rank-profile and target-field switch, not a two-embedder deployment.
- The switch and rollback are exercised at 100% migrated on a quiescent index,
  never mid-migration and never under concurrent writes.
- Five seeds in the simulation, one seed here. The Vespa curve matches stage 3's
  seed 0 exactly, but a single order cannot show the spread the simulation reports.
