# What happens to recall while you migrate embedding models

Code and raw results for the experiments behind the article *Half-migrating your
embeddings is worse than not migrating at all*.

The question: when you move a vector index from one embedding model to another a
slice at a time, what does recall do on the way? The expected answer is a dip that
recovers as documents convert. It is not what these runs show.

## The finding

On SciFact, with queries switched to the new model first, recall@10 fell **below both
the starting index and the finished one**, and bottomed out with less than a third of
the documents moved.

| Documents re-embedded | recall@10, raw | recall@10, normalized per space |
|---|---|---|
| 0% (queries already switched) | 0.59 | 0.59 |
| 10% | 0.35 | 0.59 |
| 20% | **0.33 (worst)** | 0.60 |
| 50% | 0.41 | 0.61 |
| 80% | 0.58 | 0.65 |
| 100% | 0.65 | 0.65 |

Mean of five random migration orders. Before anything moved, old queries against old
documents scored 0.78. The five seeds bottomed at 0.307–0.369, at 10% migrated once,
20% three times and 30% once, so the depth varies but every order dipped below both
ends.

Two further results:

- **Per-space score normalization repairs it in exact search but not on a graph.**
  Exact search recovers 0.33 to 0.60. On an HNSW graph updated in place, the same fix
  only reaches 0.38, because by 20% migrated the graph returns just 40% of each
  query's relevant old-space documents, down from 79%. Normalization can only reorder
  what the graph returned. Rebuilding the graph on the mix instead gives 0.39 and 42%,
  so the protocol barely matters. Widening the search fourfold (ef_search 200) lifts
  it to 0.48 and reach to 75%, against a 91% baseline at that width, still short.
- **Switching queries last mirrors the problem rather than avoiding it.** The low
  point moves to 80–90% migrated and still sits below both ends.

## Layout

```
scripts/     the four runs
results/     raw JSON written by each run
```

| Script | What it runs | Output |
|---|---|---|
| `stage2_cosine_separation.py` | Mean cosine of migrated vs un-migrated documents to a new-model query, and the share of top-10 slots migrated documents take, at every fraction | `stage2_cosine_results.json` |
| `stage3_hnsw_and_distant_pair.py` | The main run. Five seeds; HNSW in-place vs rebuilt-on-mix; a distant model pair; top-10 share | `stage3_results.json` |
| `stage5_queries_last.py` | The reverse order: documents migrate first, queries switch last | `stage5_results.json` |
| `stage6_fiqa.py` | Everything above repeated on FiQA, 57,638 documents | `fiqa_results.json` |
| `stage7_ef_search.py` | Does a wider graph search (ef_search 200) recover more than ef 50? | `stage7_ef_results.json` |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`hnswlib` ships as a source distribution with no wheels, so this step needs a C++
toolchain. On a bare container, install build tools first.

## Running

Run stage 3 first. It embeds the documents and caches them as `emb_*.npy`; stages 2,
5 and 7 reuse that cache and will stop with a clear message if it is missing. Set
`EMB_CACHE=/some/path` to keep the cache somewhere other than next to the scripts.

```bash
cd scripts
python stage3_hnsw_and_distant_pair.py   # first run embeds 5,183 docs x3; ~10 min CPU
python stage2_cosine_separation.py       # ~1 min once cached
python stage5_queries_last.py            # ~2 min once cached
python stage7_ef_search.py               # ~2 min once cached
python stage6_fiqa.py                    # 57,638 docs x3 models, 30-90 min CPU
```

Each script writes its JSON next to itself in `scripts/`. The copies in `results/`
are the runs the article quotes.

## Setup used

- **Datasets:** [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) (5,183
  documents, 300 test queries) and [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa)
  (57,638 documents, 648 test queries)
- **Close model pair:**
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) →
  [paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
- **Distant model pair:** all-MiniLM-L6-v2 →
  [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- All three models are 384-dimensional, so the index width never changes
- **Metric:** recall@10, pooled across queries. Total relevant documents retrieved in
  the top 10 divided by total relevant documents, not the per-query mean, so queries
  with several relevant documents weigh more. Published BeIR numbers use the per-query
  mean, so do not compare these figures to a leaderboard directly.
- Cosine similarity throughout, vectors L2-normalized
- **Graph:** [hnswlib](https://github.com/nmslib/hnswlib), M 16, ef_construction 200,
  ef_search 50 (and 200 in stage 7), tuned so the fully migrated graph matched exact
  recall within 2%. The candidate set a graph run scores is the `k = ef_search`
  neighbours it returns, so "reach" is relative to that width.
- **Seeds:** five migration orders, 0 through 4, eleven fractions from 0% to 100%

## Reading the results

Each JSON holds per-seed curves rather than summaries, so the means in the article can
be recomputed. For example, the raw SciFact curve:

```python
import json, numpy as np
d = json.load(open("results/stage3_results.json"))
np.round(np.mean(d["exact_curves_minilm"]["raw"], axis=0), 3)
# array([0.59 , 0.355, 0.33 , 0.342, 0.371, 0.413, 0.468, 0.527, 0.578, 0.612, 0.646])
```

Those eleven values are the 0%, 10%, ... 100% points. The article's table quotes six
of them.

`hnsw.protocols` holds `raw`, `norm` and `reach` for both build protocols;
`distant_pair` holds the second model pair; `fiqa_results.json` mirrors the same shape
with `qfirst_*` and `qlast_*` sections. Reach at 100% migrated is `null`, since no
old-space documents remain to reach.

## Caveats

- On these benchmarks the model standing in for the "new" one is the weaker retriever.
  The experiment is about the shape of the migration window, not whether the upgrade
  is worth making.
- The low point appeared only where the query swap left some recall to lose. On FiQA
  the distant pair lost almost everything at the query swap alone, so the migration
  was a climb from near zero rather than a dip.
- Two model pairs, both 384-dimensional and both small. Nothing here says a 768- or
  3072-dimensional pair behaves the same way.
- Five seeds on 300 SciFact queries and 648 FiQA queries. The spread is reported above
  but there are no confidence intervals, and the per-seed depth varies by 0.06.
- Per-space normalization is an oracle fix. It needs a model tag on every vector and a
  rescoring pass, and it assumes you know which space each document is in.
- hnswlib inserts in parallel, so graph results are not bit-reproducible. Comparing the
  ef 50 runs in `stage3_results.json` and `stage7_ef_results.json`, which run identical
  code on identical embeddings, the per-seed figures differ by up to about 0.003.
- A linear map between the two spaces was tried and is not reported, because it was
  scored on the same documents it was fitted on.
- Neighbour overlap in `stage6_fiqa.py` is computed on a 3,000-document sample, not the
  full corpus. The SciFact overlap figures the article quotes are over the full corpus.
