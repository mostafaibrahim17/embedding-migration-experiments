# What happens to recall while you migrate embedding models

Code and raw results for the experiments behind the article *Half-migrating your
embeddings is worse than not migrating at all*.

The question: when you move a vector index from one embedding model to another a
slice at a time, what does recall do on the way? The expected answer is a dip that
recovers as documents convert. That is not what happens.

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
documents scored 0.78.

Two further results:

- **Per-space score normalization repairs it in exact search but not on a graph.**
  Exact search recovers 0.33 to 0.60. On an HNSW graph the same fix only reaches 0.38,
  because by 20% migrated the graph returns just 40% of each query's relevant
  old-space documents, down from 79%. Normalization can only reorder what the graph
  returned. Widening the search fourfold (ef_search 200) lifts it to 0.48 and reach to
  75%, still short.
- **Switching queries last mirrors the problem rather than avoiding it.** The low
  point moves to 80–90% migrated and still sits below both ends.

## Layout

```
scripts/     the five runs, in order
results/     raw JSON written by each run
notebook_scifact_first_run.ipynb    the original exploratory run (seed 0)
```

| Script | What it runs | Output |
|---|---|---|
| `stage1_scifact_baseline.py` | First SciFact migration curve, exact search | (notebook) |
| `stage3_hnsw_and_distant_pair.py` | Five seeds; HNSW in-place vs rebuilt-on-mix; a distant model pair; share of top-10 slots held by migrated documents | `stage3_results.json` |
| `stage5_queries_last.py` | The reverse order: documents migrate first, queries switch last | `stage5_results.json` |
| `stage6_fiqa.py` | Everything above repeated on FiQA, 57,638 documents | `fiqa_results.json` |
| `stage7_ef_search.py` | Does a wider graph search (ef_search 200) recover more than ef 50? | `stage7_ef_results.json` |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
cd scripts
python stage3_hnsw_and_distant_pair.py      # ~2 min after embeddings are cached
python stage5_queries_last.py
python stage6_fiqa.py                        # slowest; 57k documents, three models
python stage7_ef_search.py
```

Each script writes its JSON next to itself. Embeddings are cached as `emb_*.npy` on
first run and reused afterwards; set `EMB_CACHE=/some/path` to put them elsewhere.
The first run of `stage6_fiqa.py` embeds 57,638 documents three times, so give it a
while.

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
- **Metric:** recall@10, cosine similarity, vectors L2-normalized
- **Graph:** [hnswlib](https://github.com/nmslib/hnswlib), M 16, ef_construction 200,
  ef_search 50 (and 200 in stage 7), fixed build seed
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
with `qfirst_*` and `qlast_*` sections.

## Caveats

- On these benchmarks the model standing in for the "new" one is the weaker retriever.
  The experiment is about the shape of the migration window, not whether the upgrade
  is worth making.
- The low point appeared only where the query swap left some recall to lose. On FiQA
  the distant pair lost almost everything at the query swap alone, so the migration
  was a climb from near zero rather than a dip.
- A linear map between the two spaces was tried and is not reported, because it was
  scored on the same documents it was fitted on.
