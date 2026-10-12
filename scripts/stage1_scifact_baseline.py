"""
embedding_migration_experiment.py

A small, runnable experiment behind the article
"Your embedding upgrade is a migration, not a config change."

It reproduces the article's claims on a real (small) IR benchmark, on a laptop,
for free -- local models, no API key, no cost -- and it tests one thing whose
answer is genuinely UNKNOWN in advance, so you have a real shot at a finding of
your own to write about.

  TEST 1  Swapping only the QUERY model, leaving documents in the old space,
          fails SILENTLY: nothing errors, recall collapses.
  TEST 2  During a half-migrated window, quality is worst at the START, not the
          middle. (The recall-vs-fraction-migrated curve.)
  TEST 3  You can MEASURE whether two models share a space (neighbour overlap +
          linear-map residual) BEFORE committing to a re-embed.
  TEST 4  *** THE INTERESTING ONE ***  Does a "drift adapter" actually work?
          We learn a linear map from the OLD space to the NEW space on a sample
          of documents, apply it to the rest, and measure how much of the lost
          recall it recovers. The answer is not obvious -- report whatever you
          get, even if it surprises you.

This is a SCALED-DOWN reproduction with small local models, NOT the real OpenAI
production case. Describe it honestly as exactly that.

------------------------------------------------------------------------------
SETUP (one time):
    pip install "sentence-transformers" "datasets" numpy matplotlib

RUN:
    python embedding_migration_experiment.py

Runs on CPU in ~5-10 minutes. Prints numbers and saves two PNG charts.
------------------------------------------------------------------------------
"""

from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Two DIFFERENT models with the SAME output dimension (384).
# Same dimension is the whole point: the cross-space comparison runs with NO
# error. Different dimensions would crash loudly instead of failing silently.
OLD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # the "old" model
NEW_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"   # the "new" model
K = 10                      # recall@K
DATASET = "BeIR/scifact"    # small scientific-claims benchmark (~5k docs)
SEED = 0

# ------------------------------------------------------------------ load data
print(f"Loading {DATASET} ...")
corpus  = load_dataset(DATASET, "corpus")["corpus"]
queries = load_dataset(DATASET, "queries")["queries"]
qrels   = load_dataset(DATASET + "-qrels")["test"]

doc_text   = {d["_id"]: (d.get("title", "") + " " + d["text"]).strip() for d in corpus}
query_text = {q["_id"]: q["text"] for q in queries}

relevant = defaultdict(set)                       # query_id -> {relevant doc_ids}
for r in qrels:
    if r["score"] > 0:
        relevant[str(r["query-id"])].add(str(r["corpus-id"]))

doc_ids   = list(doc_text.keys())
doc_index = {d: i for i, d in enumerate(doc_ids)}
q_ids = [q for q in relevant
         if q in query_text and any(d in doc_index for d in relevant[q])]
print(f"{len(doc_ids)} documents, {len(q_ids)} evaluation queries.\n")

docs   = [doc_text[d] for d in doc_ids]
qtexts = [query_text[q] for q in q_ids]

def embed(model_name, texts):
    print(f"  embedding {len(texts)} texts with {model_name} ...")
    m = SentenceTransformer(model_name)
    return np.asarray(
        m.encode(texts, batch_size=64, show_progress_bar=True,
                 normalize_embeddings=True),
        dtype=np.float32,
    )

def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

docs_old = embed(OLD_MODEL, docs)
docs_new = embed(NEW_MODEL, docs)
q_old    = embed(OLD_MODEL, qtexts)
q_new    = embed(NEW_MODEL, qtexts)

def recall_at_k(query_vecs, doc_vecs, k=K):
    sims = query_vecs @ doc_vecs.T                # cosine (vectors normalized)
    topk = np.argpartition(-sims, k, axis=1)[:, :k]
    hits = total = 0
    for i, qid in enumerate(q_ids):
        rel = relevant[qid]
        retrieved = {doc_ids[j] for j in topk[i]}
        hits  += len(retrieved & rel)
        total += len(rel)
    return hits / total

# --------------------------------------------- TEST 1: the silent failure
baseline = recall_at_k(q_new, docs_new)      # target: everything in the NEW space
crossed  = recall_at_k(q_new, docs_old)      # NEW queries vs OLD docs (no error!)
print("\n=== TEST 1: swapping only the query model fails silently ===")
print(f"  recall@{K}  NEW queries vs NEW docs : {baseline:.3f}   (target, fully migrated)")
print(f"  recall@{K}  NEW queries vs OLD docs : {crossed:.3f}   (nothing errored)")
if baseline:
    print(f"  => relative drop while half-migrated: {100*(baseline-crossed)/baseline:.0f}%")

# ------------------------------------ TEST 2: worst at the START, not middle
print("\n=== TEST 2: worst at the START of the migration, not the middle ===")
rng = np.random.default_rng(SEED)
order = rng.permutation(len(doc_ids))        # random migration order
fractions = np.linspace(0, 1, 11)
curve = []
for f in fractions:
    n = int(f * len(doc_ids))
    mixed = docs_old.copy()
    mixed[order[:n]] = docs_new[order[:n]]   # these docs are "re-embedded"
    curve.append(recall_at_k(q_new, mixed))  # queries live in the NEW space
    print(f"  fraction re-embedded {f:0.1f}  ->  recall@{K} = {curve[-1]:.3f}")

# --------------------------------- TEST 3: do the two models share a space?
print("\n=== TEST 3: measure whether the two models share a space ===")
def neighbours(vecs, k=10):
    s = vecs @ vecs.T
    np.fill_diagonal(s, -1)
    return np.argpartition(-s, k, axis=1)[:, :k]
nn_old, nn_new = neighbours(docs_old), neighbours(docs_new)
overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn_old, nn_new)])
print(f"  avg nearest-neighbour overlap between the two spaces: {overlap:.2f}")
print("  (1.0 = identical neighbourhoods, ~0 = unrelated spaces)")

# ------------------------ TEST 4: does a drift adapter actually recover recall?
print("\n=== TEST 4: does a learned drift adapter recover the lost recall? ===")
# Fit a linear map W (least squares) from OLD -> NEW on HALF the documents,
# then apply it to ALL old vectors and search NEW queries against them.
half = len(doc_ids) // 2
train = order[:half]                         # docs we "have in both spaces"
W, *_ = np.linalg.lstsq(docs_old[train], docs_new[train], rcond=None)
adapted = normalize(docs_old @ W)            # map every old vector into ~new space
recovered = recall_at_k(q_new, adapted)
# how well the map fits, as a sanity number (0 = perfect, 1 = useless)
resid = np.linalg.norm(docs_old[order[half:]] @ W - docs_new[order[half:]]) \
        / np.linalg.norm(docs_new[order[half:]])
span = baseline - crossed
frac_recovered = (recovered - crossed) / span if span > 0 else float("nan")
print(f"  recall@{K}  floor  (NEW q vs OLD docs, broken) : {crossed:.3f}")
print(f"  recall@{K}  adapter (NEW q vs mapped-OLD docs) : {recovered:.3f}")
print(f"  recall@{K}  target (NEW q vs NEW docs)         : {baseline:.3f}")
print(f"  => the drift adapter recovered {100*frac_recovered:.0f}% of the lost recall")
print(f"     (held-out linear-map residual: {resid:.2f}; lower = the two spaces "
      f"are closer to a linear transform of each other)")

# ------------------------------------------------------------------- charts
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(fractions, curve, marker="o")
    plt.axhline(curve[-1], ls="--", color="gray", label="fully re-embedded")
    plt.xlabel("fraction of the corpus re-embedded")
    plt.ylabel(f"recall@{K}")
    plt.title("The consistency window: worst at the start")
    plt.legend(); plt.tight_layout(); plt.savefig("migration_curve.png", dpi=140)

    plt.figure(figsize=(6, 4))
    labels = ["broken\n(no fix)", "drift\nadapter", "target\n(re-embed)"]
    vals   = [crossed, recovered, baseline]
    plt.bar(labels, vals)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center")
    plt.ylabel(f"recall@{K}")
    plt.title("Does a drift adapter recover the lost recall?")
    plt.tight_layout(); plt.savefig("drift_adapter.png", dpi=140)
    print("\nSaved migration_curve.png and drift_adapter.png")
except Exception as e:
    print("(skipped plots:", e, ")")

print("\n----------------------------------------------------------------------")
print("Now write down, in your own words, what actually happened -- especially:")
print("  * how big the silent drop was (Test 1),")
print("  * whether the curve really was worst at the very start (Test 2),")
print("  * and how much the drift adapter recovered (Test 4) -- that number is")
print("    the one nobody could have told you in advance. Report it honestly,")
print("    even if it surprises you or cuts against the article.")
