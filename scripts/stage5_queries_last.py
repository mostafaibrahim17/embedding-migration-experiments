"""
embedding_migration_stage5.py
Queries-LAST migration order. Documents migrate old -> new in place while queries
stay on the OLD model; queries switch only at the end. Same corpus, seeds, fractions
as stage 3. Reuses cached document embeddings; embeds the 300 queries with the old model.
Runs exact search for both pairs, and HNSW (in-place update, same M/ef as stage 3) for
the MiniLM pair.
"""
import json, os, time
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import hnswlib
OLD_MODEL="sentence-transformers/all-MiniLM-L6-v2"
K=10; SEEDS=[0,1,2,3,4]; FR=np.linspace(0,1,11); CACHE=os.environ.get("EMB_CACHE",".")
def log(*a): print(*a, flush=True)
corpus=load_dataset("BeIR/scifact","corpus")["corpus"]; queries=load_dataset("BeIR/scifact","queries")["queries"]
qrels=load_dataset("BeIR/scifact-qrels")["test"]
doc_text={d["_id"]:(d.get("title","")+" "+d["text"]).strip() for d in corpus}; query_text={q["_id"]:q["text"] for q in queries}
relevant=defaultdict(set)
for r in qrels:
    if r["score"]>0: relevant[str(r["query-id"])].add(str(r["corpus-id"]))
doc_ids=list(doc_text.keys()); doc_index={d:i for i,d in enumerate(doc_ids)}
q_ids=[q for q in relevant if q in query_text and any(d in doc_index for d in relevant[q])]; N=len(doc_ids)
qtexts=[query_text[q] for q in q_ids]
def load(tag): return np.load(os.path.join(CACHE,f"emb_{tag}.npy"))
docs_old=load("docs_old"); docs_new=load("docs_new"); docs_far=load("docs_far")
p=os.path.join(CACHE,"emb_q_old.npy")
if os.path.exists(p): q_old=np.load(p)
else:
    q_old=np.asarray(SentenceTransformer(OLD_MODEL).encode(qtexts,batch_size=64,show_progress_bar=False,normalize_embeddings=True),dtype=np.float32); np.save(p,q_old)
def recall(top):
    h=t=0
    for i,qid in enumerate(q_ids):
        rel=relevant[qid]; got={doc_ids[j] for j in top[i]}; h+=len(got&rel); t+=len(rel)
    return h/t
def exact(qv,dv): return recall(np.argpartition(-(qv@dv.T),K,axis=1)[:,:K])
res={"seeds":SEEDS,"fractions":[float(f) for f in FR]}
res["baselines"]={"oldq_vs_olddocs":exact(q_old,docs_old),"oldq_vs_newdocs":exact(q_old,docs_new),"oldq_vs_fardocs":exact(q_old,docs_far)}
log("=== queries-last baselines:", {k:round(v,3) for k,v in res["baselines"].items()})
for name,target in [("minilm_pair",docs_new),("distant_pair",docs_far)]:
    curves=[]
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(N); c=[]
        for f in FR:
            n=int(f*N); mixed=docs_old.copy(); mixed[order[:n]]=target[order[:n]]; c.append(exact(q_old,mixed))
        curves.append(c); log(f"  {name} seed {seed}: min {min(c):.3f} @ {FR[int(np.argmin(c))]:.1f}  start {c[0]:.3f} end {c[-1]:.3f}")
    res[name+"_exact"]=curves
    m=np.mean(curves,axis=0); log("  frac  "+" ".join(f"{f:.1f}" for f in FR)); log("  mean  "+" ".join(f"{v:.2f}" for v in m))
# HNSW in-place, MiniLM pair, old queries
M,EFC,EF=16,200,50; curves=[]
for seed in SEEDS:
    order=np.random.default_rng(seed).permutation(N)
    idx=hnswlib.Index(space="cosine",dim=384); idx.init_index(max_elements=N,ef_construction=EFC,M=M,random_seed=100); idx.add_items(docs_old,np.arange(N)); idx.set_ef(EF)
    prev=0; c=[]
    for f in FR:
        n=int(f*N)
        if n>prev: idx.add_items(docs_new[order[prev:n]],order[prev:n]); prev=n
        lab,_=idx.knn_query(q_old,k=K); c.append(recall(lab))
    curves.append(c); log(f"  hnsw seed {seed}: min {min(c):.3f} @ {FR[int(np.argmin(c))]:.1f}")
res["minilm_pair_hnsw_inplace"]=curves; m=np.mean(curves,axis=0); log("  hnsw mean "+" ".join(f"{v:.2f}" for v in m))
json.dump(res,open("stage5_results.json","w"),indent=1); log("saved stage5_results.json")
