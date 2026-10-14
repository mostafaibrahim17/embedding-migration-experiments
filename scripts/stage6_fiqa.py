"""Stage 6: repeat stages 3 and 5 on BeIR/fiqa (~57k docs, 648 test queries)."""
import json, os, time
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import hnswlib
DS="fiqa"; OLD="sentence-transformers/all-MiniLM-L6-v2"; NEW="sentence-transformers/paraphrase-MiniLM-L6-v2"; FAR="BAAI/bge-small-en-v1.5"
K=10; SEEDS=[0,1,2,3,4]; FR=np.linspace(0,1,11); CACHE=os.environ.get("EMB_CACHE",os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fiqa_results.json")
def log(*a): print(*a, flush=True)
t0=time.time()
corpus=load_dataset(f"BeIR/{DS}","corpus")["corpus"]; queries=load_dataset(f"BeIR/{DS}","queries")["queries"]; qrels=load_dataset(f"BeIR/{DS}-qrels")["test"]
doc_text={d["_id"]:(d.get("title","")+" "+d["text"]).strip() for d in corpus}; query_text={q["_id"]:q["text"] for q in queries}
relevant=defaultdict(set)
for r in qrels:
    if r["score"]>0: relevant[str(r["query-id"])].add(str(r["corpus-id"]))
doc_ids=list(doc_text.keys()); doc_index={d:i for i,d in enumerate(doc_ids)}
q_ids=[q for q in relevant if q in query_text and any(d in doc_index for d in relevant[q])]; N=len(doc_ids)
log(f"{DS}: {N} docs, {len(q_ids)} queries"); docs=[doc_text[d] for d in doc_ids]; qtexts=[query_text[q] for q in q_ids]
rel_rows=[np.array([doc_index[d] for d in relevant[q] if d in doc_index]) for q in q_ids]
def embed(model,texts,tag):
    p=os.path.join(CACHE,f"emb_{DS}_{tag}.npy")
    if os.path.exists(p): return np.load(p)
    log(f"  embedding {len(texts)} with {model} ..."); v=np.asarray(SentenceTransformer(model).encode(texts,batch_size=128,show_progress_bar=False,normalize_embeddings=True),dtype=np.float32); np.save(p,v); return v
docs_old=embed(OLD,docs,"docs_old"); docs_new=embed(NEW,docs,"docs_new"); docs_far=embed(FAR,docs,"docs_far")
q_new=embed(NEW,qtexts,"q_new"); q_far=embed(FAR,qtexts,"q_far"); q_old=embed(OLD,qtexts,"q_old")
log(f"embeddings ready ({time.time()-t0:.0f}s)")
def recall(top):
    h=t=0
    for i,qid in enumerate(q_ids):
        rel=relevant[qid]; got={doc_ids[j] for j in top[i]}; h+=len(got&rel); t+=len(rel)
    return h/t
def exact_topk(qv,dv):
    top=np.empty((qv.shape[0],K),dtype=int); sims=np.empty((qv.shape[0],dv.shape[0]),dtype=np.float32)
    for i in range(0,qv.shape[0],64):
        s=qv[i:i+64]@dv.T; sims[i:i+64]=s; top[i:i+64]=np.argpartition(-s,K,axis=1)[:,:K]
    return top,sims
def exact(qv,dv): return recall(exact_topk(qv,dv)[0])
def norm_top(sims,mask):
    out=np.empty((sims.shape[0],K),dtype=int)
    for i in range(sims.shape[0]):
        s=sims[i].copy()
        for g in (mask,~mask):
            if g.sum()>=2: x=s[g]; s[g]=(x-x.mean())/(x.std()+1e-9)
        out[i]=np.argpartition(-s,K)[:K]
    return out
def overlap(a,b,n=3000):
    idx=np.random.default_rng(0).choice(N,size=min(n,N),replace=False)
    def nn(v):
        s=v[idx]@v.T; s[np.arange(len(idx)),idx]=-1; return np.argpartition(-s,10,axis=1)[:,:10]
    return float(np.mean([len(set(x)&set(y))/10 for x,y in zip(nn(a),nn(b))]))
R={"dataset":DS,"n_docs":N,"n_queries":len(q_ids),"seeds":SEEDS,"fractions":[float(f) for f in FR]}
R["baselines"]={"new_new":exact(q_new,docs_new),"new_old":exact(q_new,docs_old),"old_old":exact(q_old,docs_old),"old_new":exact(q_old,docs_new),"far_far":exact(q_far,docs_far),"far_old":exact(q_far,docs_old),"old_far":exact(q_old,docs_far),"overlap_minilm":overlap(docs_old,docs_new),"overlap_far":overlap(docs_old,docs_far)}
log("baselines:",{k:round(v,3) for k,v in R["baselines"].items()})
# queries-first exact, both pairs, + top10 share (MiniLM)
for name,tgt,qv in [("minilm",docs_new,q_new),("distant",docs_far,q_far)]:
    C={"raw":[],"norm":[],"share":[]}
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(N); r=[];n_=[];sh=[]
        for f in FR:
            n=int(f*N); mixed=docs_old.copy(); mixed[order[:n]]=tgt[order[:n]]; mask=np.zeros(N,bool); mask[order[:n]]=True
            top,sims=exact_topk(qv,mixed); r.append(recall(top)); n_.append(recall(norm_top(sims,mask))); sh.append(float(mask[top].mean()))
        C["raw"].append(r); C["norm"].append(n_); C["share"].append(sh); log(f"  qfirst {name} seed {seed}: trough {min(r):.3f} @ {FR[int(np.argmin(r))]:.1f}")
    R[f"qfirst_{name}"]=C; log(f"  qfirst {name} mean raw: "+" ".join(f"{v:.2f}" for v in np.mean(C['raw'],axis=0))+" | norm: "+" ".join(f"{v:.2f}" for v in np.mean(C['norm'],axis=0))+" | share: "+" ".join(f"{v:.2f}" for v in np.mean(C['share'],axis=0)))
# HNSW (MiniLM, queries-first), tune ef
M,EFC=16,200
def build(vecs,ef):
    idx=hnswlib.Index(space="cosine",dim=vecs.shape[1]); idx.init_index(max_elements=N,ef_construction=EFC,M=M,random_seed=100); idx.add_items(vecs,np.arange(N),num_threads=8); idx.set_ef(ef); return idx
base=R["baselines"]["new_new"]; ef_used=None
for ef in [50,100,200,400,800]:
    idx=build(docs_new,ef); lab,_=idx.knn_query(q_new,k=K); rr=recall(lab); log(f"  ef={ef}: {rr:.3f} vs exact {base:.3f}")
    if rr>=0.98*base: ef_used=ef; break
ef_used=ef_used or 800; R["hnsw"]={"M":M,"ef_construction":EFC,"ef_search":ef_used,"protocols":{}}
def heval(idx,mask,ef):
    labs,d=idx.knn_query(q_new,k=min(ef,N)); sims=1.0-d; raw=recall(labs[:,:K]); nt=np.empty((len(q_ids),K),dtype=int)
    for i in range(len(q_ids)):
        s=sims[i].copy(); mm=mask[labs[i]]
        for g in (mm,~mm):
            if g.sum()>=2: x=s[g]; s[g]=(x-x.mean())/(x.std()+1e-9)
        nt[i]=labs[i][np.argsort(-s)[:K]]
    nr=recall(nt); reach=cnt=0.0
    for i in range(len(q_ids)):
        ro=[x for x in rel_rows[i] if not mask[x]]
        if not ro: continue
        c=set(labs[i].tolist()); reach+=sum(1 for x in ro if x in c)/len(ro); cnt+=1
    return raw,nr,(reach/cnt if cnt else float("nan"))
for proto in ("A_inplace_update","B_rebuild_on_mix"):
    C={"raw":[],"norm":[],"reach":[]}
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(N); r=[];n_=[];re_=[]; idx=build(docs_old,ef_used) if proto.startswith("A") else None; prev=0
        for f in FR:
            n=int(f*N); mask=np.zeros(N,bool); mask[order[:n]]=True
            if proto.startswith("A"):
                if n>prev: idx.add_items(docs_new[order[prev:n]],order[prev:n],num_threads=8); prev=n
            else:
                mixed=docs_old.copy(); mixed[order[:n]]=docs_new[order[:n]]; idx=build(mixed,ef_used)
            a,b,c=heval(idx,mask,ef_used); r.append(a); n_.append(b); re_.append(c)
        C["raw"].append(r); C["norm"].append(n_); C["reach"].append(re_); log(f"  hnsw {proto} seed {seed}: trough {min(r):.3f} @ {FR[int(np.argmin(r))]:.1f}  norm-min {min(n_):.3f}  reach@0.2 {re_[2]:.2f}")
    R["hnsw"]["protocols"][proto]=C
    log(f"  {proto} mean raw: "+" ".join(f"{v:.2f}" for v in np.mean(C['raw'],axis=0))+" | norm: "+" ".join(f"{v:.2f}" for v in np.mean(C['norm'],axis=0))+" | reach: "+" ".join(f"{v:.2f}" for v in np.nanmean(C['reach'],axis=0)))
# queries-last exact, both pairs
for name,tgt in [("minilm",docs_new),("distant",docs_far)]:
    C=[]
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(N); r=[]
        for f in FR:
            n=int(f*N); mixed=docs_old.copy(); mixed[order[:n]]=tgt[order[:n]]; r.append(exact(q_old,mixed))
        C.append(r); log(f"  qlast {name} seed {seed}: trough {min(r):.3f} @ {FR[int(np.argmin(r))]:.1f}")
    R[f"qlast_{name}"]=C; log(f"  qlast {name} mean: "+" ".join(f"{v:.2f}" for v in np.mean(C,axis=0)))
json.dump(R,open(OUT,"w"),indent=1); log(f"saved {OUT} ({time.time()-t0:.0f}s)")
