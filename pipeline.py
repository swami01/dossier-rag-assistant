"""Dossier RAG pipeline: access control + metadata filters + hybrid (vector+BM25) + rerank + grounded LLM answer."""
import json
import os
import re

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

BASE = os.path.dirname(os.path.abspath(__file__))    # works from any working directory
DATA = os.path.join(BASE, "data", "dossiers.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
MAX_DIST = 0.9          # relevance gate: tune with your own distance printouts
NO_ANSWER = "I don't know based on the available dossiers."

NATIONALITIES = ["Nigerian", "Ghanaian", "Bangladeshi", "Nepali", "Afghan",
                 "Ugandan", "Kenyan", "Sudanese", "Sri Lankan", "Iranian"]
OFFICES = ["FRRO_MUMBAI", "FRO_NAGPUR", "FRO_PUNE", "FRO_THANE", "FRO_RAIPUR",
           "FRO_BILASPUR", "FRO_INDORE", "FRO_BHOPAL", "FRO_GOA"]
ID_RE = re.compile(r"DOS/\d{6}/\d{4}", re.I)

# ---------- ingest ----------
def to_text(r):
    return (f"Dossier {r['dossier_no']}: {r['name']} {r['surname']}, "
            f"{r['nationality']} national, {r['entry_type']} entry, "
            f"{r['office']} ({r['state']}). Status: {r['status']}. "
            f"{r['case_notes']} {r['deportation_notes']}")

col = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db")).get_or_create_collection("dossiers")

if col.count() == 0:
    rows = json.load(open(DATA, encoding="utf-8"))
    for i in range(0, len(rows), 1000):
        c = rows[i:i + 1000]
        col.add(ids=[r["dossier_no"] for r in c],
                documents=[to_text(r) for r in c],
                metadatas=[{"office": r["office"], "state": r["state"], "dossier_no": r["dossier_no"],
                            "nationality": r["nationality"], "entry_type": r["entry_type"]} for r in c])
        print(f"added {i + len(c)} / {len(rows)}")

# ---------- access control & filters ----------
def access_filter(user):
    return None if user["role"] == "admin" else {"office": user["office"]}

def extract_filters(question):
    q = question.lower()
    f = {}
    for n in NATIONALITIES:
        if n.lower() in q:
            f["nationality"] = n
    if "illegal" in q:                      # check illegal BEFORE legal
        f["entry_type"] = "Illegal"
    elif "legal" in q:
        f["entry_type"] = "Legal"
    return f

def build_where(question, user, use_filters=True):
    conds = [c for c in [access_filter(user)] if c]
    if use_filters:
        conds += [{k: v} for k, v in extract_filters(question).items()]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}

def out_of_scope(question, user):
    """True if the question names another office's city but not the user's own."""
    if user["role"] == "admin":
        return False
    city = lambda o: o.split("_")[-1].lower()
    q = question.lower()
    return (any(city(o) in q for o in OFFICES if o != user["office"])
            and city(user["office"]) not in q)

# ---------- retrieval ----------
def tok(s):
    return re.findall(r"[a-z0-9/]+", s.lower())

_bm25 = {}
def get_bm25(office):                       # office=None -> all (admin only)
    if office not in _bm25:
        got = col.get(where={"office": office} if office else None)
        _bm25[office] = (BM25Okapi([tok(d) for d in got["documents"]]),
                         got["ids"], got["documents"], got["metadatas"])
    return _bm25[office]

STOP = set("""the a an of in on to is are was were be been who what which how many much all any me my
show tell give list find get details detail info information about from for with by at and or please
can could you your their they them his her it this that these those do does did have has had where when
passengers passenger people person persons records record dossiers dossier""".split())

def keyword_hit(question, user):
    """True if the question has a distinctive word that occurs in the user's visible records."""
    office = user["office"] if user["role"] != "admin" else None
    idf = get_bm25(office)[0].idf
    return any(t not in STOP and idf.get(t, -1) > 0.5 for t in tok(question))

def relevant(question, where, user):
    """Gate: answer only if a close vector match OR a distinctive keyword match exists."""
    if ID_RE.search(question) or keyword_hit(question, user):
        return True
    d = col.query(query_texts=[question], n_results=1, where=where)["distances"][0]
    return bool(d) and d[0] <= MAX_DIST

def _matches(meta, filters):
    return all(meta.get(k) == v for k, v in filters.items())

def hybrid(question, user, n=5, pool=30, use_filters=True):
    where = build_where(question, user, use_filters)
    v_ids = col.query(query_texts=[question], n_results=pool, where=where)["ids"][0]
    office = user["office"] if user["role"] != "admin" else None
    bm, ids, _, metas = get_bm25(office)
    scores = bm.get_scores(tok(question))
    filters = extract_filters(question) if use_filters else {}
    top = sorted(range(len(ids)), key=lambda i: -scores[i])[:pool]
    b_ids = [ids[i] for i in top if scores[i] > 0 and _matches(metas[i], filters)]
    fused = {}                               # reciprocal rank fusion
    for lst in (v_ids, b_ids):
        for rank, id_ in enumerate(lst):
            fused[id_] = fused.get(id_, 0) + 1 / (60 + rank)
    best = sorted(fused, key=fused.get, reverse=True)[:n]
    if not best:
        return [], []
    got = col.get(ids=best)
    order = {i: k for k, i in enumerate(best)}
    pairs = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda p: order[p[0]])
    return [p[1] for p in pairs], [p[2] for p in pairs]

_ce = None
def rerank(question, docs, metas, n=5):
    global _ce
    if _ce is None:
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _ce.predict([(question, d) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: -scores[i])[:n]
    return [docs[i] for i in order], [metas[i] for i in order]

def retrieve(question, user, mode="rerank", n=5):
    """mode: vector | filters | hybrid | rerank. Returns (docs, metas); empty = nothing to answer from."""
    if out_of_scope(question, user):
        return [], []
    m = ID_RE.search(question)
    if m:                                    # exact ID lookup, still access-checked
        got = col.get(ids=[m.group(0).upper()], where=access_filter(user))
        return got["documents"], got["metadatas"]
    where = build_where(question, user, use_filters=(mode != "vector"))
    if not relevant(question, where, user):
        return [], []
    if mode in ("vector", "filters"):
        res = col.query(query_texts=[question], n_results=n, where=where)
        return res["documents"][0], res["metadatas"][0]
    if mode == "hybrid":
        return hybrid(question, user, n=n)
    docs, metas = hybrid(question, user, n=30)
    return rerank(question, docs, metas, n=n) if docs else ([], [])

# ---------- generation ----------

GREETINGS = {"hi", "hello", "hey", "how are you", "hello how are you", "hi how are you",
             "good morning", "good afternoon", "good evening"}

def build_prompt(question, docs):
    return f"""You are an assistant for a dossier management system. Use ONLY the dossier records below, never outside knowledge.
- Answer the question from these records and cite the dossier numbers, like DOS/000012/2026.
- If the records only partly match, summarise what they do show and say what matches.
- Reply exactly "{NO_ANSWER}" only if none of the records relate to the question.
- Report only facts written in the records. Never copy details from the question.


Context:
{chr(10).join(docs)}

Question: {question}
Answer:"""

def generate(prompt):
    r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    return r.json()["response"]

def ask(question, user, mode="rerank"):
    docs, metas = retrieve(question, user, mode)
    if not docs:                             # no context -> never call the LLM
        return {"answer": NO_ANSWER, "sources": []}
    return {"answer": generate(build_prompt(question, docs)),
            "sources": [m["dossier_no"] for m in metas]}

# ---------- demo users ----------
admin = {"role": "admin", "office": None}
def office_user(name):
    return {"role": "office", "office": name}

if __name__ == "__main__":
    for u in (office_user("FRO_NAGPUR"), office_user("FRO_PUNE")):   # leak test
        _, metas = retrieve("Nigerian illegal entry cases", u, "rerank", n=50)
        assert all(m["office"] == u["office"] for m in metas), "LEAK!"
    print("access control OK\n")
    for q, u in [("Show Nigerian illegal entry cases", office_user("FRO_NAGPUR")),
                 ("Show me dossiers of pune", office_user("FRO_NAGPUR")),
                 ("pizza recipe", office_user("FRO_NAGPUR"))]:
        print("Q:", q, "\n", ask(q, u), "\n")
