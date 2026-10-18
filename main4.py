#!/usr/bin/env python3
"""
DVMN 0.5 — Déjà Vu Memory Network
A research prototype for episodic familiarity, concept reactivation, and relational recall.

Python 3.10+; standard library only. No API key or cloud service required.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sqlite3, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.5.1-final"
DB_DEFAULT = "dvmn05_final.db"
MODEL_DEFAULT = "dvmn05_model.json"
STOP = set("""a an the and or but if then than of on in at to from by for with without as is are was were be been being do does did can could should would may might will this that these those it its we our you your i me my they them their he she his her what which who when where why how not no yes into over under again new old more most less very about across between through use uses using used system model memory conversation episode context past previous similar related retrieve retrieval answer question idea concept concepts relation relations ai artificial intelligence""".split())
ALIASES = {
 "vision": {"visual","camera","image","images","seeing","sight","optical","video","frame","frames"},
 "navigation": {"navigate","navigating","pathfinding","route","movement","motion","robot","robotics"},
 "memory": {"remember","recall","retrieval","retrieve","episodic","episode","episodes","remembering","familiarity"},
 "language_model": {"llm","language","transformer","chatbot","prompt","prompts","token","tokens"},
 "learning": {"train","training","trained","learn","learned","learning","optimization","optimize"},
 "neural_network": {"neural","network","networks","neuron","neurons","deep","backpropagation"},
 "fly": {"fruitfly","fruit-fly","drosophila","insect","insects","flight","flying"},
 "reasoning": {"infer","inference","reason","reasoning","logic","deduce","deduction"},
 "time": {"temporal","timeline","sequence","sequential","chronology","chronological"},
 "graph": {"relationship","relationships","linked","link","links","connection","connections","graph"},
 "rag": {"retrieval-augmented","augmented","vector","embedding","embeddings","semantic"},
 "evaluation": {"benchmark","benchmarks","metric","metrics","precision","recall","accuracy","f1"},
}
RELATION_PATTERNS = [
 ("causes", r"\b(cause|causes|caused|leads to|result in|results in)\b"),
 ("uses", r"\b(use|uses|using|used|employs|employ)\b"),
 ("improves", r"\b(improve|improves|improved|enhance|enhances|enhanced)\b"),
 ("retrieves", r"\b(retrieve|retrieves|retrieved|recall|recalls|remember|remembers)\b"),
 ("compares", r"\b(compare|compares|compared|versus|vs\.?|baseline)\b"),
 ("trains", r"\b(train|trains|trained|training|learns|learned)\b"),
 ("detects", r"\b(detect|detects|detected|recognize|recognizes|recognition)\b"),
 ("guides", r"\b(guide|guides|guided|steer|steers|influence|influences)\b"),
]

def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def tokens(s):
    return [x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", s.lower()) if x.lower() not in STOP]

def concepts(s):
    ts = set(tokens(s))
    out = set()
    for canonical, aliases in ALIASES.items():
        if canonical in ts or ts.intersection(aliases):
            out.add(canonical)
    # Retain informative terms as concepts too, but avoid over-weighting generic words.
    out.update(t for t in ts if len(t) >= 5 and t not in {a for vals in ALIASES.values() for a in vals})
    return sorted(out)

def relations(s):
    low = s.lower()
    return sorted(name for name, pattern in RELATION_PATTERNS if re.search(pattern, low))

def connect(db):
    """Open the final database and safely migrate the earlier DVMN episode schema."""
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("""CREATE TABLE IF NOT EXISTS episodes(
      id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, content TEXT NOT NULL,
      created_at TEXT NOT NULL, concepts_json TEXT NOT NULL, relations_json TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'user')""")
    cols = {r[1] for r in c.execute("PRAGMA table_info(episodes)")}
    # Older scripts created episodes without an explicit id column. SQLite rowid preserves
    # their records; copy rowid into a normal id column so all old episodes remain accessible.
    if "id" not in cols:
        c.execute("ALTER TABLE episodes ADD COLUMN id INTEGER")
        c.execute("UPDATE episodes SET id=rowid WHERE id IS NULL")
    cols = {r[1] for r in c.execute("PRAGMA table_info(episodes)")}
    for name, decl, default in [
        ("title", "TEXT", "''"), ("content", "TEXT", "''"),
        ("created_at", "TEXT", "''"), ("concepts_json", "TEXT", "'[]'"),
        ("relations_json", "TEXT", "'[]'"), ("source", "TEXT", "'legacy'")]:
        if name not in cols:
            c.execute(f"ALTER TABLE episodes ADD COLUMN {name} {decl} NOT NULL DEFAULT {default}")
    c.execute("CREATE TABLE IF NOT EXISTS feedback( id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, episode_id INTEGER NOT NULL, label INTEGER NOT NULL CHECK(label IN (0,1)), created_at TEXT NOT NULL)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_id ON episodes(id)")
    c.commit()
    return c

def episode_add(c, title, content, source="user", created_at=None):
    title, content = str(title).strip(), str(content).strip()
    if not content:
        raise ValueError("Episode content cannot be empty.")
    cs = sorted(set(concepts(title + " " + content)))
    rs = sorted(set(relations(title + " " + content)))
    cur = c.execute("INSERT INTO episodes(title,content,created_at,concepts_json,relations_json,source) VALUES(?,?,?,?,?,?)",
        (title or "Untitled episode", content, created_at or utcnow(), json.dumps(cs), json.dumps(rs), source))
    # Legacy databases may have an ordinary id column rather than an INTEGER PRIMARY KEY.
    c.execute("UPDATE episodes SET id=rowid WHERE id IS NULL")
    c.commit()
    return cur.lastrowid

def all_episodes(c):
    return [dict(r) for r in c.execute("SELECT * FROM episodes ORDER BY id")]

def jload(s):
    try: return json.loads(s or "[]")
    except Exception: return []

def overlap(a,b):
    a,b=set(a),set(b)
    return len(a & b) / max(1, len(a | b))

def cosine_counts(a,b):
    ca,cb=Counter(a),Counter(b)
    dot=sum(v*cb.get(k,0) for k,v in ca.items())
    na=math.sqrt(sum(v*v for v in ca.values())); nb=math.sqrt(sum(v*v for v in cb.values()))
    return dot/(na*nb) if na and nb else 0.0

def make_graph(episodes):
    """Concept co-activation graph built from episodes, not a vector store."""
    g=defaultdict(Counter)
    for e in episodes:
        cs=jload(e["concepts_json"])
        for i,a in enumerate(cs):
            for b in cs[i+1:]:
                g[a][b] += 1; g[b][a] += 1
    return g

def spread_score(qconcepts, econcepts, graph):
    """One-step spreading activation: query concepts activate their episode-linked neighbors."""
    if not qconcepts or not econcepts: return 0.0
    activated=Counter()
    for q in qconcepts:
        activated[q] += 1.0
        for neigh, weight in graph.get(q, {}).items():
            activated[neigh] += 0.55 * (weight / (weight + 2.0))
    total=sum(activated.values()) or 1.0
    hit=sum(activated[x] for x in econcepts if x in activated)
    return min(1.0, hit/total * min(2.0, len(qconcepts)**0.5))

def temporal_score(query_time, episode_time):
    # Recency is a weak contextual cue, not a relevance substitute.
    try:
        a=datetime.fromisoformat(query_time.replace("Z","+00:00"))
        b=datetime.fromisoformat(episode_time.replace("Z","+00:00"))
        if a.tzinfo is None: a=a.replace(tzinfo=timezone.utc)
        if b.tzinfo is None: b=b.replace(tzinfo=timezone.utc)
        days=abs((a-b).total_seconds())/86400
        return math.exp(-days/180.0)
    except Exception: return 0.5

FEATURE_NAMES = ["lexical","concept","relation","spreading","title_hint","temporal","concept_density"]

def feature_vector(query, e, graph):
    qtext=tokens(query); etext=tokens(e["title"]+" "+e["content"])
    qc=concepts(query); ec=jload(e["concepts_json"]); qr=relations(query); er=[r for r in jload(e["relations_json"]) if isinstance(r, str)]
    lexical=cosine_counts(qtext,etext)
    con=overlap(qc,ec)
    rel=overlap(qr,er) if qr else 0.0
    spread=spread_score(qc,ec,graph)
    qt=set(qtext); title=set(tokens(e["title"]))
    title_hint=len(qt & title)/max(1,len(qt))
    temporal=temporal_score(utcnow(),e["created_at"])
    density=len(set(ec)&set(qc))/max(1,len(set(qc)))
    return [lexical,con,rel,spread,title_hint,temporal,density]

def sigmoid(x):
    if x >= 35: return 1.0
    if x <= -35: return 0.0
    return 1.0/(1.0+math.exp(-x))

def init_model():
    return {"version": VERSION, "feature_names": FEATURE_NAMES,
      "mean":[0.0]*len(FEATURE_NAMES), "scale":[1.0]*len(FEATURE_NAMES),
      "w":[0.0]*len(FEATURE_NAMES), "b":-0.3, "trained":False,
      "training_examples":0, "training_note":"Untrained: uses deterministic fallback scorer."}

def model_load(path):
    if not os.path.exists(path): return init_model()
    try:
        m=json.loads(Path(path).read_text(encoding="utf-8"))
        if m.get("feature_names") != FEATURE_NAMES: return init_model()
        return m
    except Exception: return init_model()

def model_save(m,path):
    Path(path).write_text(json.dumps(m,indent=2),encoding="utf-8")

def score_features(x, model):
    if model.get("trained"):
        z=[(v-mu)/(sd or 1.0) for v,mu,sd in zip(x,model["mean"],model["scale"])]
        return sigmoid(model["b"]+sum(a*b for a,b in zip(z,model["w"])))
    # Transparent fallback before training.
    weights=[0.25,0.24,0.10,0.24,0.07,0.02,0.08]
    return min(1.0,sum(a*b for a,b in zip(x,weights)))

def retrieve(query, episodes, model, threshold=0.18, limit=5):
    graph=make_graph(episodes)
    results=[]
    for e in episodes:
        x=feature_vector(query,e,graph)
        s=score_features(x,model)
        results.append({"id":e["id"],"title":e["title"],"content":e["content"],"source":e["source"],
          "created_at":e["created_at"],"score":round(s,5),
          "features":{n:round(v,4) for n,v in zip(FEATURE_NAMES,x)},
          "matched_concepts":sorted(set(concepts(query)) & set(jload(e["concepts_json"]))),
          # Legacy databases can contain nested lists in relations_json; only
          # string relation labels are valid set members and printable labels.
          "matched_relations":sorted(
              set(relations(query)) & {
                  r for r in jload(e["relations_json"]) if isinstance(r, str)
              }
          )})
    results.sort(key=lambda r:r["score"],reverse=True)
    return [r for r in results if r["score"] >= threshold][:limit], results

def seed_demo(c):
    samples=[
      ("Visual navigation","A robot uses camera frames and visual motion cues to navigate an unfamiliar environment."),
      ("Episodic memory","A memory system stores past conversations as episodes and retrieves them when a new discussion reactivates related concepts."),
      ("Fruit fly vision","Fruit flies use visual motion signals to stabilize flight and guide navigation."),
      ("Neural training","A neural network learns relevance weights from labelled query-memory pairs using gradient descent."),
      ("RAG baseline","Retrieval augmented generation searches stored documents and supplies relevant text to a language model."),
      ("Concept graph","A concept graph connects concepts that co-occur in remembered episodes and spreads activation across related ideas."),
      ("Temporal context","A temporal memory system uses episode order and recency as contextual cues without treating recent items as automatically relevant."),
      ("Reasoning evaluation","A held-out benchmark compares memory recall using precision, recall, F1, and false positive rate."),
      ("Deja vu hypothesis","The DVMN hypothesis is that reactivated concepts and relations can create a familiarity signal before full episode retrieval."),
      ("Language model","A local language model can use retrieved memories to compose a response, but it does not train itself merely by reading them.")
    ]
    existing=c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    if existing:
        print(f"Database already has {existing} episodes; seed-demo skipped to avoid duplicates.")
        return
    for title,content in samples: episode_add(c,title,content,"demo")
    print(f"Added {len(samples)} demo episodes.")

def train_model(c, model_path, epochs=100, lr=0.05, seed=7):
    """Train a small one-layer neural relevance model on feedback and transparent pseudo-labels.
    Explicit feedback is preferred; pseudo-labels are only a bootstrap signal."""
    random.seed(seed)
    eps=all_episodes(c)
    if len(eps)<2: raise SystemExit("Add at least 2 episodes before training.")
    graph=make_graph(eps)
    examples=[]
    # Explicit human labels, if supplied.
    for fb in c.execute("SELECT query,episode_id,label FROM feedback"):
        e=next((x for x in eps if x["id"]==fb["episode_id"]),None)
        if e: examples.append((feature_vector(fb["query"],e,graph),float(fb["label"])))
    # Bootstrap pseudo-labels: deliberately conservative and explicitly reported.
    if len(examples)<20:
        for e in eps:
            text=e["title"]+" "+e["content"]
            cs=jload(e["concepts_json"])
            if len(cs)>=2:
                examples.append((feature_vector(text,e,graph),1.0))
            # Hard negatives are episodes with minimal concept overlap.
            candidates=[other for other in eps if other["id"]!=e["id"]]
            candidates.sort(key=lambda o: overlap(cs,jload(o["concepts_json"])))
            for other in candidates[:min(2,len(candidates))]:
                examples.append((feature_vector(text,other,graph),0.0))
    if len(examples)<4: raise SystemExit("Not enough training pairs. Import more episodes or add feedback.")
    # Normalize feature values.
    n=len(FEATURE_NAMES)
    means=[sum(x[i] for x,y in examples)/len(examples) for i in range(n)]
    scales=[]
    for i in range(n):
        var=sum((x[i]-means[i])**2 for x,y in examples)/len(examples)
        scales.append(max(math.sqrt(var),0.05))
    data=[([(x[i]-means[i])/scales[i] for i in range(n)],y) for x,y in examples]
    w=[random.uniform(-0.05,0.05) for _ in range(n)]; b=0.0
    random.shuffle(data)
    split=max(1,int(len(data)*0.8))
    train=data[:split]; valid=data[split:] or data[-1:]
    for epoch in range(epochs):
        random.shuffle(train); loss=0.0
        for x,y in train:
            p=sigmoid(b+sum(a*v for a,v in zip(w,x)))
            err=p-y
            loss += -(y*math.log(max(p,1e-9))+(1-y)*math.log(max(1-p,1e-9)))
            for i in range(n): w[i]-=lr*(err*x[i]+0.001*w[i])
            b-=lr*err
        if epoch in (0,9,24,49,epochs-1):
            vl=0
            for x,y in valid:
                p=sigmoid(b+sum(a*v for a,v in zip(w,x)))
                vl += -(y*math.log(max(p,1e-9))+(1-y)*math.log(max(1-p,1e-9)))
            print(f"epoch {epoch+1:>3}/{epochs} | train_loss={loss/len(train):.4f} | validation_loss={vl/len(valid):.4f}")
    model={"version":VERSION,"feature_names":FEATURE_NAMES,"mean":means,"scale":scales,"w":w,"b":b,
       "trained":True,"training_examples":len(examples),"explicit_feedback_examples":c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
       "training_note":"Trained scorer. If explicit_feedback_examples is low, much of the training signal is weakly supervised pseudo-labels."}
    model_save(model,model_path)
    print(f"\nSaved trained model: {model_path}")
    print(f"Training pairs: {len(examples)}; explicit human-labelled pairs: {model['explicit_feedback_examples']}")
    if model["explicit_feedback_examples"] == 0:
        print("RESEARCH NOTE: this is bootstrap training, not ground-truth validation. Add labels and evaluate on unseen queries.")

def cmd_import(c,path):
    """Import JSONL (one object per line) or a JSON array/object."""
    raw=Path(path).read_text(encoding="utf-8-sig")
    rows=[]
    try:
        parsed=json.loads(raw)
        if isinstance(parsed,list): rows=parsed
        elif isinstance(parsed,dict):
            candidate=parsed.get("episodes", parsed.get("memories", parsed))
            rows=candidate if isinstance(candidate,list) else [candidate]
        else: raise SystemExit("JSON file must contain an object or array.")
    except json.JSONDecodeError:
        for line_no,line in enumerate(raw.splitlines(),1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as ex: raise SystemExit(f"Invalid JSON/JSONL at line {line_no}: {ex}")
    count=0; skipped=0
    for i,row in enumerate(rows,1):
        if not isinstance(row,dict): skipped+=1; continue
        content=row.get("content") or row.get("text") or row.get("episode") or row.get("memory") or row.get("message") or ""
        title=row.get("title") or row.get("name") or row.get("topic") or f"Imported episode {i}"
        if isinstance(content,(dict,list)): content=json.dumps(content,ensure_ascii=False)
        if not str(content).strip(): skipped+=1; continue
        episode_add(c,title,str(content),row.get("source","import"),row.get("created_at"))
        count+=1
    print(f"Imported {count} episodes from {path}; skipped {skipped} empty/unsupported records.")
    print("Import appends records; it does not overwrite existing memories. Check `list` for the total.")


def cmd_reindex(c):
    """Re-extract concepts/relations from every saved episode, including legacy empty fields."""
    rows=c.execute("SELECT id,title,content,concepts_json,relations_json FROM episodes ORDER BY id").fetchall()
    updated=0
    for e in rows:
        cs=sorted(set(concepts((e["title"] or "")+" "+(e["content"] or ""))))
        rs=sorted(set(relations((e["title"] or "")+" "+(e["content"] or ""))))
        oldcs=jload(e["concepts_json"]); oldrs=jload(e["relations_json"])
        if not oldcs or not oldrs:
            c.execute("UPDATE episodes SET concepts_json=?,relations_json=? WHERE id=?",(json.dumps(cs),json.dumps(rs),e["id"]))
            updated+=1
    c.commit()
    print(f"Reindexed {updated} episodes with missing concept/relation metadata. Total memories: {len(rows)}.")


def cmd_export(c,path):
    rows=all_episodes(c)
    with open(path,"w",encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps({"title":e["title"],"content":e["content"],"created_at":e["created_at"],"source":e["source"]},ensure_ascii=False)+"\n")
    print(f"Exported {len(rows)} episodes to {path}.")


def cmd_ask(query, hits, model_name, endpoint):
    """Send retrieved episodes to an optional local Ollama model via its HTTP API."""
    import urllib.request, urllib.error
    if not hits:
        print("No memories were retrieved. Lower the threshold or add/import memories first.")
        return
    memories="\n\n".join(f"[Episode #{h['id']} | {h['title']} | DVMN familiarity={h['score']:.3f}]\n{h['content']}" for h in hits)
    prompt=("Answer the user's question using the recalled episodes as prior experience. "
            "Clearly distinguish facts from memories and say when the memories do not support a claim. "
            "Do not claim that these memories change your model weights.\n\n"
            f"Recalled episodes:\n{memories}\n\nQuestion: {query}")
    payload={"model":model_name,"stream":False,"messages":[{"role":"user","content":prompt}]}
    req=urllib.request.Request(endpoint.rstrip("/")+"/api/chat",data=json.dumps(payload).encode("utf-8"),headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=180) as resp: data=json.loads(resp.read().decode("utf-8"))
        print(f"\nDVMN {VERSION} + Ollama model {model_name}\n")
        print(data.get("message",{}).get("content","(Model returned no text.)"))
        print("\nMemories supplied to model:")
        for h in hits: print(f"  #{h['id']} {h['title']} (score={h['score']:.3f})")
    except Exception as ex:
        print(f"Could not call Ollama at {endpoint}: {ex}")
        print("Check that Ollama is running and that the model is installed: ollama list")

def cmd_feedback(c, query, episode_id, label):
    if not c.execute("SELECT 1 FROM episodes WHERE id=?",(episode_id,)).fetchone():
        raise SystemExit(f"No episode with id {episode_id}.")
    c.execute("INSERT INTO feedback(query,episode_id,label,created_at) VALUES(?,?,?,?)",(query,episode_id,label,utcnow()))
    c.commit()
    print(f"Saved label={label} for query and episode #{episode_id}.")

def cmd_evaluate(c, model, threshold=0.5):
    eps=all_episodes(c)
    rows=list(c.execute("SELECT query,episode_id,label FROM feedback"))
    if not rows:
        print("No human-labelled evaluation pairs yet.")
        print('Label results with: python main.py feedback "query text" EPISODE_ID 1')
        print("Use label 1 for relevant, 0 for irrelevant. Keep some queries entirely out of training for a valid held-out test.")
        return
    graph=make_graph(eps)
    tp=fp=tn=fn=0
    for r in rows:
        e=next((x for x in eps if x["id"]==r["episode_id"]),None)
        if not e: continue
        p=score_features(feature_vector(r["query"],e,graph),model)>=threshold
        y=bool(r["label"])
        if p and y: tp+=1
        elif p and not y: fp+=1
        elif not p and not y: tn+=1
        else: fn+=1
    prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); f1=2*prec*rec/max(1e-9,prec+rec)
    print(f"Labelled-pair evaluation at threshold {threshold:.2f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
    print("This is only a held-out test if these labels/queries were not used during training.")

def main():
    p=argparse.ArgumentParser(description="DVMN 0.5 — Déjà Vu Memory Network")
    p.add_argument("--db",default=DB_DEFAULT,help="SQLite database path")
    p.add_argument("--model",default=MODEL_DEFAULT,help="trained model JSON path")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("init",help="initialize database")
    sub.add_parser("seed-demo",help="insert ten example episodes into an empty database")
    sub.add_parser("reindex",help="rebuild missing concept/relation metadata for saved memories")
    ex=sub.add_parser("export-jsonl",help="export all saved memories as JSONL"); ex.add_argument("path")
    a=sub.add_parser("add",help="add an episode"); a.add_argument("content"); a.add_argument("--title",default="Untitled episode")
    imp=sub.add_parser("import-jsonl",help="import a JSONL episode dataset"); imp.add_argument("path")
    li=sub.add_parser("list",help="list episodes"); li.add_argument("--limit",type=int,default=50)
    rec=sub.add_parser("recall",help="retrieve related episodes"); rec.add_argument("query"); rec.add_argument("--threshold",type=float,default=0.18); rec.add_argument("--limit",type=int,default=5)
    tr=sub.add_parser("train",help="train relevance network"); tr.add_argument("--epochs",type=int,default=100); tr.add_argument("--lr",type=float,default=0.05); tr.add_argument("--seed",type=int,default=7)
    fb=sub.add_parser("feedback",help="label one query/episode pair"); fb.add_argument("query"); fb.add_argument("episode_id",type=int); fb.add_argument("label",type=int,choices=[0,1])
    ev=sub.add_parser("evaluate",help="evaluate saved feedback labels"); ev.add_argument("--threshold",type=float,default=0.5)
    ask=sub.add_parser("ask",help="retrieve memories and use an installed local Ollama model"); ask.add_argument("query"); ask.add_argument("--threshold",type=float,default=0.05); ask.add_argument("--limit",type=int,default=5); ask.add_argument("--llm",default="llama3.2",help="installed Ollama model name"); ask.add_argument("--endpoint",default="http://localhost:11434",help="Ollama URL")
    args=p.parse_args()
    c=connect(args.db)
    if args.cmd=="init":
        print(f"DVMN {VERSION} initialized. Database: {Path(args.db).resolve()}")
        print(f"Saved episodes: {c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]}")
    elif args.cmd=="seed-demo": seed_demo(c)
    elif args.cmd=="reindex": cmd_reindex(c)
    elif args.cmd=="export-jsonl": cmd_export(c,args.path)
    elif args.cmd=="add":
        eid=episode_add(c,args.title,args.content); print(f"Added episode #{eid}: {args.title}")
    elif args.cmd=="import-jsonl": cmd_import(c,args.path)
    elif args.cmd=="list":
        for e in all_episodes(c)[:args.limit]:
            print(f"#{e['id']} [{e['created_at']}] {e['title']} — {e['content'][:160]}")
        print(f"Total episodes: {c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]}")
    elif args.cmd=="recall":
        eps=all_episodes(c); model=model_load(args.model)
        hits,allrows=retrieve(args.query,eps,model,args.threshold,args.limit)
        print(f"DVMN {VERSION} | {'trained model' if model.get('trained') else 'bootstrap scorer'} | query: {args.query}")
        if not hits: print(f"No episode passed threshold {args.threshold:.3f}. Try a lower threshold, or add memories.")
        for i,r in enumerate(hits,1):
            print(f"\n{i}. #{r['id']} {r['title']} | score={r['score']:.3f}")
            print(f"   Concepts: {', '.join(r['matched_concepts'][:12]) or '(none detected)'}")
            print(f"   Relations: {', '.join(r['matched_relations']) or '(none detected)'}")
            print(f"   Why: {json.dumps(r['features'],sort_keys=True)}")
            print(f"   {r['content']}")
    elif args.cmd=="train": train_model(c,args.model,args.epochs,args.lr,args.seed)
    elif args.cmd=="feedback": cmd_feedback(c,args.query,args.episode_id,args.label)
    elif args.cmd=="evaluate": cmd_evaluate(c,model_load(args.model),args.threshold)
    elif args.cmd=="ask":
        eps=all_episodes(c); model=model_load(args.model)
        hits,_=retrieve(args.query,eps,model,args.threshold,args.limit)
        cmd_ask(args.query,hits,args.llm,args.endpoint)
    c.close()

if __name__=="__main__":
    main()
