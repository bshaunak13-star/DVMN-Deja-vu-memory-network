#!/usr/bin/env python3
"""DVMN 0.3: local associative episodic-memory research prototype. Python 3.10+."""
import argparse, json, math, re, sqlite3, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION="0.3.0"
DB=Path(__file__).with_name("dvmn_memory.db")
STOP=set("a an the and or but if then than of to in on at by for from with into is are was were be been being am do does did can could should would may might will this that these those it its as about how what why when where who which i you we they he she them us my your our their me him her not no yes very more most less also use using used make makes made build building model system memory concept concepts data".split())
ONTOLOGY={
"language_model":["llm","large language model","language model","chat model"],
"episodic_memory":["episodic memory","past episode","previous discussion","past conversation","conversation history"],
"concept_activation":["concept activation","spreading activation","associative activation"],
"semantic_similarity":["semantic similarity","similar meaning","semantic matching"],
"retrieval":["retrieval","recall","retrieve","remember","memory search"],
"deja_vu":["deja vu","déjà vu","familiarity signal","familiar feeling","familiarity"],
"neural_network":["neural network","neural model","deep learning","artificial neuron"],
"embedding":["embedding","embeddings","vector embedding","text vector"],
"computer_vision":["computer vision","visual input","camera input","image processing"],
"autonomous_navigation":["autonomous navigation","robot navigation","navigation"],
"robot":["robot","autonomous machine","mobile robot"],
"fruit_fly":["fruit fly","fruit-fly","drosophila","fly brain"],
"brain_architecture":["brain architecture","neural architecture","biological brain","neural circuitry"],
"evaluation":["evaluation","benchmark","test set","accuracy","precision","recall metric"],
"false_positive":["false positive","irrelevant retrieval","wrong memory"],
"concept_graph":["knowledge graph","concept graph","graph of concepts"],
"reinforcement_learning":["reinforcement learning","rlvr","verifiable rewards"],
"privacy":["privacy","private data","local only","data protection"],
"baking":["bake","baking","bread","flour","yeast","dough"],
"astronomy":["astronomy","stars","galaxy","milky way","planet"],
"materials_engineering":["carbon content","steel composition","material properties"]}
ALIASES={}
for c, vals in ONTOLOGY.items():
    ALIASES[c.replace("_"," ")]=c
    for v in vals: ALIASES[v]=c
PRED=[(r"\b(uses?|using|processes?|takes)\b","uses"),(r"\b(retrieves?|recalls?|remembers?)\b","retrieves"),(r"\b(activates?|activation|triggers?)\b","activates"),(r"\b(improves?|enhances?|boosts?)\b","improves"),(r"\b(connects?|links?|matches?)\b","connects"),(r"\b(stores?|records?|saves?)\b","stores"),(r"\b(navigates?)\b","navigates"),(r"\b(evaluates?|measures?|tests?)\b","evaluates")]

def stamp(): return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
def dbopen():
 c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
 c.executescript("""
 CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT,source TEXT DEFAULT 'user');
 CREATE TABLE IF NOT EXISTS episode_concepts(episode_id INTEGER,concept TEXT,PRIMARY KEY(episode_id,concept));
 CREATE TABLE IF NOT EXISTS episode_terms(episode_id INTEGER,term TEXT,PRIMARY KEY(episode_id,term));
 CREATE TABLE IF NOT EXISTS episode_relations(episode_id INTEGER,subject TEXT,predicate TEXT,object TEXT,PRIMARY KEY(episode_id,subject,predicate,object));
 CREATE TABLE IF NOT EXISTS concept_edges(source TEXT,target TEXT,weight REAL NOT NULL DEFAULT 1,PRIMARY KEY(source,target));
 """)
 # preserve and index older v0.1/v0.2 databases
 cols={r["name"] for r in c.execute("PRAGMA table_info(episodes)")}
 if "created_at" not in cols: c.execute("ALTER TABLE episodes ADD COLUMN created_at TEXT")
 if "source" not in cols: c.execute("ALTER TABLE episodes ADD COLUMN source TEXT DEFAULT 'migrated'")
 c.commit()
 for row in c.execute("SELECT id,title,content FROM episodes").fetchall():
  if not c.execute("SELECT 1 FROM episode_terms WHERE episode_id=? LIMIT 1",(row["id"],)).fetchone():
   cs=concepts(row["title"]+" "+row["content"]); ts=set(tokens(row["title"]+" "+row["content"]))
   c.executemany("INSERT OR IGNORE INTO episode_concepts VALUES(?,?)",[(row["id"],x) for x in cs])
   c.executemany("INSERT OR IGNORE INTO episode_terms VALUES(?,?)",[(row["id"],x) for x in ts])
   c.executemany("INSERT OR IGNORE INTO episode_relations VALUES(?,?,?,?)",[(row["id"],*x) for x in relations(row["content"],cs)])
 c.commit()
 if c.execute("SELECT COUNT(*) FROM concept_edges").fetchone()[0]==0:
  by=defaultdict(list)
  for r in c.execute("SELECT episode_id,concept FROM episode_concepts"): by[r["episode_id"]].append(r["concept"])
  for vals in by.values(): link_edges(c,vals,False)
 c.commit()
 return c

def tokens(s): return [w for w in re.findall(r"[a-z0-9]+",s.lower()) if len(w)>1 and w not in STOP]
def concepts(s):
 s=s.lower().replace("–","-"); out=set()
 for alias,c in sorted(ALIASES.items(),key=lambda x:len(x[0]),reverse=True):
  if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",s): out.add(c)
 return out
def relations(s,cs):
 out=set()
 for sent in re.split(r"[.!?;\n]+",s):
  found=sorted(concepts(sent))
  if len(found)<2: continue
  pred=next((p for pat,p in PRED if re.search(pat,sent,re.I)),None)
  if pred:
   for i in range(len(found)-1): out.add((found[i],pred,found[i+1]))
 return out
def link_edges(c, vals, increment=True):
 vals=sorted(set(vals))
 for i,a in enumerate(vals):
  for b in vals[i+1:]:
   for x,y in ((a,b),(b,a)):
    if increment:
     c.execute("INSERT INTO concept_edges VALUES(?,?,1) ON CONFLICT(source,target) DO UPDATE SET weight=MIN(5,weight+0.25)",(x,y))
    else: c.execute("INSERT OR IGNORE INTO concept_edges VALUES(?,?,1)",(x,y))
def add(c, title, content, source="user"):
    cs = concepts(title + " " + content)
    ts = set(tokens(title + " " + content))
    rs = relations(title + " " + content, cs)
    created = stamp()

    # Support the older DVMN 0.1 database schema too.
    columns = {
        row["name"]
        for row in c.execute("PRAGMA table_info(episodes)")
    }

    values = {
        "title": title.strip(),
        "content": content.strip(),
        "concepts_json": json.dumps(sorted(cs)),
        "relations_json": json.dumps(
            [list(r) for r in sorted(rs)]
        ),
        "created_at": created,
        "source": source,
    }

    insert_columns = [
        name for name in values if name in columns
    ]
    placeholders = ", ".join("?" for _ in insert_columns)
    column_sql = ", ".join(insert_columns)

    cur = c.execute(
        f"INSERT INTO episodes ({column_sql}) VALUES ({placeholders})",
        [values[name] for name in insert_columns],
    )
    eid = cur.lastrowid

    c.executemany(
        "INSERT OR IGNORE INTO episode_concepts VALUES (?, ?)",
        [(eid, x) for x in cs],
    )
    c.executemany(
        "INSERT OR IGNORE INTO episode_terms VALUES (?, ?)",
        [(eid, x) for x in ts],
    )
    c.executemany(
        "INSERT OR IGNORE INTO episode_relations VALUES (?, ?, ?, ?)",
        [(eid, *r) for r in rs],
    )

    link_edges(c, cs)
    c.commit()
    return eid

def seed(c):
 demos=[
 ("Fruit-fly neural architecture","I study the fruit fly brain and neural architecture to understand how a biological brain processes information while flying."),
 ("DVMN déjà vu memory idea","A language model uses episodic memory and a familiarity signal. Concept activation connects a new discussion to a previous conversation and retrieves the relevant episode."),
 ("Camera navigation system","An autonomous machine processes camera input using computer vision to navigate its environment. Visual input helps a robot choose actions."),
 ("Semantic retrieval benchmark","Evaluate memory retrieval using a benchmark with paraphrases, precision, recall, accuracy and false positive measurements."),
 ("Embedding-based memory search","Text embeddings represent semantic similarity so a language model can retrieve memories with similar meaning despite different wording."),
 ("Concept graph","A knowledge graph connects concepts through weighted relations. Spreading activation moves from one concept to related concepts and episodes."),
 ("RLVR optimization","Reinforcement learning with verifiable rewards trains a model using checks that determine whether an answer is correct."),
 ("Local memory privacy","A local database stores private conversation history on the computer without sending it to a cloud service."),
 ("Bread baking","To bake bread, mix flour, water and yeast, knead the dough, let it rise and bake it in an oven."),
 ("Astronomy observations","Astronomy involves observing stars, planets and the Milky Way galaxy in the night sky."),
 ("Materials engineering","Carbon content changes steel composition and can influence hardness, strength and material properties.")]
 existing={r["title"] for r in c.execute("SELECT title FROM episodes")}; n=0
 for t,b in demos:
  if t not in existing: add(c,t,b,"demo"); n+=1
 print(f"Added {n} demo memories; total episodes: {c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]}")
def cosine(a,b):
 dot=sum(v*b.get(k,0) for k,v in a.items()); na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
 return dot/(na*nb) if na and nb else 0
def weighted_docs(docs):
 df=Counter()
 for d in docs: df.update(d.keys())
 n=max(1,len(docs))
 return [Counter({k:v*(math.log((n+1)/(df[k]+1))+1) for k,v in d.items()}) for d in docs]
def activation(c,qcs):
 act={x:1.0 for x in qcs}; front=set(qcs)
 for depth in range(1,3):
  nxt=set()
  for src in front:
   for r in c.execute("SELECT target,weight FROM concept_edges WHERE source=?",(src,)):
    val=act.get(src,0)*min(1,r["weight"]/2)*(0.45**depth)
    if val>act.get(r["target"],0): act[r["target"]]=val; nxt.add(r["target"])
  front=nxt-set(qcs)
  if not front: break
 return act
def retrieve(c,query,limit=5,threshold=.20):
 rows=c.execute("SELECT * FROM episodes ORDER BY id DESC").fetchall()
 if not rows:return [],{"query_concepts":[],"threshold":threshold,"message":"No memories stored."}
 qt=set(tokens(query)); qc=concepts(query); qr=relations(query,qc)
 terms=[Counter(r["term"] for r in c.execute("SELECT term FROM episode_terms WHERE episode_id=?",(row["id"],))) for row in rows]
 qvec,*dvecs=weighted_docs(terms+[Counter(qt)])
 act=activation(c,qc); results=[]
 for row,dv in zip(rows,dvecs):
  ts={r["term"] for r in c.execute("SELECT term FROM episode_terms WHERE episode_id=?",(row["id"],))}
  cs={r["concept"] for r in c.execute("SELECT concept FROM episode_concepts WHERE episode_id=?",(row["id"],))}
  rs={(r["subject"],r["predicate"],r["object"]) for r in c.execute("SELECT subject,predicate,object FROM episode_relations WHERE episode_id=?",(row["id"],))}
  direct=len(qc&cs)/len(qc|cs) if qc or cs else 0
  lex=cosine(qvec,dv); graph=sum(act.get(x,0) for x in cs)/max(1,len(cs))
  rel=len(qr&rs)/len(qr|rs) if qr or rs else 0
  score=.43*lex+.32*direct+.17*graph+.08*rel
  results.append(dict(id=row["id"],title=row["title"],content=row["content"],score=score,lexical=lex,direct=direct,graph=graph,relation=rel,matched=sorted(qc&cs),activated=sorted((x,round(act[x],3)) for x in cs if x in act and x not in qc)))
 results.sort(key=lambda x:(-x["score"],-x["id"]))
 chosen=[x for x in results if x["score"]>=threshold][:limit]
 return chosen,{"query_concepts":sorted(qc),"activation":dict(sorted(((k,round(v,3)) for k,v in act.items()),key=lambda z:-z[1])[:20]),"threshold":threshold,"message":"Matches passed threshold." if chosen else "No memory passed threshold; abstaining."}
CASES=[
("A model brings back an old discussion when an idea feels familiar.","DVMN déjà vu memory idea"),
("How can spreading activation trigger old episodes?","Concept graph"),
("Use text vectors to find differently worded memories.","Embedding-based memory search"),
("Test retrieval on paraphrases and false matches.","Semantic retrieval benchmark"),
("How does fly-inspired circuitry process information?","Fruit-fly neural architecture"),
("A robot uses camera images to move around.","Camera navigation system"),
("Keep chat history on-device for confidentiality.","Local memory privacy"),
("Use verifiable rewards to train a model.","RLVR optimization"),
("Stars and planets in the Milky Way.","Astronomy observations"),
("Flour and yeast rise before baking.","Bread baking"),
("How does carbon affect steel hardness?","Materials engineering"),
("What should I cook for dinner?",None),("How do I repair a bicycle tire?",None),
("What is the capital of France?",None),("Tell me a joke about penguins.",None),
("How do I grow tomatoes?",None),("Explain ocean tides.",None),("How do I replace a faucet?",None),
("What are the rules of chess?",None),("Plan a beach weekend.",None)]
def cmd_init(a):
 c=dbopen(); print(f"DVMN {VERSION} initialized at {DB}\nExisting episodes preserved: {c.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]}"); c.close()
def cmd_seed(a):
 c=dbopen(); seed(c); c.close()
def cmd_add(a):
 c=dbopen(); eid=add(c,a.title or a.content[:60],a.content); print(f"Stored episode #{eid}"); c.close()
def cmd_list(a):
 c=dbopen()
 for r in c.execute("SELECT id,title,source,created_at FROM episodes ORDER BY id DESC LIMIT ?",(a.limit,)): print(f"#{r['id']} [{r['source']}] {r['title']} — {r['created_at'] or ''}")
 c.close()
def cmd_recall(a):
 c=dbopen(); ms,info=retrieve(c,a.query,a.limit,a.threshold)
 print(f"DVMN {VERSION} | {a.query}\nConcepts: {', '.join(info['query_concepts']) or '(none)'}\n{info['message']} Threshold={a.threshold:.2f}\n")
 for i,m in enumerate(ms,1):
  print(f"{i}. #{m['id']} {m['title']} [score={m['score']:.3f}]")
  print(f"   score: lexical={m['lexical']:.3f}, concept={m['direct']:.3f}, graph={m['graph']:.3f}, relation={m['relation']:.3f}")
  print("   matched: "+(", ".join(m["matched"]) or "(none)"))
  if m["activated"]: print("   activated: "+", ".join(f"{x}({v:.2f})" for x,v in m["activated"]))
  print("   "+m["content"])
 if not ms: print("No strong match. Add relevant memories or test a lower --threshold.")
 c.close()
def cmd_evaluate(a):
 c=dbopen(); seed(c); tp=fp=fn=tn=correct=0
 print(f"DVMN {VERSION}: {len(CASES)} fixed hand-written cases (smoke test, not general accuracy)\n")
 for q,expected in CASES:
  ms,_=retrieve(c,q,1,a.threshold); pred=ms[0]["title"] if ms else None; ok=pred==expected; correct+=ok
  if expected and pred==expected: tp+=1
  elif expected and pred is None: fn+=1
  elif expected is None and pred is None: tn+=1
  elif expected is None and pred is not None: fp+=1
  print(f"{'PASS' if ok else 'FAIL'} expected={expected or 'NONE'} | predicted={pred or 'NONE'} | {q}")
 total=len(CASES); print(f"\nTop-1 accuracy: {correct/total:.1%} ({correct}/{total})")
 print(f"Positive precision: {tp/(tp+fp):.1%}" if tp+fp else "Positive precision: n/a")
 print(f"Positive recall: {tp/(tp+fn):.1%}" if tp+fn else "Positive recall: n/a")
 print(f"False-positive rate: {fp/(fp+tn):.1%}" if fp+tn else "False-positive rate: n/a")
 c.close()
def cmd_chat(a):
 c=dbopen(); print("DVMN console. /add text, /list, /exit")
 while True:
  try:q=input("\n 👀").strip()
  except (EOFError,KeyboardInterrupt):break
  if q.lower() in ("/exit","exit","quit"):break
  if q=="/list": 
   for r in c.execute("SELECT id,title FROM episodes ORDER BY id DESC LIMIT 30"):print(f"#{r['id']} {r['title']}")
   continue
  if q.startswith("/add "): print("Stored episode #",add(c,q[5:][:60],q[5:])); continue
  ms,inf=retrieve(c,q,3,a.threshold); print(inf["message"])
  for m in ms: print(f"- #{m['id']} {m['title']} ({m['score']:.2f})\n  {m['content']}")
 c.close()
def parser():
 p=argparse.ArgumentParser(description=f"DVMN {VERSION} associative episodic-memory prototype"); s=p.add_subparsers(dest="cmd",required=True)
 for name,fn in [("init",cmd_init),("seed-demo",cmd_seed)]:
  x=s.add_parser(name); x.set_defaults(fn=fn)
 x=s.add_parser("add"); x.add_argument("content"); x.add_argument("--title"); x.set_defaults(fn=cmd_add)
 x=s.add_parser("list"); x.add_argument("--limit",type=int,default=30); x.set_defaults(fn=cmd_list)
 x=s.add_parser("recall"); x.add_argument("query"); x.add_argument("--limit",type=int,default=5); x.add_argument("--threshold",type=float,default=.18); x.set_defaults(fn=cmd_recall)
 for name,fn in [("evaluate",cmd_evaluate),("chat",cmd_chat)]:
  x=s.add_parser(name); x.add_argument("--threshold",type=float,default=.18); x.set_defaults(fn=fn)
 return p
if __name__=="__main__":
 args=parser().parse_args()
 try: args.fn(args)
 except (sqlite3.Error,OSError,ValueError) as e: print(f"Error: {e}",file=sys.stderr); sys.exit(1)
