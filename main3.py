#!/usr/bin/env python3
"""DVMN 0.4: trainable episodic memory retrieval prototype.
Python 3.10+. Core CLI uses scikit-learn; neural training uses PyTorch.
"""
import argparse, json, math, os, re, sqlite3, sys, urllib.request
from pathlib import Path
from collections import Counter
import json
from datetime import datetime,timezone

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'dvmn_memory.db'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'relevance_net.pt'
VECT_PATH = MODEL_DIR / 'tfidf.joblib'
WORDS = re.compile(r"[a-zA-Z0-9_'-]+")

def tokens(s): return [x.lower() for x in WORDS.findall(s or '')]
def connect():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('''CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP,source TEXT DEFAULT 'user')''')
    c.commit(); return c

def add_episode(title, content, source="user"):
    c = connect()

    try:
        cur = c.execute(
            """
            INSERT INTO episodes
            (title, content, concepts_json, relations_json, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                content,
                json.dumps([]),
                json.dumps([]),
                datetime.now(timezone.utc).isoformat(),
                source,
            ),
        )

        eid = cur.lastrowid
        c.commit()
        return eid

    finally:
        c.close()
    
def import_jsonl(path):
    n=0
    with open(path,encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: x=json.loads(line)
            except Exception as e: raise ValueError(f'Invalid JSON on line {i}: {e}')
            title=str(x.get('title') or x.get('question') or f'Episode {i}').strip()
            content=str(x.get('content') or x.get('text') or x.get('answer') or '').strip()
            if content: add_episode(title,content,str(x.get('source','dataset'))); n+=1
    print(f'Imported {n} episodes from {path}')

def get_episodes():
    c=connect(); rows=c.execute('SELECT id,title,content FROM episodes ORDER BY id').fetchall(); c.close(); return [dict(r) for r in rows]

def features(query, doc, vectorizer=None):
    qset=set(tokens(query)); dset=set(tokens(doc)); inter=len(qset&dset)
    overlap=inter/max(1,len(qset)); jacc=inter/max(1,len(qset|dset))
    qbig=set(zip(tokens(query),tokens(query)[1:])); dbig=set(zip(tokens(doc),tokens(doc)[1:]))
    big=len(qbig&dbig)/max(1,len(qbig))
    cosine=0.0
    if vectorizer:
        try:
            X=vectorizer.transform([query,doc]); a=X[0]; b=X[1]
            cosine=float(a.multiply(b).sum()/(math.sqrt(a.multiply(a).sum())*math.sqrt(b.multiply(b).sum())+1e-9))
        except Exception: pass
    return [overlap,jacc,big,cosine,min(len(qset)/40,1),min(len(dset)/500,1)]

def train(args):
    try:
        import numpy as np
        import torch
        from torch import nn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
    except ImportError:
        print('Missing dependencies. Run: python -m pip install -r requirements.txt'); sys.exit(2)
    episodes=get_episodes()
    if len(episodes)<4: print('Add/import at least 4 episodes first.'); sys.exit(2)
    # TF-IDF is fit on the full memory corpus. Pair labels are weakly supervised:
    # query terms sampled from an episode are positive; other episodes are negatives.
    corpus=[e['title']+' '+e['content'] for e in episodes]
    vectorizer=TfidfVectorizer(max_features=args.max_features,ngram_range=(1,2),sublinear_tf=True)
    vectorizer.fit(corpus)
    import joblib; joblib.dump(vectorizer,VECT_PATH)
    rng=np.random.default_rng(args.seed); X=[]; y=[]
    for i,e in enumerate(episodes):
        words=list(dict.fromkeys(tokens(e['title']+' '+e['content'])))
        if len(words)<3: continue
        for _ in range(args.pairs_per_episode):
            k=min(len(words),int(rng.integers(3,min(13,len(words))+1)))
            chosen=rng.choice(words,size=k,replace=False).tolist()
            q=' '.join(chosen)
            X.append(features(q,e['title']+' '+e['content'],vectorizer)); y.append(1.0)
            if len(episodes)>1:
                j=int(rng.integers(0,len(episodes)-1)); j += (j>=i)
                neg=episodes[j]
                X.append(features(q,neg['title']+' '+neg['content'],vectorizer)); y.append(0.0)
    if len(X)<8: print('Not enough usable text; import a larger dataset.'); sys.exit(2)
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
    ix=np.arange(len(y)); rng.shuffle(ix); X=X[ix]; y=y[ix]
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=args.seed,stratify=y)
    torch.manual_seed(args.seed)
    model=nn.Sequential(nn.Linear(X.shape[1],32),nn.ReLU(),nn.Dropout(.12),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); lossfn=nn.BCEWithLogitsLoss()
    xt=torch.tensor(Xtr); yt=torch.tensor(ytr[:,None]); xv=torch.tensor(Xv); yvt=torch.tensor(yv[:,None])
    best=1e9; best_state=None
    for epoch in range(args.epochs):
        model.train(); order=torch.randperm(len(xt)); total=0
        for ids in order.split(args.batch_size):
            opt.zero_grad(); loss=lossfn(model(xt[ids]),yt[ids]); loss.backward(); opt.step(); total+=loss.item()*len(ids)
        model.eval()
        with torch.no_grad(): vl=lossfn(model(xv),yvt).item(); probs=torch.sigmoid(model(xv)).numpy().ravel(); pred=probs>=.5
        acc=float((pred==yv).mean()); print(f'Epoch {epoch+1:02d}/{args.epochs} train_loss={total/len(xt):.4f} val_loss={vl:.4f} val_acc={acc:.3f}')
        if vl<best: best=vl; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({'state_dict':model.state_dict(),'input_dim':X.shape[1],'version':'0.4'},MODEL_PATH)
    print(f'Trained relevance network saved: {MODEL_PATH}')
    print(f'Weakly-labelled training pairs: {len(X)}. This is a research baseline, not proof of semantic understanding.')

def load_ranker():
    try:
        import torch
        from torch import nn
        import joblib
        if not MODEL_PATH.exists() or not VECT_PATH.exists(): return None,None,None
        vec=joblib.load(VECT_PATH); ck=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
        model=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Dropout(.12),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1)); model.load_state_dict(ck['state_dict']); model.eval()
        return model,vec,torch
    except Exception as e:
        print('Neural model not loaded:',e); return None,None,None

def recall(query, threshold, limit):
    episodes=get_episodes()
    if not episodes: print('No episodes yet. Use add or import.'); return
    model,vec,torch=load_ranker(); scored=[]
    for e in episodes:
        doc=e['title']+' '+e['content']; f=features(query,doc,vec)
        lex=f[3]*.65+f[0]*.2+f[1]*.15
        neural=None
        if model:
            with torch.no_grad(): neural=float(torch.sigmoid(model(torch.tensor([f],dtype=torch.float32))).item())
        score=(.65*neural+.35*lex) if neural is not None else lex
        scored.append((score,e,lex,neural))
    scored.sort(key=lambda z:z[0],reverse=True)
    hits=[x for x in scored if x[0]>=threshold][:limit]
    if not hits: print(f'No memory passed threshold {threshold:.2f}. Best score: {scored[0][0]:.3f}'); return
    for s,e,l,n in hits:
        print(f"\n[{s:.3f}] episode #{e['id']} — {e['title']}")
        print(e['content'][:900]+('…' if len(e['content'])>900 else ''))
        print(f"  lexical={l:.3f}"+(f' neural={n:.3f}' if n is not None else ' neural=not-trained'))

def llama(query, limit):
    # Optional local Ollama integration. It does not train Llama; it asks a locally running model to reason over retrieved memories.
    model=os.environ.get('DVMN_OLLAMA_MODEL','llama3.2')
    try:
        prompt='You are the reasoning component of DVMN. Use only the memories below as evidence, distinguish facts from hypotheses, and state when evidence is missing.\n\nUSER QUERY: '+query+'\n\nRETRIEVED MEMORIES:\n'
        import io
        old=sys.stdout; buf=io.StringIO(); sys.stdout=buf
        recall(query,0.0,limit); memories=buf.getvalue(); sys.stdout=old
        prompt+=memories+'\n\nAnswer the query using these memories.'
        data=json.dumps({'model':model,'prompt':prompt,'stream':False}).encode()
        req=urllib.request.Request('http://localhost:11434/api/generate',data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as r: result=json.loads(r.read().decode())
        print(result.get('response','(No response)'))
    except Exception as e: print('Could not contact Ollama. Install/start Ollama and pull a model, e.g. `ollama run llama3.2`. Details:',e)

def main():
    p=argparse.ArgumentParser(description='DVMN 0.4 trainable memory research prototype'); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('init'); a=sub.add_parser('add'); a.add_argument('content'); a.add_argument('--title',default='Untitled episode')
    im=sub.add_parser('import-jsonl'); im.add_argument('path')
    ls=sub.add_parser('list'); ls.add_argument('--limit',type=int,default=20)
    t=sub.add_parser('train'); t.add_argument('--epochs',type=int,default=12); t.add_argument('--pairs-per-episode',type=int,default=20); t.add_argument('--max-features',type=int,default=30000); t.add_argument('--batch-size',type=int,default=128); t.add_argument('--lr',type=float,default=.001); t.add_argument('--seed',type=int,default=42)
    r=sub.add_parser('recall'); r.add_argument('query'); r.add_argument('--threshold',type=float,default=.18); r.add_argument('--limit',type=int,default=5)
    ch=sub.add_parser('ask'); ch.add_argument('query'); ch.add_argument('--limit',type=int,default=5)
    args=p.parse_args()
    if args.cmd=='init': connect().close(); print(f'DVMN 0.4 initialized: {DB}')
    elif args.cmd=='add': print('Added episode',add_episode(args.title,args.content))
    elif args.cmd=='import-jsonl': import_jsonl(args.path)
    elif args.cmd=='list':
        for e in get_episodes()[-args.limit:]: print(f"#{e['id']} — {e['title']}\n{e['content'][:220]}\n")
    elif args.cmd=='train': train(args)
    elif args.cmd=='recall': recall(args.query,args.threshold,args.limit)
    elif args.cmd=='ask': llama(args.query,args.limit)
if __name__=='__main__': main()
