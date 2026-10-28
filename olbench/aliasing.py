from __future__ import annotations
import re, itertools, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import canonical

def regex_alias(terms, **_):
    u=terms.sort_values('score',ascending=False).drop_duplicates('term_canonical')
    vals=u[['term_canonical','raw_term']].to_dict('records'); pairs=[]
    # Acronym pairs and normalized surface variants.
    for a,b in itertools.combinations(vals,2):
        ta,tb=a['raw_term'],b['raw_term']; ca,cb=a['term_canonical'],b['term_canonical']
        aa=''.join(w[0] for w in re.findall(r'[A-Za-z]+',ta) if w.lower() not in {'of','the','and'})
        ab=''.join(w[0] for w in re.findall(r'[A-Za-z]+',tb) if w.lower() not in {'of','the','and'})
        same=ca==cb or (len(tb)<=8 and aa.lower()==tb.lower()) or (len(ta)<=8 and ab.lower()==ta.lower())
        singular=ca.rstrip('s')==cb.rstrip('s') and min(len(ca),len(cb))>3
        if same or singular: pairs.append({'term_a':ta,'term_b':tb,'canonical_a':ca,'canonical_b':cb,'method':'regex','score':1.0,'is_alias_candidate':True})
    return pd.DataFrame(pairs,columns=['term_a','term_b','canonical_a','canonical_b','method','score','is_alias_candidate'])

def _representatives(terms,representation):
    u=terms.sort_values('score',ascending=False).drop_duplicates('term_canonical').copy()
    if representation=='term': u['representation']=u.raw_term
    else: u['representation']=u.raw_term+' [CONTEXT] '+u.context_text.fillna('')
    return u

def cosine_alias(terms,representation='term',model='all-mpnet-base-v2',threshold=0.7,batch_size=32,**_):
    u=_representatives(terms,representation); texts=u.representation.tolist(); backend='sentence_transformers'
    try:
        from sentence_transformers import SentenceTransformer
        E=SentenceTransformer(model).encode(texts,normalize_embeddings=True,batch_size=batch_size,show_progress_bar=False); S=E@E.T
    except Exception:
        backend='tfidf_fallback'; X=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1).fit_transform(texts); S=cosine_similarity(X)
    accepted=[]; rejected=[]
    for i,j in itertools.combinations(range(len(u)),2):
        r={'term_a':u.iloc[i].raw_term,'term_b':u.iloc[j].raw_term,'canonical_a':u.iloc[i].term_canonical,'canonical_b':u.iloc[j].term_canonical,'method':'cosine','representation':representation,'model':model,'backend':backend,'score':float(S[i,j]),'threshold':threshold,'is_alias_candidate':bool(S[i,j]>threshold)}
        (accepted if r['is_alias_candidate'] else rejected).append(r)
    return pd.DataFrame(accepted),pd.DataFrame(rejected)

def canonical_concepts(terms,pairs):
    names=terms.term_canonical.unique().tolist(); parent={x:x for x in names}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    if pairs is not None and not pairs.empty:
        for r in pairs.itertuples(): union(r.canonical_a,r.canonical_b)
    rows=[]
    best=terms.sort_values('score',ascending=False).drop_duplicates('term_canonical').set_index('term_canonical')
    groups={}
    for x in names: groups.setdefault(find(x),[]).append(x)
    for gid,members in groups.items():
        label=max(members,key=lambda x:float(best.loc[x,'score']))
        for m in members: rows.append({'concept_id':'concept_'+str(abs(hash(gid)))[:10],'canonical_term':label,'member_term':m,'aliases':' | '.join(sorted(set(members)-{label}))})
    return pd.DataFrame(rows)
