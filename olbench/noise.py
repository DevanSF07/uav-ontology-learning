from __future__ import annotations
import re, pandas as pd
NEGATIVE=['person','organization','country','location','city','state','date','publication','book','university']
COUNTRIES={'united states','india','china','france','germany','canada','japan','australia','brazil','russia','uk','usa'}

def filter_noise(terms, model='urchade/gliner_multi-v2.1', threshold=0.5, allow_fallback=True, **_):
    drops={}; unique=terms[['term_canonical','raw_term']].drop_duplicates()
    try:
        from gliner import GLiNER
        gl=GLiNER.from_pretrained(model)
        raw_terms=unique['raw_term'].tolist()
        term_canonicals=unique['term_canonical'].tolist()
        all_ents=[]
        batch_size=256
        for i in range(0, len(raw_terms), batch_size):
            b_terms=raw_terms[i:i+batch_size]
            if hasattr(gl, 'predict_entities_batch'):
                all_ents.extend(gl.predict_entities_batch(b_terms,NEGATIVE,threshold=threshold))
            else:
                all_ents.extend([gl.predict_entities(t,NEGATIVE,threshold=threshold) for t in b_terms])
        for term_canonical, ents in zip(term_canonicals, all_ents):
            if ents: drops[term_canonical]=max(ents,key=lambda x:x.get('score',0))
    except Exception:
        if not allow_fallback: raise
        org=re.compile(r'\b(university|corporation|corp|inc|ltd|agency|department|commission|authority|ministry|office)\b',re.I)
        person=re.compile(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$')
        for r in unique.itertuples():
            if r.term_canonical in COUNTRIES: drops[r.term_canonical]={'label':'country','score':1.0,'fallback':True}
            elif org.search(r.raw_term): drops[r.term_canonical]={'label':'organization','score':0.8,'fallback':True}
            elif person.match(r.raw_term) and not re.search(r'\b(system|vehicle|aircraft|sensor|control)\b',r.raw_term,re.I): drops[r.term_canonical]={'label':'person','score':0.6,'fallback':True}
    mask=terms.term_canonical.isin(drops)
    dropped=terms[mask].copy(); kept=terms[~mask].copy()
    dropped['noise_label']=dropped.term_canonical.map(lambda x:drops[x].get('label','unknown'))
    dropped['noise_score']=dropped.term_canonical.map(lambda x:drops[x].get('score',0))
    return kept.reset_index(drop=True), dropped.reset_index(drop=True)
