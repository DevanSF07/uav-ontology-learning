from __future__ import annotations
import json, re, pandas as pd
from .utils import ollama_json
TYPES=['System','Subsystem','Part','Component','Interface','Flow','Parameter','Classification Class','Mission','Capability','Activity','Environment','Unknown']
STRUCTURAL={'System','Subsystem','Part','Component','Interface','Flow','Classification Class'}

def fallback_type(term,context=''):
    t=(' '+term+' '+context[:300]+' ').lower()
    rules=[('Subsystem',r'\bsubsystem\b'),('Interface',r'\b(interface|connector|port|link)\b'),('Flow',r'\b(flow|signal|data stream|fuel|energy)\b'),('Parameter',r'\b(parameter|rate|speed|altitude|temperature|pressure|mass)\b'),('Mission',r'\bmission\b'),('Capability',r'\bcapability|ability\b'),('Activity',r'\b(activity|operation|process|planning|control)\b'),('Environment',r'\b(environment|weather|terrain|airspace)\b'),('Classification Class',r'\b(class|category|type|classification)\b'),('Component',r'\b(component|module|sensor|controller|processor|actuator|motor|battery|antenna|camera)\b'),('Part',r'\b(part|wing|blade|shaft|housing)\b'),('System',r'\b(system|vehicle|aircraft|platform|network)\b')]
    for typ,pat in rules:
        if re.search(pat,t): return typ,0.65,'fallback_rules'
    return 'Unknown',0.2,'fallback_rules'

def type_entities(concepts,group_labels=None,model='llama3.1:8b',allow_fallback=True,**_):
    id_col='cluster_id' if 'cluster_id' in concepts else 'factor_id' if 'factor_id' in concepts else None
    label_map={}
    if id_col and group_labels is not None and not group_labels.empty: label_map=group_labels.set_index(id_col).group_label.to_dict()
    rows=[]
    for r in concepts.itertuples():
        term=r.raw_term; context=getattr(r,'context_text',''); gid=getattr(r,id_col) if id_col else None; glabel=label_map.get(gid,'')
        try:
            prompt=f'Classify one technical ontology term. Allowed labels: {TYPES}. Return JSON only with entity_type, confidence, reason. Term: {term}\nContext: {context}\nGroup label: {glabel}'
            d=ollama_json(prompt,model); typ=d.get('entity_type','Unknown'); typ=typ if typ in TYPES else 'Unknown'; conf=float(d.get('confidence',0.5)); reason=d.get('reason','ollama')
        except Exception:
            if not allow_fallback: raise
            typ,conf,reason=fallback_type(term,context)
        rows.append({'term':term,'term_canonical':r.term_canonical,'entity_type':typ,'confidence':conf,'reason':reason,'group_id':gid,'group_label':glabel,'context_text':context})
    return pd.DataFrame(rows)

def structural_filter(typed):
    mask=typed.entity_type.isin(STRUCTURAL)
    return typed[mask].reset_index(drop=True),typed[~mask].reset_index(drop=True)
