from __future__ import annotations
from pathlib import Path
import pandas as pd

def _prf(pred,gold):
    pred=set(pred); gold=set(gold); tp=len(pred&gold); p=tp/len(pred) if pred else 0; r=tp/len(gold) if gold else 0; f=2*p*r/(p+r) if p+r else 0
    return p,r,f,tp,len(pred),len(gold)

def evaluate(terms=None,alias_pairs=None,typed=None,gt_dir=None):
    rows=[]; gt=Path(gt_dir) if gt_dir else None
    if gt and (gt/'GT1_domain_terms.csv').exists() and terms is not None:
        g=pd.read_csv(gt/'GT1_domain_terms.csv'); vals=_prf(terms.term_canonical,g.term.astype(str).str.lower().str.strip()); rows.append(dict(metric='term_extraction_f1',value=vals[2],precision=vals[0],recall=vals[1]))
    if gt and (gt/'GT2_alias_acronym.csv').exists() and alias_pairs is not None:
        g=pd.read_csv(gt/'GT2_alias_acronym.csv'); gold={tuple(sorted((str(a).lower(),str(b).lower()))) for a,b in zip(g.term_a,g.term_b)}; pred={tuple(sorted((str(a).lower(),str(b).lower()))) for a,b in zip(alias_pairs.term_a,alias_pairs.term_b)}; vals=_prf(pred,gold); rows.append(dict(metric='alias_f1',value=vals[2],precision=vals[0],recall=vals[1]))
    if gt and (gt/'GT4_taxonomy.csv').exists() and typed is not None:
        g=pd.read_csv(gt/'GT4_taxonomy.csv'); m=typed.merge(g,left_on='term_canonical',right_on=g.term.astype(str).str.lower().str.strip(),how='inner'); acc=(m.entity_type_x==m.entity_type_y).mean() if len(m) else 0; rows.append(dict(metric='typing_accuracy',value=acc,precision=None,recall=None))
    if not rows: rows=[{'metric':'candidate_terms','value':0 if terms is None else terms.term_canonical.nunique(),'precision':None,'recall':None},{'metric':'alias_candidates','value':0 if alias_pairs is None else len(alias_pairs),'precision':None,'recall':None},{'metric':'typed_entities','value':0 if typed is None else len(typed),'precision':None,'recall':None}]
    return pd.DataFrame(rows)
