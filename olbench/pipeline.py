from __future__ import annotations
import hashlib, json, shutil, time
from pathlib import Path
import pandas as pd
from .utils import load_json,dump_json,save_df,file_hash,dependency_status
from .preprocess import clean_mmd,chunk_text
from .extractors import EXTRACTORS
from .noise import filter_noise
from .aliasing import regex_alias,cosine_alias,canonical_concepts
from .concepts import embedding_cluster,embedding_nmf
from .typing import type_entities,structural_filter
from .evaluate import evaluate

def _cache_key(input_hash,experiment): return hashlib.sha256(json.dumps({'input':input_hash,'experiment':experiment},sort_keys=True).encode()).hexdigest()
def _write(df,out,name): save_df(df,out/name); return name

def run(config_path,input_path,out_root,gt_dir=None):
    cfg=load_json(config_path); out_root=Path(out_root); out_root.mkdir(parents=True,exist_ok=True); ih=file_hash(input_path); run_manifest={'input':str(input_path),'input_sha256':ih,'config':str(config_path),'dependencies':dependency_status(),'experiments':[]}
    cleaned,report=clean_mmd(input_path); chunks=chunk_text(cleaned,**cfg.get('chunking',{}))
    for exp in cfg['experiments']:
        eid=exp['id']; out=out_root/eid; out.mkdir(parents=True,exist_ok=True); key=_cache_key(ih,exp); manifest_path=out/'manifest.json'
        if manifest_path.exists() and load_json(manifest_path).get('cache_key')==key:
            run_manifest['experiments'].append({'id':eid,'status':'reused','path':str(out)}); continue
        start=time.time(); (out/'cleaned_corpus.txt').write_text(cleaned,encoding='utf-8'); _write(report,out,'preprocess_report.csv'); _write(chunks,out,'chunks.csv')
        ex=exp['extraction']; terms=EXTRACTORS[ex['method']](chunks,**ex.get('params',{})); _write(terms,out,'terms_extracted.csv')
        noise_cfg=exp.get('noise',{'enabled':True}); kept,dropped=filter_noise(terms,**noise_cfg.get('params',{})) if noise_cfg.get('enabled',True) else (terms,terms.iloc[0:0]); _write(kept,out,'terms_noise_filtered.csv'); _write(dropped,out,'terms_noise_dropped.csv')
        alias_cfg=exp.get('alias',{'method':'regex'}); rejected=pd.DataFrame()
        if alias_cfg['method']=='regex': pairs=regex_alias(kept,**alias_cfg.get('params',{}))
        else: pairs,rejected=cosine_alias(kept,**alias_cfg.get('params',{}))
        _write(pairs,out,'alias_pairs.csv'); _write(rejected,out,'non_alias_pairs.csv'); canon=canonical_concepts(kept,pairs); _write(canon,out,'canonical_concepts.csv')
        concept_cfg=exp.get('concept'); typed=None
        if concept_cfg:
            if concept_cfg['method']=='cluster': concepts,labels=embedding_cluster(kept,canon,**concept_cfg.get('params',{})); idname='cluster'
            else: concepts,labels=embedding_nmf(kept,canon,**concept_cfg.get('params',{})); idname='factor'
            _write(concepts,out,f'concepts_{idname}.csv'); _write(labels,out,f'{idname}_labels.csv')
            if exp.get('typing',{}).get('enabled',False):
                typed=type_entities(concepts,labels,**exp['typing'].get('params',{})); _write(typed,out,'typed_entities.csv')
                structural,nonstruct=structural_filter(typed); _write(structural,out,'typed_entities_structural_only.csv'); _write(nonstruct,out,'typed_entities_nonstructural_dropped.csv')
        metrics=evaluate(kept,pairs,typed,gt_dir); metrics.insert(0,'experiment_id',eid); _write(metrics,out,'metrics.csv')
        manifest={'experiment_id':eid,'cache_key':key,'input_sha256':ih,'config':exp,'duration_seconds':round(time.time()-start,3),'artifacts':sorted(p.name for p in out.iterdir())}; dump_json(manifest_path,manifest); run_manifest['experiments'].append({'id':eid,'status':'completed','path':str(out)})
    dump_json(out_root/'experiment_manifest.json',run_manifest); return run_manifest

def collect(out_root):
    out_root=Path(out_root); frames=[]
    for p in out_root.glob('*/metrics.csv'):
        try: frames.append(pd.read_csv(p))
        except Exception: pass
    result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(); save_df(result,out_root/'metrics_summary.csv'); return result
