#!/usr/bin/env python3
"""Complete single-file corpus-grounded agentic taxonomy pipeline.

Pipeline
========
1. Load *all* entity-typed terms and select predicted System/Subsystem terms.
2. Corpus Presence & Noise Filter:
   exact -> morphology -> aliases -> controlled reconstruction -> Qwen review.
3. Canonicalize accepted terms with union-find and explicit aliases.
4. Build same-type candidate pools directly from the canonical concept inventory.
5. Resolve configured roots.
6. Recursively induce parent-child relations with three agents:
   Child-Proposal -> Evidence Validator -> Relation Critic.
7. Discover candidate facets from bounded groups of unplaced concepts.
8. Run a Global Graph Auditor.
9. Enforce DAG checks, remove duplicates/self-loops/cycles, and perform
   per-view transitive reduction.
10. Export CSV, JSON, GraphML, statistics, prompts, caches, and manifests.

No reference taxonomy is consumed. LLM agents inspect the canonical same-type inventory; corpus evidence grounds decisions; graph algorithms enforce structure.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, re, sys, time, unicodedata
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DEFAULT_TYPING = PROJECT_DIR / "stage_outputs/10_entity_typing_all_qwen_terms/08_all_qwen_terms_with_entity_types.csv"
DEFAULT_CONTEXTS = PROJECT_DIR / "stage_outputs/10_entity_typing_all_qwen_terms/03_retrieved_contexts.csv"
DEFAULT_CORPUS = PROJECT_DIR / "data/combined_corpus_raw.txt"
DEFAULT_ALIASES = PROJECT_DIR / "data/Ground_Truth.xlsx"
DEFAULT_OUTPUT = PROJECT_DIR / "stage_outputs/14_complete_agentic_taxonomy_single"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_URL = "http://localhost:11434"
PROMPT_VERSION = "single-complete-taxonomy-no-embeddings-v1"

ACCEPT_PRESENCE = {"exact_present", "morphological_present", "alias_present", "corpus_reconstructed"}
REJECT_PRESENCE = {"not_present", "fragment", "artifact", "context_only"}
FILTER_DECISIONS = {"accept_variant", "accept_alias", "accept_reconstructed", "reject_not_present", "reject_fragment", "reject_artifact", "reject_context_only", "manual_review"}
PROPOSAL_DECISIONS = {"direct_child", "indirect_descendant", "equivalent_to_parent", "related_not_hierarchical", "wrong_semantic_level", "insufficient_evidence"}
VALIDATION_DECISIONS = {"accept_direct_edge", "accept_as_indirect_descendant", "reverse_direction", "equivalent", "related_only", "insufficient_evidence", "reject"}
CRITIC_DECISIONS = {"accept", "reject", "reverse", "convert_to_indirect", "mark_equivalent", "change_classification_view", "manual_review"}


def now_utc(): return datetime.now(timezone.utc).isoformat()
def safe(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return ""
    return str(v).strip()
def norm(v):
    x = unicodedata.normalize("NFKC", safe(v)).casefold().replace("–", "-").replace("—", "-").replace("−", "-")
    x = re.sub(r"[‐‑‒]", "-", x); x = re.sub(r"[-_/]+", " ", x)
    x = re.sub(r"[^\w\s]", " ", x); return re.sub(r"\s+", " ", x).strip()
def toks(v): return norm(v).split()
def sing(t):
    d={"systems":"system","vehicles":"vehicle","stations":"station","links":"link","platforms":"platform","subsystems":"subsystem","aircraft":"aircraft","classes":"class","categories":"category","facilities":"facility","capabilities":"capability"}
    t=t.casefold()
    if t in d:return d[t]
    if len(t)>4 and t.endswith("ies"):return t[:-3]+"y"
    if len(t)>4 and t.endswith(("sses","xes")):return t[:-2]
    if len(t)>3 and t.endswith("s") and not t.endswith(("ss","us","is")):return t[:-1]
    return t
def morph(v):
    z=toks(v)
    if z:z[-1]=sing(z[-1])
    return " ".join(z)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def durable(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
def load_jsonl(path):
    if not path.is_file():return []
    out=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        try: out.append(json.loads(line))
        except Exception: pass
    return out
def col(df,names):
    m={str(c).casefold():c for c in df.columns}
    for n in names:
        if n.casefold() in m:return m[n.casefold()]
    return None
def clip(x): return max(0.0,min(1.0,float(x)))

class UF:
    def __init__(self): self.p={}
    def add(self,x): self.p.setdefault(x,x)
    def find(self,x):
        self.add(x)
        if self.p[x]!=x:self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a


def split_sentences(text):
    text=unicodedata.normalize("NFKC",text);text=re.sub(r"\r\n?","\n",text);text=re.sub(r"[ \t]+"," ",text)
    chunks=re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[]|[\"'])|\n{2,}",text)
    seen=set();out=[]
    for x in chunks:
        x=re.sub(r"\s+"," ",x).strip();k=norm(x)
        if len(x)<20 or len(x.split())<4 or not k or k in seen:continue
        seen.add(k);out.append(x)
    return out

@dataclass
class Corpus:
    sents:list[str]; ns:list[str]; inv:dict[str,set[int]]
    @classmethod
    def build(cls,text):
        s=split_sentences(text);ns=[norm(x) for x in s];inv=defaultdict(set)
        for i,x in enumerate(ns):
            for t in set(x.split()):inv[t].add(i)
        return cls(s,ns,dict(inv))
    def exact(self,phrase):
        k=norm(phrase);ts=k.split()
        if not ts:return []
        sets=[self.inv.get(t,set()) for t in set(ts)]
        if not sets or any(not q for q in sets):return []
        return [i for i in sorted(set.intersection(*sets)) if re.search(r"(?<!\w)"+re.escape(k)+r"(?!\w)",self.ns[i])]
    def mhits(self,phrase):
        k=morph(phrase);width=len(k.split());cand=set()
        for t in toks(phrase):cand|=self.inv.get(t,set())|self.inv.get(sing(t),set())
        out=[]
        for i in sorted(cand):
            z=self.ns[i].split()
            if any(morph(" ".join(z[j:j+width]))==k for j in range(max(0,len(z)-width+1))):out.append(i)
        return out
    def evidence(self,ids,limit=6):
        out=[];used=set()
        for h in ids:
            for i in (h,h-1,h+1):
                if 0<=i<len(self.sents) and i not in used:
                    used.add(i);out.append({"sentence_id":f"SENT_{i:06d}","sentence":self.sents[i],"direct":i==h})
                    if len(out)>=limit:return out
        return out
    def loose(self,phrase,limit=6):
        ids=set();tt=set(toks(phrase))
        for t in tt:ids|=self.inv.get(t,set())
        ids=sorted(ids,key=lambda i:-len(tt&set(self.ns[i].split())))[:limit]
        return self.evidence(ids,limit)

class Ollama:
    def __init__(self,url,model,out,temp,timeout,retries):
        self.url=url.rstrip("/");self.model=model;self.out=out;self.temp=temp;self.timeout=timeout;self.retries=retries
        self.cache=out/"cache"/"qwen";self.cache.mkdir(parents=True,exist_ok=True);self.log=out/"qwen_requests.jsonl";self.live=0;self.hits=0
    def check(self):
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url+"/api/tags"),timeout=10) as r:
                if r.status!=200:raise RuntimeError(r.status)
        except Exception as e:raise RuntimeError(f"Ollama unavailable at {self.url}; run 'ollama serve'. {e}")
    def key(self,stage,prompt):
        return hashlib.sha256(json.dumps({"v":PROMPT_VERSION,"stage":stage,"model":self.model,"temp":self.temp,"prompt":prompt},sort_keys=True).encode()).hexdigest()
    def parse(self,text):
        text=re.sub(r"^```(?:json)?\s*|\s*```$","",text.strip(),flags=re.I)
        try:v=json.loads(text)
        except Exception:
            a,b=text.find("{"),text.rfind("}");v=json.loads(text[a:b+1])
        if not isinstance(v,dict):raise ValueError("JSON object required")
        return v
    def ask(self,stage,prompt,validator):
        k=self.key(stage,prompt);p=self.cache/f"{k}.json"
        if p.is_file():
            try:
                row=json.loads(p.read_text());v=validator(row["result"]);self.hits+=1;return v
            except Exception:p.unlink(missing_ok=True)
        err=None
        for n in range(1,self.retries+1):
            st=time.perf_counter();payload={"model":self.model,"messages":[{"role":"system","content":"Return valid JSON only. Be conservative, evidence-grounded, and do not invent corpus facts."},{"role":"user","content":prompt}],"stream":False,"options":{"temperature":self.temp}}
            req=urllib.request.Request(self.url+"/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
            try:
                with urllib.request.urlopen(req,timeout=self.timeout) as r:env=json.loads(r.read().decode())
                raw=env["message"]["content"];v=validator(self.parse(raw));rec={"stage":stage,"attempt":n,"prompt":prompt,"raw":raw,"result":v,"runtime":time.perf_counter()-st}
                p.write_text(json.dumps(rec,ensure_ascii=False,indent=2));durable(self.log,rec);self.live+=1;return v
            except Exception as e:err=e;time.sleep(min(2*n,6))
        raise RuntimeError(f"Qwen stage {stage} failed: {err}")

# ---------------------------- INPUTS AND FILTER ----------------------------
def load_aliases(path,sheet):
    if not path.is_file():return pd.DataFrame(columns=["canonical","alias","cn","an"])
    book=pd.ExcelFile(path,engine="openpyxl")
    if sheet not in book.sheet_names:return pd.DataFrame(columns=["canonical","alias","cn","an"])
    d=pd.read_excel(path,sheet_name=sheet,engine="openpyxl");c=col(d,["canonical","term_canonical"]);a=col(d,["alias","synonym"])
    if not c or not a:return pd.DataFrame(columns=["canonical","alias","cn","an"])
    o=pd.DataFrame({"canonical":d[c].map(safe),"alias":d[a].map(safe)});o["cn"]=o.canonical.map(norm);o["an"]=o.alias.map(norm)
    return o[o.cn.ne("")&o.an.ne("")].drop_duplicates()
def alias_lookup(d):
    m=defaultdict(set)
    for r in d.itertuples(index=False):m[r.cn].add(r.alias);m[r.an].add(r.canonical)
    return dict(m)
def load_typed(path):
    d=pd.read_csv(path,low_memory=False);t=col(d,["canonical_term","candidate","term","preferred_label"]);y=col(d,["final_type","predicted_type","entity_type"])
    if not t or not y:raise ValueError(f"Missing term/type columns: {list(d.columns)}")
    d=d.copy();d["taxonomy_term"]=d[t].map(safe);d["entity_type"]=d[y].map(safe);d["term_norm"]=d.taxonomy_term.map(norm)
    return d[d.entity_type.isin(["System","Subsystem"])&d.term_norm.ne("")].drop_duplicates("term_norm").reset_index(drop=True)
def recon(term):
    out=[];m=re.match(r"^\s*([\w&+.-]+)\s*/\s*([\w&+.-]+)\s+(.+?)\s*$",term)
    if m:a,b,tail=m.groups();out += [f"{a} {tail}",f"{b} {tail}"]
    return out

def val_filter(v):
    dec=safe(v.get("decision"));allowed={"accept_variant","accept_alias","accept_reconstructed","reject_not_present","reject_fragment","reject_artifact","reject_context_only","manual_review"}
    if dec not in allowed:raise ValueError("bad filter decision")
    ids=v.get("evidence_sentence_ids",[])
    if not isinstance(ids,list):raise ValueError("bad ids")
    return {"decision":dec,"confidence":clip(v.get("confidence",0)),"matched_text":safe(v.get("matched_text")),"possible_complete_form":safe(v.get("possible_complete_form")),"evidence_sentence_ids":[safe(x) for x in ids],"reason":safe(v.get("reason"))}

def filter_prompt(term,etype,desc,evidence):
    ev="\n".join(f"[{x['sentence_id']}] {x['sentence']}" for x in evidence) or "No reliable lexical evidence."
    return f'''Corpus Presence & Noise Filter Agent\nTERM: {term}\nPREDICTED TYPE: {etype}\nDESCRIPTION: {desc or 'N/A'}\nEVIDENCE:\n{ev}\n\nDecide whether this is a complete reusable technical concept explicitly represented by the corpus. Semantic similarity alone is insufficient. Allowed decisions: accept_variant, accept_alias, accept_reconstructed, reject_not_present, reject_fragment, reject_artifact, reject_context_only, manual_review. Never invent text. JSON only: {{"decision":"...","confidence":0.0,"matched_text":"","possible_complete_form":"","evidence_sentence_ids":[],"reason":""}}'''

def run_filter(args,typed,corpus,aliases,client):
    out=args.output_dir;partial=out/"01_filter_results.partial.jsonl";done={norm(x.get("candidate")):x for x in load_jsonl(partial)};lookup=alias_lookup(aliases);records=[]
    for n,row in typed.iterrows():
        term=row.taxonomy_term;k=norm(term);print(f"FILTER {n+1}/{len(typed)}: {term}")
        if k in done:records.append(done[k]);continue
        status=None;method="";hits=[];matched="";confidence=0;reason="";evidence=[];agent_dec="";complete=""
        hits=corpus.exact(term)
        if hits:status,method,matched,confidence,reason="exact_present","normalized_exact",term,1.0,"Exact normalized occurrence."
        else:
            hits=corpus.mhits(term)
            if hits:status,method,confidence,reason="morphological_present","safe_morphology",.97,"Safe singular/plural occurrence."
        if status is None:
            best=[]
            for a in lookup.get(k,set()):
                h=corpus.exact(a) or corpus.mhits(a)
                if len(h)>len(best):best=h;matched=a
            if best:hits=best;status,method,confidence,reason="alias_present","explicit_alias",.98,"Explicit alias occurs."
        if status is None:
            best=[]
            for r in recon(term):
                h=corpus.exact(r) or corpus.mhits(r)
                if len(h)>len(best):best=h;matched=r
            if best:hits=best;status,method,confidence,reason="corpus_reconstructed","controlled_reconstruction",.88,"Controlled reconstructed form occurs."
        evidence=corpus.evidence(hits) if hits else corpus.loose(term)
        if status is None:
            desc=safe(row.get("entity_description",row.get("description","")))
            try:
                ans=client.ask("filter::"+hashlib.sha1(k.encode()).hexdigest()[:12],filter_prompt(term,row.entity_type,desc,evidence),val_filter);agent_dec=ans["decision"];confidence=ans["confidence"];matched=ans["matched_text"];complete=ans["possible_complete_form"];reason=ans["reason"]
                status={"accept_variant":"morphological_present","accept_alias":"alias_present","accept_reconstructed":"corpus_reconstructed","reject_not_present":"not_present","reject_fragment":"fragment","reject_artifact":"artifact","reject_context_only":"context_only","manual_review":"manual_review"}[agent_dec];method="qwen_filter_agent"
            except Exception as e:status,method,confidence,reason="manual_review","qwen_filter_failed",0,f"Safe fallback: {e}"
        action="accept" if status in ACCEPT_PRESENCE else "reject" if status in REJECT_PRESENCE else "manual_review"
        rec=row.to_dict();rec.update({"candidate":term,"presence_status":status,"taxonomy_action":action,"matched_surface_form":matched,"possible_complete_form":complete,"occurrence_count":len(hits),"evidence_sentence_ids":[x["sentence_id"] for x in evidence],"evidence_sentences":[x["sentence"] for x in evidence],"filter_method":method,"filter_confidence":confidence,"filter_reason":reason,"filter_agent_decision":agent_dec,"node_origin":"corpus" if action=="accept" else ""})
        durable(partial,rec);done[k]=rec;records.append(rec)
    d=pd.DataFrame(records);d.to_csv(out/"01_corpus_presence_results.csv",index=False);d[d.taxonomy_action.eq("accept")].to_csv(out/"02_taxonomy_terms_accepted.csv",index=False);d[d.taxonomy_action.eq("reject")].to_csv(out/"03_taxonomy_terms_rejected.csv",index=False);d[d.taxonomy_action.eq("manual_review")].to_csv(out/"04_taxonomy_terms_manual_review.csv",index=False)
    return d

# -------------------- CANONICALIZATION / DIRECT CANDIDATES --------------------
def canonicalize(d,aliases):
    uf=UF();rows={r.term_norm:r for _,r in d.iterrows()}
    for k in rows:uf.add(k)
    by_m=defaultdict(list)
    for k in rows:by_m[morph(k)].append(k)
    for group in by_m.values():
        for x in group[1:]:uf.union(group[0],x)
    for r in aliases.itertuples(index=False):
        if r.cn in rows and r.an in rows:uf.union(r.cn,r.an)
    groups=defaultdict(list)
    for k in rows:groups[uf.find(k)].append(k)
    out=[]
    for i,keys in enumerate(groups.values(),1):
        rr=[rows[k] for k in keys];labels=sorted({safe(x.taxonomy_term) for x in rr},key=lambda x:(len(x.split()),len(x)),reverse=True);pref=labels[0];types=Counter(safe(x.entity_type) for x in rr);etype=types.most_common(1)[0][0]
        desc=next((safe(x.get("entity_description","")) for x in rr if safe(x.get("entity_description",""))),"")
        ev=[]
        for x in rr:
            v=x.get("evidence_sentences",[])
            if isinstance(v,str):
                try:v=json.loads(v)
                except Exception:v=[v] if v else []
            if isinstance(v,list):ev += [safe(z) for z in v if safe(z)]
        out.append({"concept_id":f"CONCEPT_{i:04d}","preferred_label":pref,"aliases":sorted(set(labels)-{pref}),"entity_type":etype,"description":desc,"evidence_sentences":list(dict.fromkeys(ev))[:8],"source_terms":labels})
    return pd.DataFrame(out)

def concept_text(r):return f"Concept: {r.preferred_label}. Aliases: {', '.join(r.aliases) if isinstance(r.aliases,list) else r.aliases}. Type: {r.entity_type}. Description: {r.description}. Evidence: {' '.join(r.evidence_sentences if isinstance(r.evidence_sentences,list) else [])}"

class DirectCandidateIndex:
    """Candidate provider with no embeddings and no clustering.

    For each parent, every canonical concept of the same entity type is eligible.
    The list is ordered by lightweight lexical overlap only to make chunking
    deterministic; lexical overlap never proves or scores a taxonomy edge.
    The Child-Proposal Agent evaluates every same-type candidate across chunks.
    """
    def __init__(self, concepts: pd.DataFrame):
        self.concepts = concepts.reset_index(drop=True)
        self.ids = self.concepts.concept_id.tolist()

    @staticmethod
    def _lexical_overlap(parent: dict[str, Any], child: dict[str, Any]) -> tuple[int, int, str]:
        p = set(toks(parent.get("preferred_label", "")))
        c = set(toks(child.get("preferred_label", "")))
        overlap = len(p & c)
        # Stable deterministic ordering only; not a semantic relation score.
        return (-overlap, abs(len(c) - len(p)), safe(child.get("preferred_label", "")))

    def retrieve(self, parent_id, records, topk, etype=None):
        parent = records[parent_id]
        candidates = []
        for cid, concept in records.items():
            if cid == parent_id:
                continue
            if etype and concept.get("entity_type") != etype:
                continue
            candidates.append((self._lexical_overlap(parent, concept), cid))
        candidates.sort(key=lambda item: item[0])
        # topk=0 means evaluate every same-type candidate.
        selected = candidates if topk <= 0 else candidates[:topk]
        return [
            {
                "candidate_id": cid,
                "retrieval_score": None,
                "cosine_similarity": None,
                "candidate_source": "same_type_inventory",
            }
            for _, cid in selected
        ]

# ------------------------------- TAXONOMY AGENTS -------------------------------
def vproposal(v,cands):
    arr=v.get("candidates",[])
    if not isinstance(arr,list):raise ValueError("candidates list required")
    out=[];allowed={x["candidate_id"] for x in cands}
    for x in arr:
        cid=safe(x.get("candidate_id"));dec=safe(x.get("decision"))
        if cid not in allowed or dec not in PROPOSAL_DECISIONS:continue
        out.append({"candidate_id":cid,"decision":dec,"classification_view":safe(x.get("classification_view")) or "general_is_a","confidence":clip(x.get("confidence",0)),"reason":safe(x.get("reason")),"suggested_intermediate_parent":safe(x.get("suggested_intermediate_parent"))})
    return {"candidates":out}
def vvalidate(v,parent,child):
    dec=safe(v.get("decision"))
    if dec not in VALIDATION_DECISIONS:raise ValueError("bad validation decision")
    return {"parent_id":parent,"child_id":child,"decision":dec,"confidence":clip(v.get("confidence",0)),"membership_test":bool(v.get("membership_test",False)),"kind_of_test":bool(v.get("kind_of_test",False)),"direction_test":bool(v.get("direction_test",False)),"directness_test":bool(v.get("directness_test",False)),"equivalence_exclusion":bool(v.get("equivalence_exclusion",False)),"corpus_supported":bool(v.get("corpus_supported",False)),"evidence_level":safe(v.get("evidence_level")) or "unknown","supporting_sentence_ids":v.get("supporting_sentence_ids",[]) if isinstance(v.get("supporting_sentence_ids",[]),list) else [],"reason":safe(v.get("reason"))}
def vcritic(v,parent,child):
    dec=safe(v.get("decision"))
    if dec not in CRITIC_DECISIONS:raise ValueError("bad critic decision")
    return {"parent_id":parent,"child_id":child,"decision":dec,"confidence":clip(v.get("confidence",0)),"classification_view":safe(v.get("classification_view")),"reason":safe(v.get("reason"))}

def profile(c):return {"concept_id":c["concept_id"],"label":c["preferred_label"],"aliases":c.get("aliases",[]),"entity_type":c["entity_type"],"description":c.get("description",""),"evidence":c.get("evidence_sentences",[])[:5]}
def proposal_prompt(parent,candidates,records):
    return "Child-Proposal Agent. Identify DIRECT is-a children of the parent. The candidate list comes directly from the same-type canonical inventory. The list order is not evidence of a relationship. Return aliases as equivalent, components/activities as related, distant descendants as indirect.\nPARENT:\n"+json.dumps(profile(parent),ensure_ascii=False)+"\nCANDIDATES:\n"+json.dumps([{**profile(records[x["candidate_id"]]),**x} for x in candidates],ensure_ascii=False)+"\nJSON only: {\"candidates\":[{\"candidate_id\":\"...\",\"decision\":\"direct_child|indirect_descendant|equivalent_to_parent|related_not_hierarchical|wrong_semantic_level|insufficient_evidence\",\"classification_view\":\"...\",\"confidence\":0.0,\"reason\":\"...\",\"suggested_intermediate_parent\":\"\"}]}"
def validation_prompt(parent,child,prop):
    return "Evidence Validator Agent. Relations may be implicit across multiple sentences. Compare parent/child definitions and inherited characteristics; explicit co-occurrence is not mandatory. Reject association-only evidence.\nPARENT:\n"+json.dumps(profile(parent),ensure_ascii=False)+"\nCHILD:\n"+json.dumps(profile(child),ensure_ascii=False)+"\nPROPOSAL:\n"+json.dumps(prop,ensure_ascii=False)+"\nJSON only: {\"decision\":\"accept_direct_edge|accept_as_indirect_descendant|reverse_direction|equivalent|related_only|insufficient_evidence|reject\",\"confidence\":0.0,\"membership_test\":true,\"kind_of_test\":true,\"direction_test\":true,\"directness_test\":true,\"equivalence_exclusion\":true,\"corpus_supported\":true,\"evidence_level\":\"explicit|definitionally_entailed|multi_hop_supported|similarity_only|contradicted\",\"supporting_sentence_ids\":[],\"reason\":\"...\"}"
def critic_prompt(parent,child,prop,val):
    return "Counter-Evidence Relation Critic. Try to DISPROVE the is-a edge. Check aliasing, reversed direction, missing intermediate class, association-only evidence, and contradictions.\nPARENT:\n"+json.dumps(profile(parent),ensure_ascii=False)+"\nCHILD:\n"+json.dumps(profile(child),ensure_ascii=False)+"\nPROPOSAL:\n"+json.dumps(prop,ensure_ascii=False)+"\nVALIDATION:\n"+json.dumps(val,ensure_ascii=False)+"\nJSON only: {\"decision\":\"accept|reject|reverse|convert_to_indirect|mark_equivalent|change_classification_view|manual_review\",\"confidence\":0.0,\"classification_view\":\"...\",\"reason\":\"...\"}"

def aggregate(prop,val,crit,args):
    hard=val["membership_test"] and val["kind_of_test"] and val["direction_test"] and val["equivalence_exclusion"]
    conf=.30*prop["confidence"]+.45*val["confidence"]+.25*crit["confidence"]
    if crit["decision"]=="manual_review":conf-=.10
    if not val["corpus_supported"]:conf-=.08
    conf=clip(conf)
    if not hard:return "rejected",conf
    if not val["directness_test"] or val["decision"]=="accept_as_indirect_descendant" or crit["decision"]=="convert_to_indirect":return "pending_indirect",conf
    if val["decision"]=="accept_direct_edge" and crit["decision"] in {"accept","change_classification_view"} and conf>=args.gold_threshold:return "accepted_gold",conf
    if val["decision"] in {"accept_direct_edge","insufficient_evidence"} and crit["decision"] not in {"reject","reverse","mark_equivalent"} and conf>=args.silver_threshold:return "accepted_silver",conf
    if conf>=args.review_threshold:return "manual_review",conf
    return "rejected",conf

# ------------------------------- GRAPH BUILDER -------------------------------
class Builder:
    def __init__(self,args,concepts,index,client):
        import networkx as nx
        self.nx=nx;self.args=args;self.concepts=concepts;self.records={r.concept_id:r.to_dict() for _,r in concepts.iterrows()};self.index=index;self.client=client;self.g=nx.DiGraph();self.review=[];self.expanded=set();self.queue=deque()
        for cid,c in self.records.items():self.g.add_node(cid,**{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in c.items()})
        self.resolve_roots()
    def resolve_roots(self):
        for label in [x.strip() for x in self.args.roots.split(",") if x.strip()]:
            k=norm(label);match=None
            for cid,c in self.records.items():
                if k==norm(c["preferred_label"]) or any(k==norm(a) for a in c.get("aliases",[])):match=cid;break
            if match is None:
                match="ROOT_"+hashlib.sha1(k.encode()).hexdigest()[:10].upper();c={"concept_id":match,"preferred_label":label,"aliases":[],"entity_type":"System" if "subsystem" not in k else "Subsystem","description":f"Trusted structural root for {label} concepts.","evidence_sentences":[],"source_terms":[],"node_origin":"trusted_root"};self.records[match]=c;self.g.add_node(match,**c)
            self.queue.append((match,0))
    def paths(self,cid):
        roots=[n for n in self.g if self.g.in_degree(n)==0];out=[]
        for r in roots:
            try:
                out += [p for p in self.nx.all_simple_paths(self.g,r,cid)]
            except Exception:pass
        return out[:6]
    def run(self):
        accepted=0;proposal_log=self.args.output_dir/"05_child_proposals.jsonl";val_log=self.args.output_dir/"06_edge_validations.jsonl";crit_log=self.args.output_dir/"07_edge_critiques.jsonl";edge_partial=self.args.output_dir/"08_accepted_edges.partial.jsonl"
        while self.queue and accepted<self.args.max_edges:
            parent_id,depth=self.queue.popleft()
            if parent_id in self.expanded or depth>=self.args.max_depth:continue
            self.expanded.add(parent_id);parent=self.records[parent_id];print(f"PARENT {len(self.expanded)}: {parent['preferred_label']} | depth={depth}")
            cands=self.index.retrieve(parent_id,self.records,self.args.top_k,parent.get("entity_type"))
            if not cands:continue
            proposal_rows=[]
            chunk_size=max(5,self.args.candidate_chunk_size)
            for chunk_number,start in enumerate(range(0,len(cands),chunk_size),1):
                chunk=cands[start:start+chunk_size]
                try:
                    ans=self.client.ask(
                        "proposal::"+parent_id+f"::chunk_{chunk_number}",
                        proposal_prompt(parent,chunk,self.records),
                        lambda v, chunk=chunk:vproposal(v,chunk),
                    )
                    proposal_rows.extend(ans["candidates"])
                except Exception as e:
                    self.review.append({"parent_id":parent_id,"edge_status":"manual_review","reason":f"proposal_chunk_{chunk_number}_failed:{e}"})
            for prop in proposal_rows:
                prop={"parent_id":parent_id,**prop};durable(proposal_log,prop)
                if prop["decision"]!="direct_child":
                    if prop["decision"]=="indirect_descendant":self.review.append({**prop,"edge_status":"pending_indirect"})
                    continue
                child_id=prop["candidate_id"];child=self.records[child_id]
                try:val=self.client.ask("validate::"+parent_id+"::"+child_id,validation_prompt(parent,child,prop),lambda v:vvalidate(v,parent_id,child_id));durable(val_log,val)
                except Exception as e:self.review.append({**prop,"edge_status":"manual_review","reason":f"validation_failed:{e}"});continue
                try:crit=self.client.ask("critic::"+parent_id+"::"+child_id,critic_prompt(parent,child,prop,val),lambda v:vcritic(v,parent_id,child_id));durable(crit_log,crit)
                except Exception as e:crit={"decision":"manual_review","confidence":0,"classification_view":prop["classification_view"],"reason":f"critic_failed:{e}"}
                status,conf=aggregate(prop,val,crit,self.args);view=crit.get("classification_view") or prop["classification_view"]
                edge={"parent_id":parent_id,"child_id":child_id,"relation":"is_a","classification_view":view,"edge_status":status,"edge_confidence":conf,"proposal_confidence":prop["confidence"],"validation_confidence":val["confidence"],"critic_confidence":crit["confidence"],"evidence_level":val["evidence_level"],"supporting_sentence_ids":val["supporting_sentence_ids"],"reason":crit["reason"] or val["reason"]}
                if status in {"accepted_gold","accepted_silver"}:
                    self.g.add_edge(parent_id,child_id,**{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in edge.items() if k not in {"parent_id","child_id"}})
                    if not self.nx.is_directed_acyclic_graph(self.g):self.g.remove_edge(parent_id,child_id);edge["edge_status"]="removed_cycle";self.review.append(edge);continue
                    durable(edge_partial,edge);accepted+=1;print(f"  ACCEPT {status}: {parent['preferred_label']} -> {child['preferred_label']} ({conf:.2f})")
                    if depth+1<self.args.max_depth:self.queue.append((child_id,depth+1))
                else:self.review.append(edge)
                if accepted>=self.args.max_edges:break
    def discover_facets(self):
        """LLM-only facet discovery over unplaced concepts, with no clustering.

        Terms are processed in bounded chunks. Proposed facet nodes are never
        inserted automatically; they remain explicit human-review proposals.
        """
        placed = {x for edge in self.g.edges() for x in edge}
        members = [
            cid for cid, concept in self.records.items()
            if cid not in placed and concept.get("node_origin") != "trusted_root"
        ]
        proposals = []
        log = self.args.output_dir / "09_facet_discovery.jsonl"
        chunk_size = max(8, self.args.candidate_chunk_size)
        for chunk_number, start in enumerate(range(0, len(members), chunk_size), 1):
            chunk = members[start:start + chunk_size]
            if len(chunk) < 3:
                continue
            payload = [profile(self.records[cid]) for cid in chunk]
            prompt = (
                "Facet Discovery Agent. Inspect these currently unplaced, corpus-grounded concepts. "
                "Propose an organizing facet only when at least three concepts share a clear, reusable "
                "classification principle. Do not use embedding clusters. Do not claim a generated facet "
                "occurs in the corpus; every facet is a human-review proposal.\nCONCEPTS:\n"
                + json.dumps(payload, ensure_ascii=False)
                + '\nJSON only: {"decision":"create_facet|no_coherent_facet|manual_review",'
                  '"proposed_label":"","classification_view":"","member_concept_ids":[],'
                  '"confidence":0.0,"corpus_supported":false,"reason":""}'
            )
            def vf(v):
                decision = safe(v.get("decision"))
                if decision not in {"create_facet", "no_coherent_facet", "manual_review"}:
                    raise ValueError("bad facet decision")
                ids = v.get("member_concept_ids", [])
                if not isinstance(ids, list):
                    raise ValueError("bad facet ids")
                ids = [safe(x) for x in ids if safe(x) in chunk]
                return {
                    "chunk_number": chunk_number,
                    "decision": decision,
                    "proposed_label": safe(v.get("proposed_label")),
                    "classification_view": safe(v.get("classification_view")),
                    "member_concept_ids": ids,
                    "confidence": clip(v.get("confidence", 0)),
                    "corpus_supported": bool(v.get("corpus_supported", False)),
                    "reason": safe(v.get("reason")),
                    "node_origin": "proposed",
                    "requires_human_approval": True,
                }
            try:
                row = self.client.ask(f"facet_chunk::{chunk_number}", prompt, vf)
            except Exception as exc:
                row = {
                    "chunk_number": chunk_number,
                    "decision": "manual_review",
                    "proposed_label": "",
                    "classification_view": "",
                    "member_concept_ids": chunk,
                    "confidence": 0.0,
                    "corpus_supported": False,
                    "reason": f"facet_agent_failed:{exc}",
                    "node_origin": "proposed",
                    "requires_human_approval": True,
                }
            durable(log, row)
            proposals.append(row)
        return proposals

    def global_audit(self):
        """Audit the complete graph; only removal actions are applied automatically."""
        edges=[]
        for u,v,d in self.g.edges(data=True):
            edges.append({"parent_id":u,"parent":self.records[u]["preferred_label"],"child_id":v,"child":self.records[v]["preferred_label"],"classification_view":d.get("classification_view","general_is_a"),"confidence":d.get("edge_confidence",0),"evidence_level":d.get("evidence_level","")})
        if not edges:
            result={"summary":"No edges available for global audit.","actions":[],"confidence":1.0}
            (self.args.output_dir/"10_global_graph_audit.json").write_text(json.dumps(result,indent=2))
            return result
        prompt="Global Graph Auditor. Review this induced polyhierarchical taxonomy for reversed, unsupported, equivalent, redundant, or semantically mixed edges. Multiple parents are allowed when justified. Return conservative actions; do not invent concepts or relations.\nEDGES:\n"+json.dumps(edges[:200],ensure_ascii=False)+"\nJSON only: {\"summary\":\"\",\"actions\":[{\"action\":\"keep|remove|reverse|change_view|manual_review\",\"parent_id\":\"\",\"child_id\":\"\",\"new_view\":\"\",\"reason\":\"\"}],\"confidence\":0.0}"
        valid_pairs={(e["parent_id"],e["child_id"]) for e in edges}
        def va(v):
            acts=v.get("actions",[])
            if not isinstance(acts,list):raise ValueError("bad audit actions")
            out=[]
            for a in acts:
                action=safe(a.get("action"));u=safe(a.get("parent_id"));w=safe(a.get("child_id"))
                if action not in {"keep","remove","reverse","change_view","manual_review"} or (u,w) not in valid_pairs:continue
                out.append({"action":action,"parent_id":u,"child_id":w,"new_view":safe(a.get("new_view")),"reason":safe(a.get("reason"))})
            return {"summary":safe(v.get("summary")),"actions":out,"confidence":clip(v.get("confidence",0))}
        try:result=self.client.ask("global_audit",prompt,va)
        except Exception as e:result={"summary":f"audit_failed:{e}","actions":[],"confidence":0.0}
        for a in result["actions"]:
            u,v=a["parent_id"],a["child_id"]
            if not self.g.has_edge(u,v):continue
            if a["action"]=="remove":
                data=dict(self.g.edges[u,v]);self.g.remove_edge(u,v);self.review.append({"parent_id":u,"child_id":v,"edge_status":"removed_global_audit","reason":a["reason"],**data})
            elif a["action"]=="change_view":self.g.edges[u,v]["classification_view"]=a["new_view"] or self.g.edges[u,v].get("classification_view","general_is_a")
            elif a["action"]=="reverse":self.review.append({"parent_id":u,"child_id":v,"edge_status":"manual_review_reverse_proposed","reason":a["reason"]})
            elif a["action"]=="manual_review":self.review.append({"parent_id":u,"child_id":v,"edge_status":"manual_review_global_audit","reason":a["reason"]})
        (self.args.output_dir/"10_global_graph_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        return result

    def finalize(self):
        # Remove duplicate self edges and break cycles by lowest edge confidence.
        for u,v in list(self.g.edges()):
            if u==v:self.review.append({"parent_id":u,"child_id":v,"edge_status":"removed_self_loop"});self.g.remove_edge(u,v)
        while not self.nx.is_directed_acyclic_graph(self.g):
            cyc=self.nx.find_cycle(self.g);edge=min(cyc,key=lambda e:float(self.g.edges[e[0],e[1]].get("edge_confidence",0)));data=dict(self.g.edges[edge[0],edge[1]]);self.g.remove_edge(*edge[:2]);self.review.append({"parent_id":edge[0],"child_id":edge[1],"edge_status":"removed_cycle",**data})
        # Per-view transitive reduction only where safe.
        views=set(self.g.edges[u,v].get("classification_view","general_is_a") for u,v in self.g.edges())
        for view in views:
            h=self.nx.DiGraph([(u,v) for u,v in self.g.edges() if self.g.edges[u,v].get("classification_view","general_is_a")==view])
            if len(h)>0 and self.nx.is_directed_acyclic_graph(h):
                red=self.nx.transitive_reduction(h)
                for u,v in list(h.edges()):
                    if not red.has_edge(u,v) and self.g.has_edge(u,v):data=dict(self.g.edges[u,v]);self.g.remove_edge(u,v);self.review.append({"parent_id":u,"child_id":v,"edge_status":"removed_transitive",**data})
    def export(self):
        out=self.args.output_dir;nodes=[]
        for cid,c in self.records.items():nodes.append({**c,"in_graph":cid in self.g,"in_degree":self.g.in_degree(cid) if cid in self.g else 0,"out_degree":self.g.out_degree(cid) if cid in self.g else 0,"expanded":cid in self.expanded})
        edges=[]
        for u,v,d in self.g.edges(data=True):edges.append({"parent_id":u,"parent_label":self.records[u]["preferred_label"],"child_id":v,"child_label":self.records[v]["preferred_label"],**d})
        pd.DataFrame(nodes).to_csv(out/"11_taxonomy_nodes.csv",index=False);pd.DataFrame(edges).to_csv(out/"12_taxonomy_edges.csv",index=False);pd.DataFrame(self.review).to_csv(out/"13_review_removed_edges.csv",index=False)
        placed=set([x for e in self.g.edges() for x in e]);unplaced=[c for cid,c in self.records.items() if cid not in placed and c.get("node_origin")!="trusted_root"];pd.DataFrame(unplaced).to_csv(out/"14_unplaced_concepts.csv",index=False)
        stats={"nodes":self.g.number_of_nodes(),"edges":self.g.number_of_edges(),"roots":sum(1 for n in self.g if self.g.in_degree(n)==0),"leaves":sum(1 for n in self.g if self.g.out_degree(n)==0),"multiple_parent_nodes":sum(1 for n in self.g if self.g.in_degree(n)>1),"weakly_connected_components":self.nx.number_weakly_connected_components(self.g) if len(self.g) else 0,"cycles":0,"expanded_nodes":len(self.expanded),"unplaced_concepts":len(unplaced)};pd.DataFrame([{"metric":k,"value":v} for k,v in stats.items()]).to_csv(out/"15_graph_statistics.csv",index=False)
        # GraphML scalar sanitation.
        g2=self.nx.DiGraph()
        for n,d in self.g.nodes(data=True):g2.add_node(n,**{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,set,tuple)) else v.item() if isinstance(v,np.generic) else v) for k,v in d.items() if v is not None})
        for u,v,d in self.g.edges(data=True):g2.add_edge(u,v,**{k:(json.dumps(x,ensure_ascii=False) if isinstance(x,(list,dict,set,tuple)) else x.item() if isinstance(x,np.generic) else x) for k,x in d.items() if x is not None})
        try:self.nx.write_graphml(g2,out/"16_taxonomy.graphml")
        except Exception as e:(out/"16_taxonomy_graphml_error.txt").write_text(str(e))
        (out/"17_taxonomy.json").write_text(json.dumps({"nodes":nodes,"edges":edges},ensure_ascii=False,indent=2));return stats

# ---------------------------------- CLI/MAIN ----------------------------------
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--typing-results",type=Path,default=DEFAULT_TYPING);p.add_argument("--typing-contexts",type=Path,default=DEFAULT_CONTEXTS);p.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS);p.add_argument("--alias-workbook",type=Path,default=DEFAULT_ALIASES);p.add_argument("--alias-sheet",default="S2_GT_ALIASES");p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--model",default=DEFAULT_MODEL);p.add_argument("--ollama-url",default=DEFAULT_URL);p.add_argument("--temperature",type=float,default=0.0);p.add_argument("--timeout",type=int,default=300);p.add_argument("--retries",type=int,default=3);p.add_argument("--candidate-chunk-size",type=int,default=20,help="Number of same-type candidates per Child-Proposal prompt")
    p.add_argument("--roots",default="unmanned aerial vehicle,unmanned aircraft system,control systems");p.add_argument("--top-k",type=int,default=0,help="0 evaluates all same-type candidates; positive values cap candidates per parent");p.add_argument("--max-depth",type=int,default=6);p.add_argument("--max-edges",type=int,default=100);p.add_argument("--max-concepts",type=int,default=0);p.add_argument("--gold-threshold",type=float,default=.78);p.add_argument("--silver-threshold",type=float,default=.62);p.add_argument("--review-threshold",type=float,default=.50);p.add_argument("--skip-ollama-check",action="store_true")
    return p.parse_args()

def main():
    args=parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for p in (args.typing_results,args.corpus,args.alias_workbook):
        if not p.is_file():raise FileNotFoundError(p)
    client=Ollama(args.ollama_url,args.model,args.output_dir,args.temperature,args.timeout,args.retries)
    if not args.skip_ollama_check:client.check()
    started=time.perf_counter();typed=load_typed(args.typing_results);aliases=load_aliases(args.alias_workbook,args.alias_sheet);corpus=Corpus.build(args.corpus.read_text(encoding="utf-8",errors="replace"))
    pd.DataFrame([{"sentence_id":f"SENT_{i:06d}","sentence":s} for i,s in enumerate(corpus.sents)]).to_csv(args.output_dir/"00_corpus_sentences.csv",index=False)
    filtered=run_filter(args,typed,corpus,aliases,client);accepted=filtered[filtered.taxonomy_action.eq("accept")].copy()
    if accepted.empty:raise RuntimeError("No corpus-grounded terms accepted")
    concepts=canonicalize(accepted,aliases)
    if args.max_concepts>0:concepts=concepts.head(args.max_concepts).copy()
    concepts.to_csv(args.output_dir/"01_canonical_concept_inventory.csv",index=False)
    index=DirectCandidateIndex(concepts); concepts.assign(candidate_pool="same_type_inventory").to_csv(args.output_dir/"02_candidate_inventory.csv",index=False)
    builder=Builder(args,index.concepts,index,client);print("Starting recursive parent-child induction...");builder.run();builder.discover_facets();builder.global_audit();builder.finalize();stats=builder.export()
    manifest={"created_utc":now_utc(),"prompt_version":PROMPT_VERSION,"inputs":{"typing_results":str(args.typing_results),"typing_sha256":sha(args.typing_results),"corpus":str(args.corpus),"corpus_sha256":sha(args.corpus),"alias_workbook":str(args.alias_workbook)},"filter":{"input_terms":len(typed),"accepted":len(accepted),"rejected":int(filtered.taxonomy_action.eq("reject").sum()),"manual_review":int(filtered.taxonomy_action.eq("manual_review").sum())},"taxonomy":stats,"live_qwen_calls":client.live,"cache_hits":client.hits,"runtime_seconds":time.perf_counter()-started}
    (args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("="*100);print("COMPLETE SINGLE-FILE AGENTIC TAXONOMY PIPELINE COMPLETED");print("="*100);print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
