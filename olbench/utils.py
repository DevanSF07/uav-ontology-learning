from __future__ import annotations
import csv, hashlib, json, re, urllib.request
from pathlib import Path
from typing import Any
import pandas as pd

def canonical(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def stable_id(prefix: str, text: str) -> str:
    return prefix + "_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def file_hash(path: str | Path) -> str:
    p=Path(path); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def dump_json(path: str | Path, obj: Any):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_df(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def text_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]

def ollama_json(prompt: str, model: str, url: str='http://localhost:11434/api/generate', timeout: int=120):
    payload=json.dumps({'model':model,'prompt':prompt,'stream':False,'format':'json'}).encode()
    req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        outer=json.loads(r.read().decode())
    return json.loads(outer['response'])

def dependency_status():
    import importlib.util
    return {x: bool(importlib.util.find_spec(x)) for x in ['yake','keybert','sentence_transformers','gliner','spacy']}
