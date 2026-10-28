from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from .utils import text_sentences, stable_id

REF_HEAD=re.compile(r'^\s{0,3}(?:#{1,6}\s*)?(references|bibliography|works cited)\s*:?.*$',re.I)
HEAD=re.compile(r'^\s{0,3}#{1,6}\s+(.+?)\s*$')
MISSING_PAGEFAIL=re.compile(r'\bMISSINGPAGEFAIL:\d+\b',re.I)
LATEX_ENV_START=re.compile(r'\\begin\{([a-zA-Z*]+)\}')
LATEX_ENV_END=re.compile(r'\\end\{([a-zA-Z*]+)\}')
LATEX_TEXT_CMD=re.compile(r'\\(?:section|subsection|subsubsection|paragraph|subparagraph|chapter|title|textbf|textit|emph|underline|texttt|textsc|textrm|textsf)\*?(?:\[[^\]]*\])?\{([^{}]*)\}')
LATEX_CITE_CMD=re.compile(r'\\(?:cite|citet|citep|citealt|citeauthor|citeyear|ref|eqref|autoref|pageref|label)\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{[^{}]*\}')
LATEX_INLINE_MATH=re.compile(r'\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$(.+?)\$')
LATEX_ANY_CMD=re.compile(r'\\[a-zA-Z@]+\*?(?:\[[^\]]*\])*(?:\{[^{}]*\})*')
LATEX_TABLE_RULE=re.compile(r'\\(?:hline|cline|toprule|midrule|bottomrule)\b')
LATEX_BIB_CMD=re.compile(r'\\(?:bibitem|bibliography|bibliographystyle|nocite)\b')

def _strip_latex_line(text):
    text=LATEX_TEXT_CMD.sub(r'\1',text)
    text=LATEX_CITE_CMD.sub(' ',text)
    text=LATEX_INLINE_MATH.sub(' ',text)
    text=LATEX_ANY_CMD.sub(' ',text)
    text=re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_mmd(path):
    raw=Path(path).read_text(encoding='utf-8',errors='ignore')
    lines=raw.splitlines(); out=[]; stats={'input_lines':len(lines)}
    in_fence=False; in_refs=False; in_latex_env=None; removed={'references':0,'tables':0,'captions':0,'latex':0,'noise':0}
    for i,line in enumerate(lines):
        s=line.strip()
        if s.startswith('```') or s.startswith('~~~'):
            in_fence=not in_fence; removed['noise']+=1; continue
        if REF_HEAD.match(s): in_refs=True; removed['references']+=1; continue
        if in_refs:
            if HEAD.match(s) and not REF_HEAD.match(s): in_refs=False
            else: removed['references']+=1; continue
        if in_fence: removed['noise']+=1; continue
        if in_latex_env:
            if LATEX_ENV_END.search(s):
                in_latex_env=None
            removed['latex']+=1
            continue
        latex_start=LATEX_ENV_START.search(s)
        latex_end=LATEX_ENV_END.search(s)
        if latex_start:
            if latex_end:
                removed['latex']+=1
                continue
            in_latex_env=latex_start.group(1)
            removed['latex']+=1
            continue
        if re.match(r'^\s*<table\b',s,re.I) or re.match(r'^\s*</?(tr|td|th)\b',s,re.I): removed['tables']+=1; continue
        if '|' in s and (s.count('|')>=2 or re.match(r'^\s*\|?\s*:?-+',s)): removed['tables']+=1; continue
        if re.match(r'^\s*\\begin\{(?:table|tabular)\}',s) or re.match(r'^\s*\\end\{(?:table|tabular)\}',s): removed['tables']+=1; continue
        if LATEX_TABLE_RULE.search(s) or LATEX_BIB_CMD.search(s):
            removed['latex']+=1
            continue
        if re.match(r'^\s*(table|figure|fig\.)\s+\d+[.:]',s,re.I): removed['captions']+=1; continue
        s=re.sub(r'!\[[^]]*\]\([^)]*\)',' ',s)
        s=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',s)
        s=re.sub(r'<[^>]+>',' ',s)
        s=MISSING_PAGEFAIL.sub(' ',s)
        s=_strip_latex_line(s)
        s=re.sub(r'[`*_]','',s)
        if s and sum(c.isalpha() for c in s) < 3 and not HEAD.match(s): removed['noise']+=1; continue
        out.append(s)
    text='\n'.join(out)
    text=re.sub(r'(?<=\w)-\n(?=[a-z])','',text)
    text=re.sub(r'\n{3,}','\n\n',text)
    text=re.sub(r'[ \t]+',' ',text).strip()
    stats.update(removed); stats['output_chars']=len(text)
    return text, pd.DataFrame([stats])

def chunk_text(text, max_words=220, overlap_sentences=1):
    sections=[]; title='document'; buf=[]
    for line in text.splitlines():
        m=HEAD.match(line)
        if m:
            if buf: sections.append((title,' '.join(buf)))
            title=m.group(1).strip(); buf=[]
        else: buf.append(line)
    if buf: sections.append((title,' '.join(buf)))
    rows=[]; idx=0
    for title,body in sections:
        sents=text_sentences(body); current=[]; words=0
        for sent in sents:
            n=len(sent.split())
            if current and words+n>max_words:
                txt=' '.join(current); rows.append({'chunk_id':stable_id('chunk',f'{idx}:{txt}'),'chunk_index':idx,'section':title,'text':txt,'sentence_count':len(current)}); idx+=1
                current=current[-overlap_sentences:] if overlap_sentences else []
                words=sum(len(x.split()) for x in current)
            current.append(sent); words+=n
        if current:
            txt=' '.join(current); rows.append({'chunk_id':stable_id('chunk',f'{idx}:{txt}'),'chunk_index':idx,'section':title,'text':txt,'sentence_count':len(current)}); idx+=1
    return pd.DataFrame(rows,columns=['chunk_id','chunk_index','section','text','sentence_count'])
