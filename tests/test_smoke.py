from pathlib import Path
from olbench.preprocess import clean_mmd,chunk_text
from olbench.extractors import tfidf
from olbench.aliasing import regex_alias

def test_clean_table_reference(tmp_path):
    p=tmp_path/'x.mmd'; p.write_text('# A\nText system.\n|A|B|\n|--|--|\n# References\nBook')
    text,report=clean_mmd(p)
    assert 'References' not in text and '|A|B|' not in text

def test_pipeline_parts(tmp_path):
    p=tmp_path/'x.mmd'; p.write_text('# Chapter\nThe flight control system contains a navigation component. UAS means unmanned aircraft system.')
    text,_=clean_mmd(p); chunks=chunk_text(text,50,0); terms=tfidf(chunks,5)
    assert len(chunks)==1 and len(terms)>0
    assert regex_alias(terms) is not None
