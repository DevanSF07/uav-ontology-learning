from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

def representation_table(terms,alias_concepts=None,representation='term'):
    u=terms.sort_values('score',ascending=False).drop_duplicates('term_canonical').copy()
    alias_map={}
    if alias_concepts is not None and not alias_concepts.empty:
        alias_map=alias_concepts.groupby('member_term').aliases.first().to_dict()
    if representation=='term': u['representation_text']=u.raw_term
    elif representation=='term_context': u['representation_text']=u.raw_term+' [CONTEXT] '+u.context_text.fillna('')
    elif representation=='term_context_alias': u['representation_text']=u.raw_term+' [ALIASES] '+u.term_canonical.map(alias_map).fillna('')+' [CONTEXT] '+u.context_text.fillna('')
    else: raise ValueError('Unknown representation '+representation)
    return u

def _label_groups(df,id_col,top_n=5):
    return df.groupby(id_col).apply(lambda g:', '.join(g.sort_values('membership_score',ascending=False).raw_term.head(top_n)),include_groups=False).reset_index(name='group_label')

def embedding_cluster(terms,alias_concepts=None,representation='term_context',model='all-mpnet-base-v2',n_clusters=8,random_state=42,**_):
    u=representation_table(terms,alias_concepts,representation); texts=u.representation_text.tolist(); backend='sentence_transformers'
    try:
        from sentence_transformers import SentenceTransformer
        X=SentenceTransformer(model).encode(texts,normalize_embeddings=True,show_progress_bar=False)
    except Exception:
        backend='tfidf_fallback'; X=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=1).fit_transform(texts)
    k=max(1,min(n_clusters,len(u))); km=KMeans(n_clusters=k,random_state=random_state,n_init=10).fit(X)
    u['cluster_id']=km.labels_; u['membership_score']=1-np.minimum(km.transform(X).min(axis=1),1); u['embedding_backend']=backend
    labels=_label_groups(u,'cluster_id'); return u,labels

def embedding_nmf(terms,alias_concepts=None,representation='term_context',n_factors=8,random_state=42,**_):
    u=representation_table(terms,alias_concepts,representation); texts=u.representation_text.tolist()
    # NMF requires non-negative input, so use TF-IDF rather than raw signed embeddings.
    v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=1,max_features=20000); X=v.fit_transform(texts)
    k=max(1,min(n_factors,X.shape[0],X.shape[1])); nmf=NMF(n_components=k,init='nndsvda',random_state=random_state,max_iter=500); W=nmf.fit_transform(X)
    u['factor_id']=W.argmax(axis=1); u['membership_score']=W.max(axis=1); u['nmf_feature_space']='tfidf_nonnegative'
    labels=_label_groups(u,'factor_id'); return u,labels
