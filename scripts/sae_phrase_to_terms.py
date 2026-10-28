#!/usr/bin/env python
"""
uav_terms_pipeline_config.py
============================
SAE feature-phrases CSV  ->  clean candidate TERMS.

>>> NO command-line arguments. <<<
Just edit the CONFIG section below and run:

        python uav_terms_pipeline_config.py

Pipeline:
    CSV (`phrase` column)
      -> tokenize
      -> mark VERBS / ADVERBS / function words as BOUNDARIES
         (keep useful modifiers: unmanned, powered, remotely piloted, tactical, ...)
      -> noun chunks = runs between boundaries that END in a noun-ish head
      -> emit terms (all sub-spans, or whole chunk if MAXIMAL=True)
      -> optional plural/acronym folding (FOLD=True)
      -> aggregate frequency + max score + source feature ids
      -> write OUTPUT_CSV

Auto-uses spaCy POS tagging if an English model is installed (cleaner);
otherwise falls back to the built-in rule-based tagger (no downloads needed).
"""

import csv, re
from collections import defaultdict, Counter

# ============================================================================ #
#                                  CONFIG                                      #
#            >>>>>  EDIT THESE VALUES, THEN RUN THE FILE  <<<<<                 #
# ============================================================================ #

INPUT_CSV   = "ontology_experiment_framework/stage_outputs/sae_yake_keybert/uav_sae_e50_kw1to5_feature_phrases.csv"   # <-- path to your CSV
OUTPUT_CSV  = "ontology_experiment_framework/stage_outputs/sae_yake_keybert/uav_clean_terms.csv"                      # <-- where to save terms

MIN_WORDS   = 1        # shortest term (in words) to keep
MAX_WORDS   = 4        # longest term (in words) to keep
MIN_FREQ    = 2        # drop terms that occur fewer than this many times

MAXIMAL     = True     # True  = keep only whole noun chunks (tight, readable)
                       # False = keep ALL contiguous sub-spans (fuller list)
FOLD        = True     # True  = fold plurals/acronyms (uavs->uav, systems->system)
KEEP_NUMBERS = False   # True  = keep numeric tokens (e.g. 18, 2013); False = drop

USE_SPACY   = True     # True  = use spaCy POS tagging IF a model is installed
SPACY_MODEL = "en_core_web_sm"

# ============================================================================ #
#                       (usually no need to edit below)                        #
# ============================================================================ #

FUNC = {
 "the","a","an","of","and","or","to","in","on","for","with","by","from","as","at",
 "is","are","be","been","being","was","were","that","this","these","those","it","its",
 "their","they","them","we","our","you","your","he","she","his","her","will","would",
 "shall","should","can","could","may","might","must","has","have","had","do","does","did",
 "than","then","such","into","out","up","down","over","under","both","each","any","some",
 "all","more","most","other","another","no","not","but","if","so","because","however",
 "while","during","after","before","which","who","whom","whose","what","when","where",
 "how","there","here","one","two","three","four","five","six","also","per","via","about",
 "between","within","without","many","several","various","much","less","least","enough",
 "just","still","yet","already","ever","like","likely","recent","recently","current",
 "currently","future","near","new","major","key","first","last","own","same","only",
}
ADV = {
 "particularly","especially","generally","usually","typically","mostly","largely",
 "directly","indirectly","significantly","additionally","finally","already","nearly",
 "always","often","sometimes","widely","widespread","heavily","cautiously","previously",
 "continually","slowly","laboriously","surprisingly","basically","actively",
}
VERBS = {
 "carry","carries","carried","sent","send","sends","provide","provides","provided",
 "operate","operates","include","includes","included","given","give","gives","make",
 "makes","made","take","takes","taken","get","gets","got","become","becomes","seen",
 "see","sees","cause","causes","caused","need","needs","use","uses","used","order",
 "orders","ordered","market","markets","marketed","categorize","categorized","launch",
 "launches","launched","develop","develops","developed","acquire","acquires","acquired",
 "envision","envisioned","incorporate","incorporated","delegate","delegated",
 "disassemble","disassembled","integrate","integrates","integrated","procure","procured",
 "deploy","deployed","present","presents","presented","publish","published","announce",
 "announced","estimate","estimated","estimates","derive","derived","identify","identifies",
 "identified","enable","enables","enabled","overcome","address","addresses","addressed",
 "convert","converts","converted","replace","replaces","replaced","fly","flown","flew",
 "flies","fitted","fit","fits","designate","designated","demonstrate","demonstrated",
 "correlate","correlates","correlated","exhibit","exhibiting","exhibited","adopt","adopts",
 "adopted","recover","recovers","recovered","hover","hovering","monitor","monitors",
 "manufacture","manufactured","manufactures","test","tests","tested","spent",
 "spend","spends","grown","grow","grows","grew","refers","refer","referred","signify",
 "signified","meant","mean","means","consist","consisted","divided","divide",
 "divides","classify","classifies","classified","investigate","investigated",
 "start","starts","started","handed","hand","hands","aim","aims","aimed",
 "gain","gains","gained","struggle","offer","offers","offered","promise",
 "promises","promised","depend","depends","depended","determined","determine",
 "considered","consider","considers","ensure","ensures","ensured","access","accesses",
 "accessed","issue","issues","issued","transport","transports","transported",
 "control","controls","controlled","dictate","dictates","dictated",
 "affect","affects","affected","impart","imparts","imparted","extend","extends","extended",
}
KEEP_ING = {
 "training","monitoring","landing","rulemaking","targeting","engineering","manufacturing",
 "planning","warning","surveillance","modeling","imaging","tracking","sensing","mapping",
 "unmanned","remotely",
}
KEEP_ED = {
 "unmanned","manned","armed","powered","piloted","remotely-piloted","gimbaled","ducted",
 "based","fixed","advanced","integrated","controlled","automated","dedicated","related",
 "increased","reduced","improved","combined","expected","proposed","defined","designated",
 "tactical",
}
MODIFIER_ONLY = {"remotely","highly","fully","semi","very","low","high","non",
                 "anti","multi","co","sub","pre","post","re"}

ACRONYM_CANON = {          # variant -> canonical (used when FOLD=True)
    "uavs":"uav", "ua":"uav", "aircrafts":"aircraft", "vehicles":"vehicle",
    "systems":"system", "technologies":"technology", "applications":"application",
    "operators":"operator", "sensors":"sensor", "links":"link", "classes":"class",
    "programs":"program", "tasks":"task", "roles":"role", "requirements":"requirement",
    "companies":"company", "missions":"mission", "definitions":"definition",
}

NUM_RE = re.compile(r"^\d+([.,]\d+)?$")
TOK_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]*|\d+")

# ----------------------------------------------------------- fallback tagger --
def classify(tok, drop_numbers):
    t = tok.lower()
    if t in FUNC or t in ADV:            return "BOUND"
    if drop_numbers and NUM_RE.match(t): return "BOUND"
    if t in KEEP_ING or t in KEEP_ED:    return "MOD"
    if t in VERBS:                       return "BOUND"
    if t.endswith("ing") and t not in KEEP_ING: return "BOUND"
    return "WORD"

def noun_runs_fallback(phrase, drop_numbers):
    runs, cur = [], []
    for tk in TOK_RE.findall(phrase.lower()):
        c = classify(tk, drop_numbers)
        if c == "BOUND":
            if cur: runs.append(cur); cur=[]
        else:
            cur.append((tk, c))
    if cur: runs.append(cur)
    return runs

# ----------------------------------------------------------- spaCy tagger -----
def noun_runs_spacy(nlp, phrase, drop_numbers):
    KEEP={"NOUN","PROPN","ADJ"}
    doc=nlp(phrase.lower()); runs=[]; cur=[]
    for tok in doc:
        keep = tok.pos_ in KEEP and not tok.is_stop
        if drop_numbers and tok.like_num: keep=False
        if keep:
            role="WORD" if tok.pos_ in {"NOUN","PROPN"} else "MOD"
            cur.append((tok.text, role))
        else:
            if cur: runs.append(cur); cur=[]
    if cur: runs.append(cur)
    return runs

# ----------------------------------------------------------- term spans -------
def spans_from_run(run, min_w, max_w, maximal):
    words=[w for w,_ in run]; n=len(words)
    if maximal:
        if run[-1][1]=="WORD" and min_w<=n<=max_w:
            yield " ".join(words)
        return
    for i in range(n):
        for j in range(i+1,n+1):
            if not (min_w<=(j-i)<=max_w): continue
            if run[j-1][1]!="WORD": continue
            if (j-i)==1 and words[i] in MODIFIER_ONLY: continue
            yield " ".join(words[i:j])

def fold(term):
    return " ".join(ACRONYM_CANON.get(w, w) for w in term.split())

# ----------------------------------------------------------- io ---------------
def load(path):
    rows=[]
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd=csv.DictReader(fh)
        cols={c.replace("\\","").strip():c for c in rd.fieldnames}
        pc=cols.get("phrase") or cols.get("term")
        if not pc:
            raise SystemExit("ERROR: CSV must have a `phrase` (or `term`) column")
        fc=cols.get("feature_id"); sc=cols.get("combined_score")
        for r in rd:
            ph=(r.get(pc) or "").strip()
            if not ph: continue
            fid=(r.get(fc) or "").strip() if fc else ""
            try: s=float(r.get(sc) or 0)
            except: s=0.0
            rows.append((ph,fid,s))
    return rows

# ----------------------------------------------------------- run --------------
def run():
    drop_numbers = not KEEP_NUMBERS

    nlp=None; mode="fallback"
    if USE_SPACY:
        try:
            import spacy
            nlp=spacy.load(SPACY_MODEL, disable=["ner","lemmatizer"])
            mode="spacy"
        except Exception:
            print(f"[info] spaCy model '{SPACY_MODEL}' not found -> using fallback tagger")
            mode="fallback"

    rows=load(INPUT_CSV)
    freq=defaultdict(int); mx=defaultdict(float); feats=defaultdict(set)
    variants=defaultdict(set)

    for ph,fid,s in rows:
        runs = noun_runs_spacy(nlp,ph,drop_numbers) if mode=="spacy" \
               else noun_runs_fallback(ph,drop_numbers)
        for r in runs:
            for term in spans_from_run(r, MIN_WORDS, MAX_WORDS, MAXIMAL):
                key = fold(term) if FOLD else term
                if FOLD and key!=term: variants[key].add(term)
                freq[key]+=1; mx[key]=max(mx[key],s)
                if fid: feats[key].add(fid)

    terms=[t for t,c in freq.items() if c>=MIN_FREQ]
    terms.sort(key=lambda t:(-freq[t],-mx[t],t))

    with open(OUTPUT_CSV,"w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh)
        header=["term","n_words","frequency","max_combined_score","num_features","feature_ids"]
        if FOLD: header.append("folded_variants")
        w.writerow(header)
        for t in terms:
            row=[t,len(t.split()),freq[t],round(mx[t],4),len(feats[t]),";".join(sorted(feats[t]))]
            if FOLD: row.append(";".join(sorted(variants[t])))
            w.writerow(row)

    print(f"[mode={mode}] input rows: {len(rows)}  ->  clean terms: {len(terms)}")
    print(f"saved: {OUTPUT_CSV}")
    c=Counter(len(t.split()) for t in terms)
    for k in sorted(c): print(f"   {k}-word: {c[k]}")

if __name__=="__main__":
    run()
