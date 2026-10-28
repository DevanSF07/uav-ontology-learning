# src/preprocess.py

import re


def clean_text(text: str) -> str:
    """
    Clean already-processed MMD text for term extraction benchmarking.
    Main goal: remove table noise and obvious structural artefacts
    without harming real prose content.
    """

    # ------------------------------------------------------------
    # 1) Remove page artifacts / extraction artifacts
    # ------------------------------------------------------------
    text = re.sub(r"\[MISSING_PAGE_FAIL:\d+\]", " ", text)

    # ------------------------------------------------------------
    # 2) Remove full LaTeX table environments
    # ------------------------------------------------------------
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", text, flags=re.S)

    # In case tabular appears outside table
    text = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.S)

    # ------------------------------------------------------------
    # 3) Remove table captions like:
    #    Table 5.1: ...
    #    Table 8: ...
    # ------------------------------------------------------------
    text = re.sub(r"(?im)^Table\s+\d+(\.\d+)?\s*:\s.*$", " ", text)

    # ------------------------------------------------------------
    # 4) Remove common LaTeX table/math commands that leak into text
    # ------------------------------------------------------------
    latex_commands = [
        r"\\hline",
        r"\\cline\{[^}]*\}",
        r"\\multirow\{[^}]*\}\{[^}]*\}\{[^}]*\}",
        r"\\multicolumn\{[^}]*\}\{[^}]*\}\{[^}]*\}",
        r"\\textless\{\}",
        r"\\textgreater\{\}",
        r"\\mathrm\{[^}]*\}",
        r"\\text\{[^}]*\}",
    ]
    for pattern in latex_commands:
        text = re.sub(pattern, " ", text)

    # remove inline/display LaTeX math
    text = re.sub(r"\\\((.*?)\\\)", " ", text, flags=re.S)
    text = re.sub(r"\\\[(.*?)\\\]", " ", text, flags=re.S)

    # ------------------------------------------------------------
    # 5) Remove markdown links but keep anchor text
    #    [text](url) -> text
    # ------------------------------------------------------------
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)

    # remove raw URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # ------------------------------------------------------------
    # 6) Remove email lines / obvious metadata clutter
    # ------------------------------------------------------------
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # ------------------------------------------------------------
    # 7) Optionally remove references section completely
    #    This is strongly recommended for term extraction benchmarking,
    #    because reference lists create lots of non-domain noise.
    # ------------------------------------------------------------
    text = re.split(r"(?im)^##?\s*References\s*$", text)[0]

    # ------------------------------------------------------------
    # 8) Remove repeated markdown formatting symbols
    # ------------------------------------------------------------
    text = text.replace("*", " ")
    text = text.replace("_", " ")
    text = text.replace("|", " ")

    # ------------------------------------------------------------
    # 9) Normalize some odd symbols commonly seen in OCR/export text
    # ------------------------------------------------------------
    text = text.replace("&amp;", "&")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'")

    # ------------------------------------------------------------
    # 10) Collapse whitespace
    # ------------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_into_docs(text: str):
    """
    Create segments for TF-IDF-like scoring / chunk-aware extraction.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s{2,}", text) if p.strip()]

    if len(paras) < 3:
        sents = re.split(r"(?<=[.!?])\s+", text)
        paras = [
            " ".join(sents[i:i+5]).strip()
            for i in range(0, len(sents), 5)
            if " ".join(sents[i:i+5]).strip()
        ]

    return paras


def regex_word_tokenize(text: str):
    """
    Keep technical tokens like:
    UAV, MTOW, EO/IR, C-band, line-of-sight
    """
    return re.findall(r"[A-Za-z][A-Za-z0-9\-_\/]*", text)
