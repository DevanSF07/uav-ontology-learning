# io_utils.py

import csv
from pathlib import Path
from collections import Counter

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def read_input(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError(
                "PyPDF2 not available. Install it in your virtual environment: pip install PyPDF2"
            )

        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")

        return "\n".join(pages)

    return path.read_text(encoding="utf-8", errors="ignore")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_acronyms(path, acronyms: Counter):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["acronym", "count"])
        for a, c in acronyms.most_common():
            writer.writerow([a, c])