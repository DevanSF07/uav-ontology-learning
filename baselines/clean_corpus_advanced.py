from pathlib import Path
import re


# ==================================================
# ✅ STEP 1: REMOVE STRUCTURAL BLOCKS (SAFE)
# ==================================================
def remove_blocks(text: str):

    # --------------------------
    # Remove LaTeX tables
    text = re.sub(
        r"\\begin\{table\}.*?\\end\{table\}",
        "",
        text,
        flags=re.DOTALL
    )

    # --------------------------
    # Remove LaTeX math expressions
    text = re.sub(r"\\\(.*?\\\)", "", text)
    text = re.sub(r"\\\[.*?\\\]", "", text)

    # --------------------------
    # Remove references section completely
    text = re.split(r"(?i)##\s*references", text)[0]

    return text


# ==================================================
# ✅ STEP 2: CLEAN LINES (KEEP CONTENT)
# ==================================================
def clean_line(line: str):

    line = line.strip()

    if not line:
        return None

    # --------------------------
    # Remove headings (#, ##, ###, ####)
    if line.startswith("#"):
        return None

    # --------------------------
    # Remove URLs
    if "http://" in line or "https://" in line or "www." in line:
        return None

    # --------------------------
    # Remove reference-only lines like [1]
    if re.match(r"^\[\d+\]$", line):
        return None

    # --------------------------
    # Remove email lines
    if "@" in line:
        return None

    # --------------------------
    # Remove very short junk (ONLY extreme case)
    if len(line.split()) <= 2:
        return None

    # --------------------------
    # CLEAN (not remove) Table mentions
    # Example: "Table 5.2: ..." → removed prefix only
    line = re.sub(r"Table\s+\d+[:\.]?\s*", "", line, flags=re.IGNORECASE)

    return line


# ==================================================
# ✅ STEP 3: MAIN CLEANING FUNCTION
# ==================================================
def clean_text(text: str):

    # Step 1 → remove blocks
    text = remove_blocks(text)

    # Step 2 → line cleaning
    lines = text.split("\n")

    clean_lines = []

    for line in lines:
        cleaned = clean_line(line)
        if cleaned:
            clean_lines.append(cleaned)

    # Step 3 → join clean text
    cleaned_text = "\n".join(clean_lines)

    return cleaned_text


# ==================================================
# ✅ MAIN
# ==================================================
def main():

    input_file = Path("/home/faujdar/onto_gen/data/interim/combined_corpus_raw.txt")
    output_file = Path("/home/faujdar/onto_gen/data/interim/cleaned_corpus.txt")

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found")

    print("🔄 Cleaning corpus (balanced mode)...")

    text = input_file.read_text(encoding="utf-8")

    cleaned = clean_text(text)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(cleaned, encoding="utf-8")

    print("\n✅ CLEANING COMPLETE")
    print(f"📁 Saved to: {output_file}")

    print("\n🔍 Sample cleaned output preview:\n")
    print(cleaned[:1000])   # preview first 1000 chars


if __name__ == "__main__":
    main()