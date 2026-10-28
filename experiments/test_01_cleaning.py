from pathlib import Path

from olbench.preprocess import clean_mmd


INPUT_FILE = Path("ontology_experiment_framework/data/combined_corpus_raw.txt")
OUTPUT_DIR = Path("ontology_experiment_framework/stage_outputs/01_cleaning")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    cleaned_text, cleaning_report = clean_mmd(INPUT_FILE)

    cleaned_file = OUTPUT_DIR / "cleaned_corpus.txt"
    report_file = OUTPUT_DIR / "preprocess_report.csv"

    cleaned_file.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    cleaning_report.to_csv(
        report_file,
        index=False
    )

    print("Cleaning completed")
    print(f"Input file: {INPUT_FILE.resolve()}")
    print(f"Cleaned corpus: {cleaned_file.resolve()}")
    print(f"Cleaning report: {report_file.resolve()}")
    print()
    print("Cleaning report:")
    print(cleaning_report.to_string(index=False))
    print()
    print("First 1,000 characters of cleaned text:")
    print("-" * 80)
    print(cleaned_text[:1000])
    print("-" * 80)


if __name__ == "__main__":
    main()