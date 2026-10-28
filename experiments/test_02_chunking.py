from pathlib import Path

import pandas as pd

from olbench.preprocess import chunk_text


PROJECT_DIR = Path(
    "ontology_experiment_framework"
)

INPUT_FILE = (
    PROJECT_DIR
    / "stage_outputs/01_cleaning/cleaned_corpus.txt"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "stage_outputs/02_chunking"
)

OUTPUT_FILE = OUTPUT_DIR / "chunks.csv"

MAX_WORDS = 220
OVERLAP_SENTENCES = 1


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            "Cleaned corpus not found.\n"
            f"Expected file: {INPUT_FILE}\n"
            "Run test_01_cleaning.py first."
        )

    cleaned_text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    if not cleaned_text.strip():
        raise ValueError(
            f"The cleaned corpus is empty: {INPUT_FILE}"
        )

    chunks = chunk_text(
        cleaned_text,
        max_words=MAX_WORDS,
        overlap_sentences=OVERLAP_SENTENCES
    )

    if chunks.empty:
        raise RuntimeError(
            "No chunks were produced. "
            "Inspect cleaned_corpus.txt and the chunking settings."
        )

    # Add useful validation and analysis columns.
    chunks["word_count"] = chunks["text"].apply(
        lambda text: len(str(text).split())
    )

    chunks["character_count"] = chunks["text"].apply(
        lambda text: len(str(text))
    )

    # Check that chunk identifiers are unique.
    duplicate_chunk_ids = int(
        chunks["chunk_id"].duplicated().sum()
    )

    if duplicate_chunk_ids > 0:
        raise ValueError(
            f"Found {duplicate_chunk_ids} duplicate chunk IDs."
        )

    # Check that the indexes are sequential.
    expected_indexes = list(range(len(chunks)))
    actual_indexes = chunks["chunk_index"].tolist()

    if actual_indexes != expected_indexes:
        raise ValueError(
            "Chunk indexes are not sequential."
        )

    # Save all chunks.
    chunks.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # Save a summary report.
    summary = pd.DataFrame(
        [
            {
                "input_file": str(INPUT_FILE),
                "output_file": str(OUTPUT_FILE),
                "maximum_target_words": MAX_WORDS,
                "overlap_sentences": OVERLAP_SENTENCES,
                "chunk_count": len(chunks),
                "minimum_words": int(
                    chunks["word_count"].min()
                ),
                "maximum_words": int(
                    chunks["word_count"].max()
                ),
                "average_words": round(
                    float(chunks["word_count"].mean()),
                    2
                ),
                "minimum_sentences": int(
                    chunks["sentence_count"].min()
                ),
                "maximum_sentences": int(
                    chunks["sentence_count"].max()
                ),
                "duplicate_chunk_ids": duplicate_chunk_ids
            }
        ]
    )

    summary_file = OUTPUT_DIR / "chunking_report.csv"

    summary.to_csv(
        summary_file,
        index=False,
        encoding="utf-8"
    )

    print("Chunking completed successfully")
    print()
    print(f"Input file: {INPUT_FILE}")
    print(f"Chunks file: {OUTPUT_FILE}")
    print(f"Summary file: {summary_file}")
    print()
    print(f"Number of chunks: {len(chunks)}")
    print(f"Target maximum words: {MAX_WORDS}")
    print(f"Sentence overlap: {OVERLAP_SENTENCES}")
    print()
    print("Word-count statistics:")
    print(chunks["word_count"].describe().to_string())

    print()
    print("First five chunk records:")
    print(
        chunks[
            [
                "chunk_id",
                "chunk_index",
                "section",
                "sentence_count",
                "word_count",
                "character_count"
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print()
    print("First chunk:")
    print("-" * 80)
    print(chunks.iloc[0]["text"])
    print("-" * 80)

    if len(chunks) > 1:
        print()
        print("Boundary check between chunks 0 and 1:")
        print()
        print("End of chunk 0:")
        print(str(chunks.iloc[0]["text"])[-300:])
        print()
        print("Start of chunk 1:")
        print(str(chunks.iloc[1]["text"])[:300])


if __name__ == "__main__":
    main()