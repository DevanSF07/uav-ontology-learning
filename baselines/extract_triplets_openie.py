from pathlib import Path
import pandas as pd

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


# ==================================================
# Load OpenIE model
# ==================================================
print("Loading OpenIE model (first run will download model)...")
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
)


# ==================================================
# Extract triplets from one sentence
# ==================================================
def extract_from_sentence(sentence: str):
    results = predictor.predict(sentence=sentence)

    triplets = []

    for verb in results["verbs"]:
        description = verb["description"]

        # Example format:
        # "[ARG0: UAVs] [V: operate in] [ARG1: controlled airspace]"
        parts = description.split("]")

        subject = None
        relation = None
        obj = None

        for part in parts:
            if "ARG0:" in part:
                subject = part.split("ARG0:")[-1].strip()
            elif "V:" in part:
                relation = part.split("V:")[-1].strip()
            elif "ARG1:" in part:
                obj = part.split("ARG1:")[-1].strip()

        if subject and relation and obj:
            triplets.append((
                subject.strip(),
                relation.strip(),
                obj.strip()
            ))

    return triplets


# ==================================================
# Process full document
# ==================================================
def extract_triplets(text: str):

    triplets = []

    sentences = text.split("\n")

    for sent in sentences:
        sent = sent.strip()

        if len(sent.split()) < 4:
            continue

        try:
            triples = extract_from_sentence(sent)
            triplets.extend(triples)
        except Exception:
            continue

    return triplets


# ==================================================
# MAIN
# ==================================================
def main():

    input_file = Path("data/processed/interim/cleaned_corpus.txt")
    output_file = Path("data/processed/triplets/triplets_openie.csv")

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    print("🔄 Running AllenNLP OpenIE...")

    text = input_file.read_text(encoding="utf-8")

    triplets = extract_triplets(text)

    # ✅ Deduplicate
    triplets = list(set(triplets))

    # ✅ Save
    df = pd.DataFrame(triplets, columns=["subject", "relation", "object"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Extracted {len(df)} OpenIE triplets")
    print(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()