from pathlib import Path
import pandas as pd
import spacy


# ==================================================
# Load spaCy model
# ==================================================
nlp = spacy.load("en_core_web_sm")


# ==================================================
# Extract triplets from one sentence
# ==================================================
def extract_triplets_from_sentence(doc):
    triplets = []

    for token in doc:
        # Find verbs
        if token.pos_ == "VERB":

            subject = None
            obj = None

            # --- Find subject
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = subtree_text(child)

            # --- Find object
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    obj = subtree_text(child)

                # prepositional object
                if child.dep_ == "prep":
                    for subchild in child.children:
                        if subchild.dep_ == "pobj":
                            obj = subtree_text(subchild)

            # Save triplet (only if valid)
            if subject and obj:
                triplets.append((
                    subject.strip(),
                    token.lemma_.strip(),
                    obj.strip()
                ))

    return triplets


# ==================================================
# Get full phrase from subtree
# ==================================================
def subtree_text(token):
    return " ".join([t.text for t in token.subtree])


# ==================================================
# Process full document
# ==================================================
def extract_triplets(text):
    doc = nlp(text)

    all_triplets = []

    for sent in doc.sents:
        triplets = extract_triplets_from_sentence(sent)
        all_triplets.extend(triplets)

    return all_triplets


# ==================================================
# MAIN
# ==================================================
def main():

    input_file = Path("data/interim/cleaned_corpus.txt")
    output_file = Path("data/processed/triplets/triplets_spacy.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found")

    print("🔄 Processing text...")

    text = input_file.read_text(encoding="utf-8")

    triplets = extract_triplets(text)

    # ✅ Remove duplicates
    triplets = list(set(triplets))

    # ✅ Convert to DataFrame (NO sentence column)
    df = pd.DataFrame(triplets, columns=["subject", "relation", "object"])

    # ✅ Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Extracted {len(df)} triplets")
    print(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()
