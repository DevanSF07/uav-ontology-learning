#!/usr/bin/env python3
"""
SAE sentence concept discovery with YAKE + KeyBERT(MPNet).

Pipeline:
  cleaned TXT corpus
    -> sentence splitting + basic artifact filtering
    -> MPNet sentence embeddings
    -> sparse autoencoder
    -> top-activating sentences per sparse feature
    -> YAKE and KeyBERT multiword phrase extraction
    -> CSV outputs

Outputs:
  <prefix>_features.csv          one row per retained SAE feature
  <prefix>_feature_sentences.csv one row per feature/sentence activation
  <prefix>_feature_phrases.csv   one row per feature/phrase with component scores
  <prefix>_training_history.csv  epoch-level losses
  <prefix>_sae_model.pt          trained SAE checkpoint
  <prefix>_config.json           run configuration
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
import yake


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# -----------------------------
# Text preparation
# -----------------------------

ARTIFACT_PATTERNS = [
    r"\bdoi\b",
    r"\be-?mail\b",
    r"\bcopyright\b",
    r"springer science",
    r"\bprofessor emeritus\b",
    r"\buniversity of\b",
    r"\bschool of\b.*\bengineering\b",
    r"\bdepartment of\b.*\bengineering\b",
    r"@\w+\.",
    r"\(eds?\.\)",
    r"missingpagefail",
]


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"\[MISSINGPAGEFAIL:\d+\]", " ", text, flags=re.I)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_artifact_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(re.search(pattern, lowered, flags=re.I) for pattern in ARTIFACT_PATTERNS)


def split_sentences(
    text: str,
    min_chars: int = 35,
    max_chars: int = 1000,
    remove_artifacts: bool = True,
) -> List[str]:
    """Rule-based sentence splitter that preserves technical acronyms reasonably well."""
    text = normalize_text(text)

    # Protect common abbreviations before splitting.
    protected = {
        "e.g.": "eg<prd>",
        "i.e.": "ie<prd>",
        "et al.": "et al<prd>",
        "U.S.": "US<prd>",
        "U.K.": "UK<prd>",
        "Fig.": "Fig<prd>",
        "Eq.": "Eq<prd>",
        "No.": "No<prd>",
    }
    for src, dst in protected.items():
        text = text.replace(src, dst)

    # Newlines often mark headings/list boundaries in cleaned technical text.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)

    sentences: List[str] = []
    seen = set()
    for part in parts:
        for src, dst in protected.items():
            part = part.replace(dst, src)
        sentence = re.sub(r"\s+", " ", part).strip(" -\t\r\n")
        if not sentence:
            continue
        if len(sentence) < min_chars or len(sentence) > max_chars:
            continue
        if remove_artifacts and is_artifact_sentence(sentence):
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
    return sentences


# -----------------------------
# Sparse autoencoder
# -----------------------------

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


def train_sae(
    embeddings: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sparse_lambda: float,
    weight_decay: float,
    device: str,
) -> Tuple[SparseAutoencoder, pd.DataFrame]:
    x_tensor = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=min(batch_size, len(x_tensor)),
        shuffle=True,
    )

    model = SparseAutoencoder(embeddings.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    mse = nn.MSELoss()
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_sparsity = 0.0
        sample_count = 0

        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, latent = model(batch)
            reconstruction_loss = mse(reconstruction, batch)
            sparsity_loss = latent.abs().mean()
            loss = reconstruction_loss + sparse_lambda * sparsity_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = batch.shape[0]
            sample_count += n
            total_loss += float(loss.detach().cpu()) * n
            total_reconstruction += float(reconstruction_loss.detach().cpu()) * n
            total_sparsity += float(sparsity_loss.detach().cpu()) * n

        row = {
            "epoch": epoch,
            "total_loss": total_loss / sample_count,
            "reconstruction_loss": total_reconstruction / sample_count,
            "mean_absolute_activation": total_sparsity / sample_count,
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss={row['total_loss']:.6f} | "
                f"recon={row['reconstruction_loss']:.6f} | "
                f"activation={row['mean_absolute_activation']:.6f}"
            )

    return model, pd.DataFrame(history)


def get_latent_activations(
    model: SparseAutoencoder,
    embeddings: np.ndarray,
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        return model.encode(tensor).cpu().numpy()


# -----------------------------
# YAKE + KeyBERT phrase extraction
# -----------------------------

GENERIC_PHRASES = {
    "et al",
    "presented table",
    "shown table",
    "outlined presented",
    "simple classification",
    "different types",
}


def normalize_phrase(phrase: str) -> str:
    phrase = phrase.lower().strip()
    phrase = re.sub(r"[^a-z0-9/+\-&() ]+", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip(" -")
    return phrase


def valid_phrase(phrase: str, min_words: int, max_words: int) -> bool:
    words = phrase.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if len(phrase) < 4 or phrase in GENERIC_PHRASES:
        return False
    if all(re.fullmatch(r"\d+(?:\.\d+)?", word) for word in words):
        return False
    if len(set(words)) == 1:
        return False
    return True


def phrase_sentence_frequency(phrase: str, sentences: Sequence[str]) -> int:
    pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", flags=re.I)
    return sum(bool(pattern.search(sentence)) for sentence in sentences)


def extract_keyphrases(
    sentences: Sequence[str],
    keybert_model: KeyBERT,
    top_n: int,
    candidate_pool: int,
    min_words: int,
    max_words: int,
    yake_weight: float,
    keybert_weight: float,
    frequency_weight: float,
    keybert_diversity: float,
) -> List[Dict[str, float]]:
    feature_text = " ".join(sentences)
    combined: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"yake": 0.0, "keybert": 0.0}
    )

    # YAKE lower raw score is better. Convert selected candidates into a bounded rank score.
    yake_extractor = yake.KeywordExtractor(
        lan="en",
        n=max_words,
        dedupLim=0.85,
        top=candidate_pool,
    )
    raw_yake = yake_extractor.extract_keywords(feature_text)
    cleaned_yake = []
    for phrase, raw_score in raw_yake:
        phrase = normalize_phrase(phrase)
        if valid_phrase(phrase, min_words, max_words):
            cleaned_yake.append((phrase, float(raw_score)))
    for rank, (phrase, _raw_score) in enumerate(cleaned_yake, start=1):
        rank_score = 1.0 / rank
        combined[phrase]["yake"] = max(combined[phrase]["yake"], rank_score)

    # KeyBERT uses the same MPNet family as sentence embedding.
    raw_keybert = keybert_model.extract_keywords(
        feature_text,
        keyphrase_ngram_range=(min_words, max_words),
        stop_words="english",
        use_mmr=True,
        diversity=keybert_diversity,
        top_n=candidate_pool,
    )
    for phrase, score in raw_keybert:
        phrase = normalize_phrase(phrase)
        if valid_phrase(phrase, min_words, max_words):
            combined[phrase]["keybert"] = max(
                combined[phrase]["keybert"], float(score)
            )

    rows: List[Dict[str, float]] = []
    sentence_count = max(1, len(sentences))
    for phrase, scores in combined.items():
        frequency = phrase_sentence_frequency(phrase, sentences)
        frequency_ratio = frequency / sentence_count
        agreement = int(scores["yake"] > 0 and scores["keybert"] > 0)
        combined_score = (
            yake_weight * scores["yake"]
            + keybert_weight * scores["keybert"]
            + frequency_weight * frequency_ratio
            + 0.10 * agreement
        )
        rows.append(
            {
                "phrase": phrase,
                "combined_score": combined_score,
                "yake_rank_score": scores["yake"],
                "keybert_score": scores["keybert"],
                "sentence_frequency": frequency,
                "extractor_agreement": agreement,
            }
        )

    rows.sort(
        key=lambda row: (
            row["combined_score"],
            row["extractor_agreement"],
            row["sentence_frequency"],
            len(row["phrase"].split()),
        ),
        reverse=True,
    )
    return rows[:top_n]


# -----------------------------
# Feature analysis
# -----------------------------

def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return float(upper.mean()) if len(upper) else 1.0


def analyze_features(
    sentences: Sequence[str],
    embeddings: np.ndarray,
    latent: np.ndarray,
    keybert_model: KeyBERT,
    top_k_sentences: int,
    top_n_phrases: int,
    candidate_pool: int,
    min_phrase_words: int,
    max_phrase_words: int,
    min_mean_activation: float,
    min_max_activation: float,
    yake_weight: float,
    keybert_weight: float,
    frequency_weight: float,
    keybert_diversity: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    sentence_rows = []
    phrase_rows = []

    for feature_id in range(latent.shape[1]):
        activations = latent[:, feature_id]
        mean_activation = float(activations.mean())
        max_activation = float(activations.max())
        nonzero_count = int(np.count_nonzero(activations > 0))
        nonzero_ratio = nonzero_count / len(activations)

        if mean_activation < min_mean_activation or max_activation < min_max_activation:
            continue

        top_indices = np.argsort(activations)[::-1][: min(top_k_sentences, len(sentences))]
        top_indices = [int(i) for i in top_indices if activations[i] > 0]
        if not top_indices:
            continue

        top_sentences = [sentences[i] for i in top_indices]
        top_activations = [float(activations[i]) for i in top_indices]
        top_mean_activation = float(np.mean(top_activations))
        activation_gap = top_mean_activation - mean_activation
        sentence_coherence = mean_pairwise_cosine(embeddings[top_indices])

        phrases = extract_keyphrases(
            sentences=top_sentences,
            keybert_model=keybert_model,
            top_n=top_n_phrases,
            candidate_pool=candidate_pool,
            min_words=min_phrase_words,
            max_words=max_phrase_words,
            yake_weight=yake_weight,
            keybert_weight=keybert_weight,
            frequency_weight=frequency_weight,
            keybert_diversity=keybert_diversity,
        )

        feature_row = {
            "feature_id": feature_id,
            "mean_activation": mean_activation,
            "max_activation": max_activation,
            "top_mean_activation": top_mean_activation,
            "activation_gap": activation_gap,
            "nonzero_sentence_count": nonzero_count,
            "nonzero_sentence_ratio": nonzero_ratio,
            "top_sentence_coherence": sentence_coherence,
            "top_phrases": " | ".join(row["phrase"] for row in phrases),
        }
        for rank, (sentence_index, activation) in enumerate(
            zip(top_indices, top_activations), start=1
        ):
            feature_row[f"sentence_{rank}"] = sentences[sentence_index]
            sentence_rows.append(
                {
                    "feature_id": feature_id,
                    "sentence_rank": rank,
                    "sentence_index": sentence_index,
                    "activation": activation,
                    "sentence": sentences[sentence_index],
                }
            )

        for rank, row in enumerate(phrases, start=1):
            phrase_rows.append(
                {
                    "feature_id": feature_id,
                    "phrase_rank": rank,
                    **row,
                }
            )

        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    sentences_df = pd.DataFrame(sentence_rows)
    phrases_df = pd.DataFrame(phrase_rows)

    if not features_df.empty:
        features_df = features_df.sort_values(
            ["top_sentence_coherence", "activation_gap", "top_mean_activation"],
            ascending=False,
        )
    if not sentences_df.empty:
        sentences_df = sentences_df.sort_values(["feature_id", "sentence_rank"])
    if not phrases_df.empty:
        phrases_df = phrases_df.sort_values(["feature_id", "phrase_rank"])

    return features_df, sentences_df, phrases_df


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover sparse sentence features and extract multiword UAV phrases."
    )
    parser.add_argument("--input", required=True, help="Cleaned UTF-8 TXT corpus")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    parser.add_argument("--prefix", default="sae", help="Output filename prefix")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path",
    )
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sparse_lambda", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--top_k_sentences", type=int, default=20)
    parser.add_argument("--top_n_phrases", type=int, default=25)
    parser.add_argument("--candidate_pool", type=int, default=75)
    parser.add_argument("--min_phrase_words", type=int, default=2)
    parser.add_argument("--max_phrase_words", type=int, default=5)
    parser.add_argument("--min_sentence_chars", type=int, default=35)
    parser.add_argument("--max_sentence_chars", type=int, default=1000)
    parser.add_argument("--min_mean_activation", type=float, default=0.01)
    parser.add_argument("--min_max_activation", type=float, default=0.05)
    parser.add_argument("--yake_weight", type=float, default=0.40)
    parser.add_argument("--keybert_weight", type=float, default=0.45)
    parser.add_argument("--frequency_weight", type=float, default=0.15)
    parser.add_argument("--keybert_diversity", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_artifacts",
        action="store_true",
        help="Do not remove likely bibliography/affiliation sentences",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(
        text,
        min_chars=args.min_sentence_chars,
        max_chars=args.max_sentence_chars,
        remove_artifacts=not args.keep_artifacts,
    )
    if len(sentences) < 10:
        raise ValueError(
            f"Only {len(sentences)} usable sentences remained. Check input or cleaning settings."
        )
    print(f"Usable unique sentences: {len(sentences)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading embedding model: {args.model}")
    embedding_model = SentenceTransformer(args.model, device=device)

    embeddings = embedding_model.encode(
        list(sentences),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"Embedding matrix: {embeddings.shape}")

    model, history_df = train_sae(
        embeddings=embeddings,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sparse_lambda=args.sparse_lambda,
        weight_decay=args.weight_decay,
        device=device,
    )
    latent = get_latent_activations(model, embeddings, device)
    print(f"Latent activation matrix: {latent.shape}")

    # Reuse the already loaded MPNet model inside KeyBERT.
    keybert_model = KeyBERT(model=embedding_model)

    features_df, feature_sentences_df, feature_phrases_df = analyze_features(
        sentences=sentences,
        embeddings=embeddings,
        latent=latent,
        keybert_model=keybert_model,
        top_k_sentences=args.top_k_sentences,
        top_n_phrases=args.top_n_phrases,
        candidate_pool=args.candidate_pool,
        min_phrase_words=args.min_phrase_words,
        max_phrase_words=args.max_phrase_words,
        min_mean_activation=args.min_mean_activation,
        min_max_activation=args.min_max_activation,
        yake_weight=args.yake_weight,
        keybert_weight=args.keybert_weight,
        frequency_weight=args.frequency_weight,
        keybert_diversity=args.keybert_diversity,
    )

    paths = {
        "features": output_dir / f"{args.prefix}_features.csv",
        "sentences": output_dir / f"{args.prefix}_feature_sentences.csv",
        "phrases": output_dir / f"{args.prefix}_feature_phrases.csv",
        "history": output_dir / f"{args.prefix}_training_history.csv",
        "model": output_dir / f"{args.prefix}_sae_model.pt",
        "config": output_dir / f"{args.prefix}_config.json",
    }

    features_df.to_csv(paths["features"], index=False)
    feature_sentences_df.to_csv(paths["sentences"], index=False)
    feature_phrases_df.to_csv(paths["phrases"], index=False)
    history_df.to_csv(paths["history"], index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(embeddings.shape[1]),
            "latent_dim": args.latent_dim,
            "embedding_model": args.model,
        },
        paths["model"],
    )
    paths["config"].write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Retained features: {len(features_df)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
