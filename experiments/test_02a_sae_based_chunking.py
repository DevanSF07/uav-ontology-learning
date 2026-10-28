from __future__ import annotations

import argparse
import json
import logging
import re
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from olbench.utils import text_sentences


LOGGER = logging.getLogger("test_02a_sae_based_chunking")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_STAGE_CHUNKS = REPO_ROOT / "ontology_experiment_framework" / "stage_outputs" / "02_chunking" / "chunks.csv"

# ---------------------------------------------------------------------------
# Editable configuration
# ---------------------------------------------------------------------------

INPUT_CSV = "ontology_experiment_framework/stage_outputs/01_cleaning/cleaned_corpus.txt"
OUTPUT_DIR = "ontology_experiment_framework/stage_outputs/02a_sae_based_chunking"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
HIDDEN_MULTIPLIER = 8
SPARSITY_METHOD = "topk"
TOP_K_VALUES = [8, 16, 32, 64]
TOP_CHUNKS_PER_FEATURE = 50
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
USE_GPU = True
CONSOLIDATED_CHUNK_MAX_WORDS = 220

CHUNK_KEY_COLUMNS = ["chunk_id", "text"]
ACTIVATION_COLUMNS = ["chunk_id", "feature_id", "activation_score"]
SUMMARY_COLUMNS = ["feature_id", "activation_count", "mean_activation", "max_activation", "top_chunk_count"]
REPORT_COLUMNS = ["K", "reconstruction_loss", "num_features_used", "avg_active_features_per_chunk"]


@dataclass(frozen=True)
class SAEConfig:
    input_csv: str | Path = INPUT_CSV
    output_dir: str | Path = OUTPUT_DIR
    embedding_model: str | Path = EMBEDDING_MODEL
    hidden_multiplier: int = HIDDEN_MULTIPLIER
    sparsity_method: str = SPARSITY_METHOD
    top_k_values: tuple[int, ...] = tuple(TOP_K_VALUES)
    top_chunks_per_feature: int = TOP_CHUNKS_PER_FEATURE
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    random_seed: int = RANDOM_SEED
    use_gpu: bool = USE_GPU


@dataclass(frozen=True)
class KExperimentResult:
    k: int
    reconstruction_loss: float
    num_features_used: int
    avg_active_features_per_chunk: float
    output_dir: Path
    hidden_dim: int


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def parse_top_k_values(raw_value: str | None) -> tuple[int, ...]:
    if not raw_value:
        return tuple(TOP_K_VALUES)
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError("TOP_K_VALUES cannot be empty.")
    if any(value <= 0 for value in parsed):
        raise ValueError("TOP_K_VALUES must contain positive integers.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover latent semantic SAE features and export top-activating chunks per feature."
    )
    parser.add_argument("--input-csv", type=Path, help="Input cleaned text file or legacy chunks CSV.")
    parser.add_argument("--output-dir", type=Path, help="Directory for SAE outputs.")
    parser.add_argument("--embedding-model", type=str, help="SentenceTransformer model name or local path.")
    parser.add_argument("--hidden-multiplier", type=int, help="Hidden dimension multiplier.")
    parser.add_argument("--top-k-values", type=str, help="Comma-separated top-k values, e.g. 8,16,32,64.")
    parser.add_argument("--top-chunks-per-feature", type=int, help="Max top chunks to retain per feature.")
    parser.add_argument("--batch-size", type=int, help="Embedding and training batch size.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument("--random-seed", type=int, help="Random seed.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU if available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    return parser.parse_args()


def resolve_existing_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                SCRIPT_DIR / path,
                REPO_ROOT / path,
            ]
        )
        if path.name == "chunks.csv":
            candidates.append(DEFAULT_STAGE_CHUNKS)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if must_exist:
        raise FileNotFoundError(f"Input file not found: {path}")

    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_output_dir(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_embedding_model(value: str | Path) -> str:
    model_path = Path(value)
    if model_path.exists():
        return str(model_path.resolve())

    local_name = model_path.name
    local_candidate = REPO_ROOT / "models" / local_name
    if local_candidate.exists():
        return str(local_candidate.resolve())

    return str(value)


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_name(feature_index: int) -> str:
    return f"feature_{feature_index + 1:04d}"


def normalize_sentence_key(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text).casefold()).strip()


def load_cleaned_sentences(cleaned_text_file: Path) -> tuple[pd.DataFrame, str]:
    if not cleaned_text_file.is_file():
        raise FileNotFoundError(f"Cleaned text file not found: {cleaned_text_file}")

    raw_text = cleaned_text_file.read_text(encoding="utf-8")
    sentences = [sentence.strip() for sentence in text_sentences(raw_text) if sentence.strip()]
    if not sentences:
        raise ValueError(f"No usable sentences were found in {cleaned_text_file}")

    rows = [
        {
            "sentence_id": f"sentence_{index + 1:06d}",
            "sentence_index": index,
            "text": sentence,
        }
        for index, sentence in enumerate(sentences)
    ]
    return pd.DataFrame(rows, columns=["sentence_id", "sentence_index", "text"]), "sentence_id"


def load_chunks(input_path: Path) -> tuple[pd.DataFrame, str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in {".txt", ".mmd", ".md"}:
        work, record_id_column = load_cleaned_sentences(input_path)
        LOGGER.info("Loaded %d cleaned sentences from %s", len(work), input_path)
        return work.reset_index(drop=True), record_id_column

    chunks = pd.read_csv(input_path, encoding="utf-8")
    missing_columns = set(CHUNK_KEY_COLUMNS) - set(chunks.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing_columns)}")

    work = chunks.copy()
    chunk_ids = work["chunk_id"].map(safe_text)
    texts = work["text"].map(safe_text)
    valid_mask = chunk_ids.ne("") & texts.ne("")
    work = work[valid_mask].copy()
    work["text"] = texts[valid_mask].str.replace(r"\s+", " ", regex=True).str.strip()
    work["chunk_id"] = chunks.loc[work.index, "chunk_id"].values

    if work.empty:
        raise ValueError("No valid chunk rows remained after validation.")

    duplicate_chunk_ids = int(chunk_ids[valid_mask].duplicated().sum())
    if duplicate_chunk_ids:
        raise ValueError(f"Found {duplicate_chunk_ids} duplicate chunk IDs.")

    LOGGER.info("Loaded %d valid chunks from %s", len(work), input_path)
    return work.reset_index(drop=True), "chunk_id"


def generate_embeddings(
    chunks: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    texts = chunks["text"].astype(str).tolist()
    LOGGER.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, device=str(device))

    LOGGER.info("Encoding %d chunks into normalized embeddings", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=max(1, batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_embeddings_artifact(
    output_dir: Path,
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    record_id_column: str,
) -> Path:
    artifact_path = output_dir / "chunk_embeddings.npz"
    np.savez_compressed(
        artifact_path,
        chunk_ids=chunks[record_id_column].astype(str).to_numpy(),
        embeddings=embeddings,
    )
    return artifact_path


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.top_k = max(1, min(int(top_k), self.hidden_dim))
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)

    def sparse_encode(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = F.relu(self.encoder(inputs))
        if self.top_k >= activations.shape[-1]:
            return activations

        values, indices = torch.topk(activations, self.top_k, dim=-1)
        sparse = torch.zeros_like(activations)
        sparse.scatter_(1, indices, values)
        return sparse

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.sparse_encode(inputs)
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden


def make_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_sae(
    embeddings: np.ndarray,
    hidden_dim: int,
    top_k: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[TopKSparseAutoencoder, pd.DataFrame]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")

    dataset = TensorDataset(torch.from_numpy(embeddings.astype(np.float32)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = TopKSparseAutoencoder(embeddings.shape[1], hidden_dim, top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history_rows: list[dict[str, float]] = []
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        sample_count = 0

        progress = tqdm(loader, desc=f"train K={top_k} epoch {epoch}/{epochs}", leave=False)
        for (batch,) in progress:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = int(batch.shape[0])
            epoch_loss += float(loss.detach().cpu()) * batch_size_actual
            sample_count += batch_size_actual
            batch_count += 1
            progress.set_postfix(reconstruction_loss=float(loss.detach().cpu()))

        mean_loss = epoch_loss / max(1, sample_count)
        history_rows.append(
            {
                "epoch": float(epoch),
                "reconstruction_loss": mean_loss,
                "batch_count": float(batch_count),
                "sample_count": float(sample_count),
            }
        )
        LOGGER.info("K=%d epoch %d/%d reconstruction_loss=%.6f", top_k, epoch, epochs, mean_loss)

    evaluation_loss = evaluate_reconstruction_loss(model, embeddings, batch_size, device)
    return model.eval(), pd.DataFrame(history_rows), evaluation_loss


def evaluate_reconstruction_loss(
    model: TopKSparseAutoencoder,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(embeddings.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, pin_memory=device.type == "cuda")

    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="evaluate reconstruction", leave=False):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch, reduction="sum")
            batch_size_actual = int(batch.shape[0])
            total_loss += float(loss.detach().cpu())
            total_items += batch_size_actual

    return total_loss / max(1, total_items)


def collect_sparse_activations(
    model: TopKSparseAutoencoder,
    embeddings: np.ndarray,
    record_ids: pd.Series,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    dataset = TensorDataset(torch.from_numpy(embeddings.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, pin_memory=device.type == "cuda")

    offset = 0
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="collect activations", leave=False):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            _, sparse_hidden = model(batch)
            dense_hidden = sparse_hidden.detach().cpu().numpy()

            for local_index, activation_vector in enumerate(dense_hidden):
                chunk_id = str(record_ids.iloc[offset + local_index])
                active_feature_indices = np.flatnonzero(activation_vector > 0)
                for feature_index in active_feature_indices:
                    rows.append(
                        {
                            "chunk_id": chunk_id,
                            "feature_index": int(feature_index),
                            "feature_id": feature_name(feature_index),
                            "activation_score": float(activation_vector[feature_index]),
                        }
                    )

            offset += len(dense_hidden)

    activations = pd.DataFrame(rows, columns=ACTIVATION_COLUMNS + ["feature_index"])
    if activations.empty:
        return pd.DataFrame(columns=ACTIVATION_COLUMNS + ["feature_index"])

    LOGGER.info("Collected %d positive chunk-feature activations", len(activations))
    return activations[["chunk_id", "feature_index", "feature_id", "activation_score"]]


def build_feature_outputs(
    chunks: pd.DataFrame,
    activations: pd.DataFrame,
    hidden_dim: int,
    top_chunks_per_feature: int,
    record_id_column: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    chunk_lookup = {
        safe_text(row[record_id_column]): row.to_dict()
        for _, row in chunks.iterrows()
    }
    chunk_columns = list(chunks.columns)

    summary_rows: list[dict[str, object]] = []
    consolidated_rows: list[dict[str, object]] = []
    overlap_map: dict[str, list[tuple[str, float]]] = {}
    feature_catalog: dict[str, dict[str, Any]] = {}

    for feature_index in range(hidden_dim):
        fid = feature_name(feature_index)
        if activations.empty:
            feature_rows = pd.DataFrame(columns=["chunk_id", "feature_index", "feature_id", "activation_score"])
        else:
            feature_rows = activations[activations["feature_index"] == feature_index].copy()

        feature_rows = feature_rows.sort_values(
            by=["activation_score", "chunk_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        activation_count = int(len(feature_rows))
        mean_activation = float(feature_rows["activation_score"].mean()) if activation_count else 0.0
        max_activation = float(feature_rows["activation_score"].max()) if activation_count else 0.0

        top_rows = feature_rows.head(max(0, top_chunks_per_feature)).copy().reset_index(drop=True)
        if not top_rows.empty:
            top_rows.insert(0, "rank", range(1, len(top_rows) + 1))

        for row in top_rows.itertuples(index=False):
            chunk_row = chunk_lookup.get(safe_text(row.chunk_id))
            if not chunk_row:
                continue
            consolidated_row: dict[str, object] = {
                "feature_id": fid,
                "rank": int(row.rank),
                "activation_score": float(row.activation_score),
            }
            for column in chunk_columns:
                consolidated_row[column] = chunk_row.get(column, "")
            consolidated_rows.append(consolidated_row)
            overlap_map.setdefault(safe_text(row.chunk_id), []).append((fid, float(row.activation_score)))

        summary_rows.append(
            {
                "feature_id": fid,
                "activation_count": activation_count,
                "mean_activation": mean_activation,
                "max_activation": max_activation,
                "top_chunk_count": int(len(top_rows)),
            }
        )

        feature_catalog[fid] = {
            "activation_count": activation_count,
            "mean_activation": mean_activation,
            "max_activation": max_activation,
            "top_chunks": top_rows.to_dict(orient="records"),
        }

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(
            by=["feature_id"],
            ascending=[True],
        ).reset_index(drop=True)

    feature_chunks_columns = ["feature_id", "rank", "activation_score", *chunk_columns]
    feature_chunks = pd.DataFrame(consolidated_rows, columns=feature_chunks_columns)
    if not feature_chunks.empty:
        feature_chunks = feature_chunks.sort_values(
            by=["feature_id", "rank", "activation_score"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    overlap_rows: list[dict[str, object]] = []
    for chunk_id, feature_items in overlap_map.items():
        if len(feature_items) < 2:
            continue
        ordered = sorted(feature_items, key=lambda item: (-item[1], item[0]))
        overlap_rows.append(
            {
                "chunk_id": chunk_id,
                "num_features": len(ordered),
                "feature_ids": "|".join(feature_id for feature_id, _ in ordered),
                "activation_scores": "|".join(f"{score:.6f}" for _, score in ordered),
            }
        )

    overlap_df = pd.DataFrame(overlap_rows, columns=["chunk_id", "num_features", "feature_ids", "activation_scores"])
    if not overlap_df.empty:
        overlap_df = overlap_df.sort_values(
            by=["num_features", "chunk_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return summary, feature_catalog, feature_chunks, overlap_df


def summarize_run(
    k: int,
    reconstruction_loss: float,
    activations: pd.DataFrame,
) -> tuple[int, float, int, float]:
    used_features = int(activations["feature_index"].nunique()) if not activations.empty else 0
    if activations.empty:
        avg_active_features = 0.0
    else:
        avg_active_features = float(activations.groupby("chunk_id")["feature_index"].nunique().mean())
    return k, reconstruction_loss, used_features, avg_active_features


def recommend_k(report_df: pd.DataFrame) -> int:
    if report_df.empty:
        return int(TOP_K_VALUES[0])

    best_loss = float(report_df["reconstruction_loss"].min())
    threshold = best_loss * 1.15
    eligible = report_df[report_df["reconstruction_loss"] <= threshold].copy()
    if eligible.empty:
        eligible = report_df.copy()

    eligible = eligible.sort_values(
        by=["K", "reconstruction_loss", "avg_active_features_per_chunk"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return int(eligible.iloc[0]["K"])


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def copy_source_chunks(source_csv: Path, destination_csv: Path) -> None:
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_csv, destination_csv)


def consolidate_feature_chunks(
    feature_chunks: pd.DataFrame,
    max_words: int = CONSOLIDATED_CHUNK_MAX_WORDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_id_column = (
        "chunk_id"
        if "chunk_id" in feature_chunks.columns
        else "sentence_id"
        if "sentence_id" in feature_chunks.columns
        else feature_chunks.columns[0]
    )

    if feature_chunks.empty:
        empty_chunks = pd.DataFrame(
            columns=["chunk_id", "chunk_index", "section", "text", "sentence_count", "word_count", "character_count"]
        )
        empty_overlap = pd.DataFrame(
            columns=["sentence", "num_features", "feature_ids", "activation_scores", "source_record_ids"]
        )
        return empty_chunks, empty_overlap

    consolidated_rows: list[dict[str, object]] = []
    sentence_overlap_map: dict[str, dict[str, Any]] = {}
    global_chunk_index = 0

    for feature_id, feature_group in feature_chunks.groupby("feature_id", sort=True):
        sort_columns = ["rank", "activation_score", record_id_column]
        sort_ascending = [True, False, True]
        if "chunk_index" in feature_group.columns:
            sort_columns = ["rank", "activation_score", "chunk_index", record_id_column]
            sort_ascending = [True, False, True, True]
        elif "sentence_index" in feature_group.columns:
            sort_columns = ["rank", "activation_score", "sentence_index", record_id_column]
            sort_ascending = [True, False, True, True]

        ordered_group = feature_group.sort_values(
            by=sort_columns,
            ascending=sort_ascending,
        ).reset_index(drop=True)

        sentence_records: list[dict[str, object]] = []
        for row in ordered_group.itertuples(index=False):
            source_text = safe_text(row.text)
            source_sentences = [sentence.strip() for sentence in text_sentences(source_text) if sentence.strip()]
            if not source_sentences and source_text:
                source_sentences = [source_text]

            for sentence in source_sentences:
                source_record_id = safe_text(getattr(row, record_id_column))
                sentence_records.append(
                    {
                        "sentence": sentence,
                        "sentence_key": normalize_sentence_key(sentence),
                        "source_record_id": source_record_id,
                        "feature_id": feature_id,
                        "activation_score": float(row.activation_score),
                    }
                )

                entry = sentence_overlap_map.setdefault(
                    normalize_sentence_key(sentence),
                    {
                        "sentence": sentence,
                        "feature_ids": [],
                        "activation_scores": [],
                        "source_record_ids": [],
                    },
                )
                if feature_id not in entry["feature_ids"]:
                    entry["feature_ids"].append(feature_id)
                    entry["activation_scores"].append(float(row.activation_score))
                if source_record_id not in entry["source_record_ids"]:
                    entry["source_record_ids"].append(source_record_id)

        buffer_sentences: list[str] = []
        buffer_word_count = 0
        part_index = 1

        def flush_buffer() -> None:
            nonlocal global_chunk_index, buffer_sentences, buffer_word_count, part_index
            if not buffer_sentences:
                return

            text = " ".join(buffer_sentences).strip()
            if not text:
                buffer_sentences = []
                buffer_word_count = 0
                return

            section_name = feature_id if part_index == 1 else f"{feature_id}_part{part_index}"
            consolidated_rows.append(
                {
                    "chunk_id": f"{feature_id}_chunk_{part_index:03d}",
                    "chunk_index": global_chunk_index,
                    "section": section_name,
                    "text": text,
                    "sentence_count": int(len(buffer_sentences)),
                    "word_count": int(len(text.split())),
                    "character_count": int(len(text)),
                }
            )
            global_chunk_index += 1
            part_index += 1
            buffer_sentences = []
            buffer_word_count = 0

        for record in sentence_records:
            sentence = record["sentence"]
            sentence_word_count = len(sentence.split())
            if buffer_sentences and buffer_word_count + sentence_word_count > max_words:
                flush_buffer()

            if sentence_word_count > max_words and not buffer_sentences:
                consolidated_rows.append(
                    {
                        "chunk_id": f"{feature_id}_chunk_{part_index:03d}",
                        "chunk_index": global_chunk_index,
                        "section": feature_id if part_index == 1 else f"{feature_id}_part{part_index}",
                        "text": sentence,
                        "sentence_count": 1,
                        "word_count": sentence_word_count,
                        "character_count": int(len(sentence)),
                    }
                )
                global_chunk_index += 1
                part_index += 1
                continue

            buffer_sentences.append(sentence)
            buffer_word_count += sentence_word_count

        flush_buffer()

    consolidated_df = pd.DataFrame(
        consolidated_rows,
        columns=["chunk_id", "chunk_index", "section", "text", "sentence_count", "word_count", "character_count"],
    )
    if not consolidated_df.empty:
        consolidated_df = consolidated_df.sort_values(by=["chunk_index", "chunk_id"], ascending=[True, True]).reset_index(drop=True)

    overlap_rows: list[dict[str, object]] = []
    for entry in sentence_overlap_map.values():
        unique_features = entry["feature_ids"]
        if len(unique_features) < 2:
            continue
        ordered_pairs = sorted(zip(unique_features, entry["activation_scores"]), key=lambda item: (-item[1], item[0]))
        overlap_rows.append(
            {
                "sentence": entry["sentence"],
                "num_features": len(unique_features),
                "feature_ids": "|".join(feature_id for feature_id, _ in ordered_pairs),
                "activation_scores": "|".join(f"{score:.6f}" for _, score in ordered_pairs),
                "source_record_ids": "|".join(entry["source_record_ids"]),
            }
        )

    overlap_df = pd.DataFrame(
        overlap_rows,
        columns=["sentence", "num_features", "feature_ids", "activation_scores", "source_record_ids"],
    )
    if not overlap_df.empty:
        overlap_df = overlap_df.sort_values(by=["num_features", "sentence"], ascending=[False, True]).reset_index(drop=True)

    return consolidated_df, overlap_df


def run_single_k_experiment(
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    config: SAEConfig,
    k: int,
    output_dir: Path,
    device: torch.device,
    record_id_column: str,
) -> KExperimentResult:
    k_dir = ensure_directory(output_dir / f"K_{k}")

    hidden_dim = int(embeddings.shape[1] * config.hidden_multiplier)
    effective_k = max(1, min(int(k), hidden_dim))
    if effective_k != k:
        LOGGER.warning("Requested K=%d exceeded hidden_dim=%d. Using K=%d.", k, hidden_dim, effective_k)

    LOGGER.info(
        "Running SAE experiment for K=%d with input_dim=%d hidden_dim=%d",
        effective_k,
        embeddings.shape[1],
        hidden_dim,
    )

    model, history_df, reconstruction_loss = train_sae(
        embeddings=embeddings,
        hidden_dim=hidden_dim,
        top_k=effective_k,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.random_seed,
        device=device,
    )

    checkpoint_path = k_dir / "sae_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": int(embeddings.shape[1]),
                "hidden_dim": hidden_dim,
                "top_k": effective_k,
                "hidden_multiplier": config.hidden_multiplier,
                "embedding_model": str(config.embedding_model),
                "random_seed": config.random_seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "sparsity_method": config.sparsity_method,
            },
        },
        checkpoint_path,
    )

    activations = collect_sparse_activations(
        model=model,
        embeddings=embeddings,
        record_ids=chunks[record_id_column],
        batch_size=config.batch_size,
        device=device,
    )

    activation_file = k_dir / "chunk_feature_activations.csv"
    activations.to_csv(activation_file, index=False, encoding="utf-8")

    summary_df, catalog, feature_chunks, overlap_df = build_feature_outputs(
        chunks=chunks,
        activations=activations,
        hidden_dim=hidden_dim,
        top_chunks_per_feature=config.top_chunks_per_feature,
        record_id_column=record_id_column,
    )

    summary_file = k_dir / "feature_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8")

    feature_chunks_file = k_dir / "feature_chunks.csv"
    feature_chunks.to_csv(feature_chunks_file, index=False, encoding="utf-8")

    overlap_file = k_dir / "chunk_feature_overlap.csv"
    overlap_df.to_csv(overlap_file, index=False, encoding="utf-8")

    catalog_file = k_dir / "feature_catalog.json"
    save_json(catalog_file, catalog)

    history_df.to_csv(k_dir / "training_history.csv", index=False, encoding="utf-8")

    metric_row = summarize_run(effective_k, reconstruction_loss, activations)
    LOGGER.info(
        "K=%d done | reconstruction_loss=%.6f | num_features_used=%d | avg_active_features_per_chunk=%.3f",
        metric_row[0],
        metric_row[1],
        metric_row[2],
        metric_row[3],
    )

    run_manifest = {
        "K": effective_k,
        "input_file": str(resolve_existing_path(config.input_csv)),
        "output_dir": str(k_dir),
        "embedding_model": str(config.embedding_model),
        "hidden_dim": hidden_dim,
        "top_chunks_per_feature": config.top_chunks_per_feature,
        "checkpoint_file": str(checkpoint_path),
        "activation_file": str(activation_file),
        "feature_summary_file": str(summary_file),
        "feature_chunks_file": str(feature_chunks_file),
        "chunk_feature_overlap_file": str(overlap_file),
        "feature_catalog_file": str(catalog_file),
    }
    save_json(k_dir / "run_manifest.json", run_manifest)

    return KExperimentResult(
        k=effective_k,
        reconstruction_loss=reconstruction_loss,
        num_features_used=metric_row[2],
        avg_active_features_per_chunk=metric_row[3],
        output_dir=k_dir,
        hidden_dim=hidden_dim,
    )


def run_experiment(config: SAEConfig) -> pd.DataFrame:
    set_random_seed(config.random_seed)
    device = make_device(config.use_gpu)
    LOGGER.info("Using device: %s", device)

    input_csv = resolve_existing_path(config.input_csv)
    output_dir = ensure_directory(resolve_output_dir(config.output_dir))
    embedding_model = resolve_embedding_model(config.embedding_model)

    chunks, record_id_column = load_chunks(input_csv)
    embeddings = generate_embeddings(
        chunks=chunks,
        model_name=embedding_model,
        batch_size=config.batch_size,
        device=device,
    )
    embeddings_file = save_embeddings_artifact(output_dir, chunks, embeddings, record_id_column)
    LOGGER.info("Saved embeddings artifact: %s", embeddings_file)

    run_rows: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []

    for k in config.top_k_values:
        result = run_single_k_experiment(
            chunks=chunks,
            embeddings=embeddings,
            config=SAEConfig(
                input_csv=input_csv,
                output_dir=output_dir,
                embedding_model=embedding_model,
                hidden_multiplier=config.hidden_multiplier,
                sparsity_method=config.sparsity_method,
                top_k_values=config.top_k_values,
                top_chunks_per_feature=config.top_chunks_per_feature,
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                random_seed=config.random_seed,
                use_gpu=config.use_gpu,
            ),
            k=int(k),
            output_dir=output_dir,
            device=device,
            record_id_column=record_id_column,
        )
        run_rows.append(
            {
                "K": result.k,
                "reconstruction_loss": result.reconstruction_loss,
                "num_features_used": result.num_features_used,
                "avg_active_features_per_chunk": result.avg_active_features_per_chunk,
            }
        )
        manifest_entries.append(
            {
                "K": result.k,
                "output_dir": str(result.output_dir),
                "hidden_dim": result.hidden_dim,
                "reconstruction_loss": result.reconstruction_loss,
                "num_features_used": result.num_features_used,
                "avg_active_features_per_chunk": result.avg_active_features_per_chunk,
            }
        )

    report_df = pd.DataFrame(run_rows, columns=REPORT_COLUMNS)
    report_df.to_csv(output_dir / "experiment_report.csv", index=False, encoding="utf-8")

    recommended_k = recommend_k(report_df)
    best_k_dir = output_dir / f"K_{recommended_k}"
    canonical_chunks_file = output_dir / "chunks.csv"
    source_chunks_file = output_dir / "source_chunks.csv"
    source_stage_chunks = resolve_existing_path(DEFAULT_STAGE_CHUNKS)
    copy_source_chunks(source_stage_chunks, source_chunks_file)

    best_feature_chunks_file = best_k_dir / "feature_chunks.csv"
    if not best_feature_chunks_file.is_file():
        raise FileNotFoundError(f"Best-K feature chunks not found: {best_feature_chunks_file}")
    best_feature_chunks = pd.read_csv(best_feature_chunks_file, encoding="utf-8")
    consolidated_chunks, sentence_overlap_df = consolidate_feature_chunks(best_feature_chunks)
    save_csv_frame(consolidated_chunks, canonical_chunks_file)
    save_csv_frame(sentence_overlap_df, output_dir / "sentence_feature_overlap.csv")

    recommendation_payload = {
        "recommended_k": recommended_k,
        "selection_rule": "smallest K within 15% of the best reconstruction loss, then lowest overlap burden",
        "available_k_values": list(config.top_k_values),
        "report": report_df.to_dict(orient="records"),
        "canonical_chunks_file": str(canonical_chunks_file),
        "canonical_chunks_source": str(source_chunks_file),
        "best_feature_chunks_file": str(best_feature_chunks_file),
        "sentence_overlap_file": str(output_dir / "sentence_feature_overlap.csv"),
    }
    save_json(output_dir / "k_recommendation.json", recommendation_payload)

    run_manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "embedding_model": embedding_model,
        "hidden_multiplier": config.hidden_multiplier,
        "top_k_values": list(config.top_k_values),
        "top_chunks_per_feature": config.top_chunks_per_feature,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "random_seed": config.random_seed,
        "use_gpu": config.use_gpu,
        "device": str(device),
        "chunk_count": int(len(chunks)),
        "embedding_dim": int(embeddings.shape[1]),
        "recommended_k": recommended_k,
        "canonical_chunks_file": str(canonical_chunks_file),
        "canonical_chunks_source": str(source_chunks_file),
        "best_feature_chunks_file": str(best_feature_chunks_file),
        "sentence_overlap_file": str(output_dir / "sentence_feature_overlap.csv"),
        "k_results": manifest_entries,
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
    }
    save_json(output_dir / "run_manifest.json", run_manifest)

    LOGGER.info("Experiment report saved to %s", output_dir / "experiment_report.csv")
    return report_df


def build_config_from_args(args: argparse.Namespace) -> SAEConfig:
    use_gpu = USE_GPU
    if args.cpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True

    top_k_values = parse_top_k_values(args.top_k_values)

    return SAEConfig(
        input_csv=args.input_csv or INPUT_CSV,
        output_dir=args.output_dir or OUTPUT_DIR,
        embedding_model=args.embedding_model or EMBEDDING_MODEL,
        hidden_multiplier=args.hidden_multiplier or HIDDEN_MULTIPLIER,
        sparsity_method=SPARSITY_METHOD,
        top_k_values=top_k_values,
        top_chunks_per_feature=args.top_chunks_per_feature or TOP_CHUNKS_PER_FEATURE,
        batch_size=args.batch_size or BATCH_SIZE,
        epochs=args.epochs or EPOCHS,
        learning_rate=args.learning_rate or LEARNING_RATE,
        random_seed=args.random_seed or RANDOM_SEED,
        use_gpu=use_gpu,
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    config = build_config_from_args(args)
    report = run_experiment(config)
    LOGGER.info("Completed SAE sweep over K values: %s", config.top_k_values)
    LOGGER.info("Final report:\n%s", report.to_string(index=False))


if __name__ == "__main__":
    main()