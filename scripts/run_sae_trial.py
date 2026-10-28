from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from olbench.utils import text_sentences


PIPELINE_VERSION = "sentence_topk_sae_trial_v1"
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_FILE = Path("data/interim/cleaned_corpus.txt")
DEFAULT_OUTPUT_DIR = Path("ontology_experiment_framework/stage_outputs/sae_trial_simple")
DEFAULT_EMBEDDING_MODEL = Path("models/all-mpnet-base-v2")
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_TOP_TERMS_PER_FEATURE = 10
FORBIDDEN_MARKERS = ("References", "\\hline", "\\begin{table}", "\\begin{tabular}")
FORBIDDEN_TERM_NORMALS = {"hline", "table", "tabular", "begin", "end"}


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_term(value: object) -> str:
    text = safe_text(value).casefold()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, top_k: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.top_k = max(1, min(int(top_k), int(feature_dim)))
        self.encoder = nn.Linear(self.input_dim, self.feature_dim)
        self.decoder = nn.Linear(self.feature_dim, self.input_dim)

    def sparse_encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.encoder(inputs))
        if self.top_k >= hidden.shape[-1]:
            return hidden
        _, indices = torch.topk(hidden, self.top_k, dim=-1)
        mask = torch.zeros_like(hidden)
        mask.scatter_(1, indices, 1.0)
        return hidden * mask

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.sparse_encode(inputs)
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / path


def load_config(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sentence-level Top-K SAE trial.")
    parser.add_argument("inputs", nargs="*", help="Cleaned corpus text files.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for trial artifacts.")
    parser.add_argument("--embedding-model", type=str, help="SentenceTransformer model path or name.")
    parser.add_argument("--feature-dim", type=int, help="Number of SAE features.")
    parser.add_argument("--top-k", type=int, help="Top-k SAE activations kept per sentence.")
    parser.add_argument("--epochs", type=int, help="Number of SAE training epochs.")
    parser.add_argument("--batch-size", type=int, help="Batch size for embedding and SAE training.")
    parser.add_argument("--learning-rate", type=float, help="SAE learning rate.")
    parser.add_argument("--device", type=str, help="Device for SAE training and inference.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--top-terms-per-feature", type=int, help="Max terms to keep per feature.")
    parser.add_argument("--spacy-model", type=str, help="spaCy model for noun chunks.")
    return parser.parse_args()


def merge_config_and_args(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    merged = dict(config)
    for key in (
        "output_dir",
        "embedding_model",
        "feature_dim",
        "top_k",
        "epochs",
        "batch_size",
        "learning_rate",
        "device",
        "seed",
        "top_terms_per_feature",
        "spacy_model",
    ):
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    if args.inputs:
        merged["input_files"] = args.inputs
    return merged


def validate_clean_input(text: str, source_path: Path) -> None:
    lowered = text.casefold()
    for marker in FORBIDDEN_MARKERS:
        if marker.casefold() in lowered:
            raise ValueError(f"Contaminated input detected in {source_path}: found {marker}")


def document_key(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
        raw_key = relative.as_posix()
    except Exception:
        raw_key = path.stem
    return re.sub(r"[^A-Za-z0-9]+", "_", raw_key).strip("_") or "document"


def split_corpus_into_sentences(input_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for document_index, input_file in enumerate(input_files):
        raw_text = input_file.read_text(encoding="utf-8", errors="ignore")
        validate_clean_input(raw_text, input_file)
        sentences = [sentence.strip() for sentence in text_sentences(raw_text) if sentence.strip()]
        doc_key = document_key(input_file)
        for sentence_index, sentence in enumerate(sentences):
            rows.append(
                {
                    "document_index": document_index,
                    "document_name": input_file.stem,
                    "source_path": str(input_file.resolve()),
                    "sentence_index": sentence_index,
                    "sentence_id": f"{doc_key}__s{sentence_index:06d}",
                    "sentence_text": sentence,
                }
            )
    columns = ["document_index", "document_name", "source_path", "sentence_index", "sentence_id", "sentence_text"]
    return pd.DataFrame(rows, columns=columns)


def encode_sentences(sentence_frame: pd.DataFrame, embedding_model: str | Path) -> np.ndarray:
    model = SentenceTransformer(str(embedding_model))
    vectors = model.encode(
        sentence_frame["sentence_text"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return np.asarray(vectors, dtype=np.float32)


def train_sae(
    embeddings: np.ndarray,
    feature_dim: int,
    top_k: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> tuple[TopKSparseAutoencoder, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)

    model = TopKSparseAutoencoder(embeddings.shape[1], feature_dim, top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(embeddings.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=max(1, batch_size), shuffle=True, generator=generator)

    history_rows: list[dict[str, object]] = []
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        history_rows.append({"epoch": epoch, "reconstruction_loss": total_loss / max(1, batches)})

    return model.eval(), pd.DataFrame(history_rows)


def infer_activations(model: TopKSparseAutoencoder, embeddings: np.ndarray, batch_size: int, device_name: str) -> np.ndarray:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    model = model.to(device)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, len(embeddings), max(1, batch_size)):
            batch = torch.from_numpy(embeddings[start_index : start_index + batch_size].astype(np.float32)).to(device)
            _, hidden = model(batch)
            rows.append(hidden.detach().cpu().numpy())
    return np.vstack(rows).astype(np.float32) if rows else np.zeros((0, model.feature_dim), dtype=np.float32)


def load_spacy_model(model_name: str = DEFAULT_SPACY_MODEL):
    try:
        import spacy
    except Exception as error:
        raise RuntimeError("spaCy is required for noun-chunk extraction.") from error
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as error:
        raise RuntimeError(f"Could not load spaCy model: {model_name}") from error


def extract_terms_from_sentence(sentence_text: str, nlp) -> list[str]:
    doc = nlp(sentence_text)
    terms: list[str] = []
    seen: set[str] = set()
    for chunk in getattr(doc, "noun_chunks", []):
        normalized = normalize_term(chunk.text)
        if not normalized or normalized in FORBIDDEN_TERM_NORMALS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(chunk.text.strip())
    return terms


def choose_top_sentences(feature_activations: np.ndarray, sentence_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_index in range(feature_activations.shape[1]):
        activations = feature_activations[:, feature_index]
        active_indices = np.flatnonzero(activations > 0)
        if len(active_indices) == 0:
            continue
        ranked_indices = active_indices[np.argsort(activations[active_indices])[::-1][:10]]
        for sentence_rank, sentence_index in enumerate(ranked_indices, start=1):
            sentence_row = sentence_frame.iloc[int(sentence_index)]
            rows.append(
                {
                    "feature_index": int(feature_index),
                    "sentence_rank": sentence_rank,
                    "sentence_id": sentence_row["sentence_id"],
                    "sentence_index": int(sentence_row["sentence_index"]),
                    "sentence_text": sentence_row["sentence_text"],
                    "activation": float(activations[int(sentence_index)]),
                }
            )
    columns = ["feature_index", "sentence_rank", "sentence_id", "sentence_index", "sentence_text", "activation"]
    return pd.DataFrame(rows, columns=columns)


def build_feature_terms(feature_top_sentences: pd.DataFrame, nlp, top_terms_per_feature: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if feature_top_sentences.empty:
        return pd.DataFrame(columns=[
            "feature_index",
            "term_rank",
            "term",
            "normalized_term",
            "supporting_sentence_count",
            "activation_sum",
            "activation_max",
            "supporting_sentence_ids",
            "evidence_sentences",
        ])

    for feature_index, group in feature_top_sentences.groupby("feature_index", sort=True):
        sentence_activation = {row.sentence_id: float(row.activation) for row in group.itertuples(index=False)}
        term_stats: dict[str, dict[str, object]] = {}

        for row in group.itertuples(index=False):
            for term in extract_terms_from_sentence(str(row.sentence_text), nlp):
                normalized_term = normalize_term(term)
                if not normalized_term or normalized_term in FORBIDDEN_TERM_NORMALS:
                    continue
                stats = term_stats.setdefault(
                    normalized_term,
                    {
                        "term": term.strip(),
                        "sentence_ids": [],
                        "evidence_sentences": [],
                        "sentence_id_set": set(),
                        "activation_sum": 0.0,
                        "activation_max": 0.0,
                    },
                )
                if row.sentence_id in stats["sentence_id_set"]:
                    continue
                stats["sentence_id_set"].add(row.sentence_id)
                stats["sentence_ids"].append(row.sentence_id)
                stats["evidence_sentences"].append(str(row.sentence_text))
                activation = sentence_activation[row.sentence_id]
                stats["activation_sum"] += activation
                stats["activation_max"] = max(stats["activation_max"], activation)

        ranked_terms = sorted(
            term_stats.items(),
            key=lambda item: (-len(item[1]["sentence_ids"]), -item[1]["activation_sum"], item[0]),
        )
        for term_rank, (normalized_term, stats) in enumerate(ranked_terms[:top_terms_per_feature], start=1):
            rows.append(
                {
                    "feature_index": int(feature_index),
                    "term_rank": term_rank,
                    "term": stats["term"],
                    "normalized_term": normalized_term,
                    "supporting_sentence_count": len(stats["sentence_ids"]),
                    "activation_sum": float(stats["activation_sum"]),
                    "activation_max": float(stats["activation_max"]),
                    "supporting_sentence_ids": " | ".join(stats["sentence_ids"]),
                    "evidence_sentences": " || ".join(stats["evidence_sentences"]),
                }
            )

    columns = [
        "feature_index",
        "term_rank",
        "term",
        "normalized_term",
        "supporting_sentence_count",
        "activation_sum",
        "activation_max",
        "supporting_sentence_ids",
        "evidence_sentences",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_feature_summary(feature_activations: np.ndarray, feature_top_terms: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_feature_indices = np.flatnonzero(feature_activations.max(axis=0) > 0)
    terms_by_feature = {key: value for key, value in feature_top_terms.groupby("feature_index")} if not feature_top_terms.empty else {}

    for feature_index in active_feature_indices:
        terms_frame = terms_by_feature.get(int(feature_index), pd.DataFrame())
        top_terms = " | ".join(terms_frame["term"].tolist()) if not terms_frame.empty else ""
        top_evidence = " || ".join(dict.fromkeys(terms_frame["evidence_sentences"].tolist())) if not terms_frame.empty else ""
        rows.append(
            {
                "feature_index": int(feature_index),
                "active_sentence_count": int((feature_activations[:, feature_index] > 0).sum()),
                "top_terms": top_terms,
                "top_evidence_sentences": top_evidence,
                "review_label": "unreviewed",
                "coherence_status": "coherent" if top_terms else "unclear",
                "ontology_utility": "candidate" if top_terms else "low",
                "review_notes": "Sentence-level SAE probe over noun chunks.",
            }
        )

    columns = [
        "feature_index",
        "active_sentence_count",
        "top_terms",
        "top_evidence_sentences",
        "review_label",
        "coherence_status",
        "ontology_utility",
        "review_notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def resolve_inputs(config: dict[str, object], cli_inputs: list[str] | None) -> list[Path]:
    raw_inputs = cli_inputs or config.get("input_files") or config.get("inputs") or [DEFAULT_INPUT_FILE]
    input_paths = [resolve_project_path(item) for item in raw_inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
    return input_paths


def run_trial(config: dict[str, object], cli_inputs: list[str] | None) -> dict[str, object]:
    input_files = resolve_inputs(config, cli_inputs)
    output_dir = resolve_project_path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to reuse stale output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Executing script: {SCRIPT_PATH}")
    print(f"Output directory: {output_dir.resolve()}")

    embedding_model = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    feature_dim = int(config.get("feature_dim", 256))
    top_k = int(config.get("top_k", 32))
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 1e-3))
    device_name = str(config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    top_terms_per_feature = int(config.get("top_terms_per_feature", DEFAULT_TOP_TERMS_PER_FEATURE))

    sentence_frame = split_corpus_into_sentences(input_files)
    sentence_frame.to_csv(output_dir / "sentences.csv", index=False, encoding="utf-8")

    embeddings = encode_sentences(sentence_frame, embedding_model)
    np.save(output_dir / "sentence_embeddings.npy", embeddings)

    model, history = train_sae(
        embeddings=embeddings,
        feature_dim=feature_dim,
        top_k=top_k,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device_name,
    )
    history.to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "top_k": top_k,
                "input_dim": int(embeddings.shape[1]),
                "pipeline_version": PIPELINE_VERSION,
            },
        },
        output_dir / "sae_model.pt",
    )

    feature_activations = infer_activations(model, embeddings, batch_size=batch_size, device_name=device_name)
    np.save(output_dir / "feature_activations.npy", feature_activations)

    nlp = load_spacy_model(str(config.get("spacy_model", DEFAULT_SPACY_MODEL)))
    feature_top_sentences = choose_top_sentences(feature_activations, sentence_frame)
    feature_top_sentences.to_csv(output_dir / "feature_top_sentences.csv", index=False, encoding="utf-8")

    feature_top_terms = build_feature_terms(feature_top_sentences, nlp, top_terms_per_feature)
    feature_top_terms.to_csv(output_dir / "feature_top_terms.csv", index=False, encoding="utf-8")

    feature_summary = build_feature_summary(feature_activations, feature_top_terms)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False, encoding="utf-8")

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "entry_point": str(SCRIPT_PATH.resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_files": [str(path.resolve()) for path in input_files],
        "feature_dim": feature_dim,
        "top_k": top_k,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device_name,
        "seed": seed,
        "sentence_count": int(len(sentence_frame)),
        "feature_count": int(feature_activations.shape[1]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["artifacts"] = sorted(path.name for path in output_dir.iterdir())
    return manifest


def main() -> None:
    args = parse_args()
    config_path = args.config or Path("ontology_experiment_framework/configs/sae_trial.json")
    config = load_config(config_path) if config_path and config_path.is_file() else {}
    merged = merge_config_and_args(config, args)
    manifest = run_trial(merged, args.inputs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
