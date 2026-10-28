from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


PROJECT_DIR = Path("ontology_experiment_framework")
INPUT_FILE = PROJECT_DIR / "stage_outputs/06_seed_guided_contextual_clustering/final_seed_guided_candidates.csv"
GROUND_TRUTH_FILE = PROJECT_DIR / "data/Ground_Truth.xlsx"
OUTPUT_DIR = PROJECT_DIR / "stage_outputs/07_nmf_filtering"

DEFAULT_EMBEDDING_MODEL = Path("models/all-mpnet-base-v2")
DEFAULT_REPRESENTATION = "term_context"
DEFAULT_N_COMPONENTS = 64
DEFAULT_TOP_TERMS_PER_FACTOR = 5
DEFAULT_BATCH_SIZE = 32

GROUND_TRUTH_SHEET = "S1_GT_TERMS"
NEGATIVE_TERMS_SHEET = "S7_NEGATIVE_TERMS"
MANUAL_REVIEW_STATUS = "needs_manual_review"

MAX_FACTOR_SHARE_THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
]

MULTI_FACTOR_RULES = [
    (2, 0.10),
    (2, 0.15),
    (2, 0.20),
    (3, 0.10),
    (3, 0.15),
    (3, 0.20),
]

PRIORITY_WEIGHTS = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


def normalize_term(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).casefold().strip()
    for dash in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(dash, "-")

    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9&/+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an NMF-based filtering trial over candidate terms."
    )
    parser.add_argument("--input-file", type=Path, help="Candidate CSV from stage 06.")
    parser.add_argument("--ground-truth-file", type=Path, help="Ground truth workbook.")
    parser.add_argument("--output-dir", type=Path, help="Directory for trial artifacts.")
    parser.add_argument("--embedding-model", type=str, help="SentenceTransformer model path or name.")
    parser.add_argument("--representation", type=str, help="term, term_sentence, or term_context.")
    parser.add_argument("--n-components", type=int, help="Number of NMF factors.")
    parser.add_argument("--top-terms-per-factor", type=int, help="How many top candidates to keep per factor.")
    parser.add_argument("--batch-size", type=int, help="Embedding batch size.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    return parser.parse_args()


def load_config(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def merge_config_and_args(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    merged = dict(config)
    for key in (
        "input_file",
        "ground_truth_file",
        "output_dir",
        "embedding_model",
        "representation",
        "n_components",
        "top_terms_per_factor",
        "batch_size",
    ):
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return merged


def validate_inputs(input_file: Path, ground_truth_file: Path) -> None:
    missing = []
    if not input_file.is_file():
        missing.append(str(input_file))
    if not ground_truth_file.is_file():
        missing.append(str(ground_truth_file))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_candidates(input_file: Path) -> pd.DataFrame:
    candidates = pd.read_csv(input_file, encoding="utf-8")

    required = {"candidate", "normalized_candidate"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"{input_file} is missing columns: {sorted(missing)}")

    for optional_column in ("source_sentence", "context_text"):
        if optional_column not in candidates.columns:
            candidates[optional_column] = ""

    candidates = candidates.copy()
    candidates["candidate"] = candidates["candidate"].map(safe_text)
    candidates["normalized_candidate"] = candidates["candidate"].map(normalize_term)
    candidates["source_sentence"] = candidates["source_sentence"].map(safe_text)
    candidates["context_text"] = candidates["context_text"].map(safe_text)
    candidates = candidates[candidates["normalized_candidate"].ne("")].copy()
    candidates = candidates.drop_duplicates("normalized_candidate", keep="first").reset_index(drop=True)

    if candidates.empty:
        raise ValueError("The input candidate inventory is empty.")

    return candidates


def load_ground_truth(ground_truth_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_excel(
        ground_truth_file,
        sheet_name=GROUND_TRUTH_SHEET,
        engine="openpyxl",
    )
    negatives = pd.read_excel(
        ground_truth_file,
        sheet_name=NEGATIVE_TERMS_SHEET,
        engine="openpyxl",
    )

    gt = gt[
        gt["keep_for_tasks"]
        .astype(str)
        .str.contains("term_extraction", case=False, na=False)
    ].copy()
    gt["normalized_gt"] = gt["term_canonical"].map(normalize_term)
    gt = gt[gt["normalized_gt"].ne("")].copy()

    status = gt["grounded_status_v2"].fillna("").astype(str).str.casefold()
    strict_gt = gt[status.ne(MANUAL_REVIEW_STATUS)].copy()
    review_gt = gt[status.eq(MANUAL_REVIEW_STATUS)].copy()

    negatives["normalized_negative"] = negatives["candidate"].map(normalize_term)
    negatives = negatives[negatives["normalized_negative"].ne("")].copy()

    return strict_gt, review_gt, negatives


def build_representations(
    candidates: pd.DataFrame,
    representation_name: str,
) -> list[str]:
    representations = []

    for row in candidates.itertuples():
        term = safe_text(row.candidate)
        sentence = safe_text(row.source_sentence)
        context = safe_text(row.context_text)

        if representation_name == "term":
            value = term
        elif representation_name == "term_sentence":
            value = f"{term}. Context sentence: {sentence}" if sentence else term
        elif representation_name == "term_context":
            value = f"{term}. Context: {context}" if context else term
        else:
            raise ValueError(f"Unknown representation: {representation_name}")

        representations.append(value)

    return representations


def encode_candidates(
    candidates: pd.DataFrame,
    model_name: str | Path,
    representation_name: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    model = SentenceTransformer(str(model_name))
    representations = build_representations(candidates, representation_name)
    embeddings = model.encode(
        representations,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=max(1, batch_size),
    )
    return np.asarray(embeddings, dtype=np.float32), representations


def annotate_matches(
    candidates: pd.DataFrame,
    strict_gt: pd.DataFrame,
    review_gt: pd.DataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    strict_lookup = strict_gt.drop_duplicates("normalized_gt").set_index("normalized_gt")
    review_lookup = review_gt.drop_duplicates("normalized_gt").set_index("normalized_gt")
    negative_set = set(negatives["normalized_negative"].dropna())

    result = candidates.copy()
    result["match_type"] = result["normalized_candidate"].map(
        lambda term: (
            "strict_exact"
            if term in strict_lookup.index
            else (
                "manual_review_exact"
                if term in review_lookup.index
                else "unmatched"
            )
        )
    )
    result["evaluation_flag_before_filter"] = result["match_type"].map(
        {
            "strict_exact": "TP",
            "manual_review_exact": "REVIEW",
            "unmatched": "FP",
        }
    )
    result["matched_gt_id"] = result["normalized_candidate"].map(
        lambda term: (
            strict_lookup.at[term, "gt_id"]
            if term in strict_lookup.index
            else (
                review_lookup.at[term, "gt_id"]
                if term in review_lookup.index
                else ""
            )
        )
    )
    result["gt_category"] = result["normalized_candidate"].map(
        lambda term: (
            strict_lookup.at[term, "gt_category"]
            if term in strict_lookup.index
            else (
                review_lookup.at[term, "gt_category"]
                if term in review_lookup.index
                else ""
            )
        )
    )
    result["negative_benchmark_hit"] = result["normalized_candidate"].isin(negative_set)

    return result


def evaluate_mask(
    experiment_name: str,
    annotated: pd.DataFrame,
    keep_mask: pd.Series,
    strict_gt: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    retained = annotated[keep_mask].copy()
    dropped = annotated[~keep_mask].copy()

    strict_unique = strict_gt.drop_duplicates("normalized_gt").copy()
    gold = set(strict_unique["normalized_gt"])

    scored_retained = retained[retained["match_type"].ne("manual_review_exact")]
    predicted = set(scored_retained["normalized_candidate"])

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gt_weights = (
        strict_unique["priority"]
        .fillna("low")
        .astype(str)
        .str.casefold()
        .map(PRIORITY_WEIGHTS)
        .fillna(1.0)
    )
    matched_mask = strict_unique["normalized_gt"].isin(predicted)
    weighted_recall = float(gt_weights[matched_mask].sum() / gt_weights.sum()) if gt_weights.sum() else 0.0

    metric = {
        "experiment": experiment_name,
        "model_name": annotated["model_name"].iloc[0],
        "representation": annotated["representation"].iloc[0],
        "n_components": int(annotated["n_components"].iloc[0]),
        "input_candidate_count": len(annotated),
        "retained_candidate_count": len(retained),
        "dropped_candidate_count": len(dropped),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "priority_weighted_recall": weighted_recall,
        "tp_retained": int(retained["match_type"].eq("strict_exact").sum()),
        "tp_dropped": int(dropped["match_type"].eq("strict_exact").sum()),
        "fp_retained": int(retained["match_type"].eq("unmatched").sum()),
        "fp_dropped": int(dropped["match_type"].eq("unmatched").sum()),
        "manual_review_retained": int(retained["match_type"].eq("manual_review_exact").sum()),
        "negative_hits_retained": int(retained["negative_benchmark_hit"].sum()),
        "negative_hits_dropped": int(dropped["negative_benchmark_hit"].sum()),
    }

    retained["filter_experiment"] = experiment_name
    retained["filter_decision"] = "retained"
    dropped["filter_experiment"] = experiment_name
    dropped["filter_decision"] = "dropped"

    return metric, retained, dropped


def factor_hint_counts(terms: list[str]) -> dict[str, int]:
    blob = " ".join(terms).lower()
    categories = {
        "structural_hint_count": ["system", "subsystem", "component", "module", "unit", "assembly", "airframe", "payload", "station"],
        "sensor_actuator_hint_count": ["sensor", "camera", "radar", "lidar", "imu", "gps", "actuator", "servo", "antenna", "receiver", "transmitter"],
        "control_navigation_hint_count": ["control", "controller", "navigation", "guidance", "autopilot", "flight", "avionics"],
        "communication_hint_count": ["communication", "link", "datalink", "telemetry", "command", "antenna", "radio"],
        "power_propulsion_hint_count": ["power", "battery", "propulsion", "motor", "engine", "fuel", "propeller"],
        "parameter_hint_count": ["altitude", "speed", "range", "endurance", "weight", "accuracy", "rate", "frequency", "bandwidth"],
        "mission_hint_count": ["mission", "surveillance", "reconnaissance", "tracking", "monitoring", "inspection", "target"],
        "environment_hint_count": ["airspace", "weather", "wind", "terrain", "obstacle", "traffic", "environment"],
    }
    return {key: sum(blob.count(hint) for hint in hints) for key, hints in categories.items()}


def summarize_factor_terms(texts: list[str], topn: int = 16) -> str:
    stop = {
        "concept",
        "supporting",
        "context",
        "alias",
        "acronym",
        "candidate",
        "source",
        "sentence",
    }
    words: list[str] = []
    for text in texts:
        for word in re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-/]{2,}\b", text.lower()):
            if word not in stop:
                words.append(word)
    if not words:
        return ""
    return ", ".join(pd.Series(words).value_counts().head(topn).index.tolist())


def build_factor_summary(records: pd.DataFrame, loadings: np.ndarray, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for factor_index in range(loadings.shape[1]):
        order = np.argsort(loadings[:, factor_index])[::-1][:top_n]
        top_terms = records.iloc[order]["normalized_candidate"].tolist()
        top_texts = records.iloc[order]["embedding_text"].tolist()
        summary_rows.append(
            {
                "factor_id": f"factor_{factor_index:03d}",
                "top_keywords_from_records": summarize_factor_terms(top_texts),
                "top_terms_preview": " | ".join(top_terms[:12]),
                **factor_hint_counts(top_terms),
            }
        )
        for rank, row_index in enumerate(order, start=1):
            row = records.iloc[row_index]
            top_rows.append(
                {
                    "factor_id": f"factor_{factor_index:03d}",
                    "rank": rank,
                    "activation": float(loadings[row_index, factor_index]),
                    "concept_id": row.concept_id,
                    "term_canonical": row.normalized_candidate,
                    "candidate": row.candidate,
                    "normalized_candidate": row.normalized_candidate,
                    "matched_gt_id": row.get("matched_gt_id", ""),
                    "gt_category": row.get("gt_category", ""),
                    "source_sentence": row.source_sentence,
                    "context_text": row.context_text,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows)


def build_nmf_filter_experiments(annotated: pd.DataFrame) -> dict[str, pd.Series]:
    experiments = {
        "baseline_no_filter": pd.Series(True, index=annotated.index),
    }

    for threshold in MAX_FACTOR_SHARE_THRESHOLDS:
        experiments[f"max_factor_share_ge_{threshold:.2f}"] = annotated["max_factor_share"] >= threshold

    for required_factor_count, share_threshold in MULTI_FACTOR_RULES:
        column = f"factor_count_ge_{share_threshold:.2f}"
        experiments[f"at_least_{required_factor_count}_factors_ge_{share_threshold:.2f}"] = annotated[column] >= required_factor_count

    for threshold in (0.05, 0.10, 0.15, 0.20):
        experiments[f"dominant_margin_ge_{threshold:.2f}"] = annotated["dominant_factor_margin"] >= threshold

    for threshold in (0.20, 0.25, 0.30):
        experiments[
            f"focused_high_share_ge_{threshold:.2f}_and_le_2_factors"
        ] = (
            (annotated["max_factor_share"] >= threshold)
            & (annotated["factor_count_ge_0.10"] <= 2)
        )

        experiments[
            f"exclude_high_share_diffuse_ge_{threshold:.2f}_and_ge_3_factors"
        ] = ~(
            (annotated["max_factor_share"] >= threshold)
            & (annotated["factor_count_ge_0.10"] >= 3)
        )

    top_factor_column = next(
        (column for column in annotated.columns if column.startswith("factor_in_top_")),
        None,
    )
    if top_factor_column is not None:
        experiments[f"in_any_{top_factor_column}"] = annotated[top_factor_column] >= 1
        experiments[f"in_at_least_2_{top_factor_column}"] = annotated[top_factor_column] >= 2

        for threshold in (0.10, 0.15, 0.20):
            experiments[
                f"top5_and_max_share_ge_{threshold:.2f}"
            ] = (
                (annotated[top_factor_column] >= 1)
                & (annotated["max_factor_share"] >= threshold)
            )

    return experiments


def write_report(
    records: pd.DataFrame,
    factor_summary: pd.DataFrame,
    best_experiment: str,
    args: dict[str, object],
    outdir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# UAV Stage 07 NMF Filtering Trial")
    lines.append("")
    lines.append("## What was scored")
    lines.append("Each record is a candidate term from stage 05 plus its source sentence/context, embedded with MPNet and factorized with NMF after min-max scaling.")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Input: `{args['input_file']}`")
    lines.append(f"- Ground truth: `{args['ground_truth_file']}`")
    lines.append(f"- MPNet model: `{args['embedding_model']}`")
    lines.append(f"- Representation: `{args['representation']}`")
    lines.append(f"- NMF factors: `{args['n_components']}`")
    lines.append(f"- Concepts scored: `{len(records)}`")
    lines.append("")
    lines.append("## Important interpretation rule")
    lines.append("NMF does not directly create ontology classes. It only exposes latent factors over the embedding space. Use the factor tables to inspect which candidate terms cluster together before deciding whether a filter is useful.")
    lines.append("")
    lines.append("## Best experiment")
    lines.append(best_experiment)
    lines.append("")
    lines.append("## First 20 factor previews")
    preview_cols = [
        "factor_id",
        "top_terms_preview",
        "top_keywords_from_records",
        "structural_hint_count",
        "sensor_actuator_hint_count",
        "control_navigation_hint_count",
        "parameter_hint_count",
        "mission_hint_count",
    ]
    if not factor_summary.empty:
        lines.append(factor_summary[preview_cols].head(20).to_markdown(index=False))
    else:
        lines.append("No factor summary rows were produced.")
    lines.append("")
    lines.append("## Files to inspect first")
    lines.append("1. `nmf_candidate_scores.csv` - candidate-level factor scores and metadata.")
    lines.append("2. `nmf_factor_summary.csv` - quick factor overview.")
    lines.append("3. `nmf_top_terms_by_factor.csv` - top activating candidates per factor.")
    lines.append("4. `nmf_filter_metrics.csv` - keep/drop metrics for each NMF rule.")

    (outdir / "nmf_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_trial(config: dict[str, object]) -> dict[str, object]:
    input_file = resolve_project_path(config.get("input_file", INPUT_FILE))
    ground_truth_file = resolve_project_path(config.get("ground_truth_file", GROUND_TRUTH_FILE))
    output_dir = resolve_project_path(config.get("output_dir", OUTPUT_DIR))
    embedding_model = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    representation = str(config.get("representation", DEFAULT_REPRESENTATION))
    n_components = int(config.get("n_components", DEFAULT_N_COMPONENTS))
    top_terms_per_factor = int(config.get("top_terms_per_factor", DEFAULT_TOP_TERMS_PER_FACTOR))
    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))

    validate_inputs(input_file, ground_truth_file)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to reuse stale output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    filters_dir = output_dir / "by_filter"
    filters_dir.mkdir(parents=True, exist_ok=True)

    print(f"Executing script: {Path(__file__).resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    candidates = load_candidates(input_file)
    strict_gt, review_gt, negatives = load_ground_truth(ground_truth_file)

    started = time.time()
    embeddings, representations = encode_candidates(
        candidates=candidates,
        model_name=embedding_model,
        representation_name=representation,
        batch_size=batch_size,
    )
    np.save(output_dir / "candidate_embeddings.npy", embeddings)

    scaler = MinMaxScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=1200,
        random_state=42,
    )
    loadings = nmf.fit_transform(scaled_embeddings)
    components = nmf.components_
    np.save(output_dir / "nmf_loadings.npy", loadings)
    np.save(output_dir / "nmf_components.npy", components)

    row_sums = loadings.sum(axis=1, keepdims=True)
    loading_shares = np.divide(
        loadings,
        row_sums,
        out=np.zeros_like(loadings),
        where=row_sums > 0,
    )

    score_df = candidates.copy()
    score_df.insert(0, "concept_id", [f"K{i:07d}" for i in range(1, len(score_df) + 1)])
    score_df["embedding_text"] = representations
    score_df["model_name"] = str(embedding_model)
    score_df["representation"] = representation
    score_df["n_components"] = n_components

    for factor_index in range(n_components):
        score_df[f"factor_{factor_index:03d}_activation"] = loadings[:, factor_index]
        score_df[f"factor_{factor_index:03d}_share"] = loading_shares[:, factor_index]

    score_df["dominant_factor_index"] = loading_shares.argmax(axis=1)
    score_df["max_factor_activation"] = loadings.max(axis=1)
    score_df["max_factor_share"] = loading_shares.max(axis=1)
    score_df["dominant_factor_activation"] = loadings[np.arange(len(loadings)), score_df["dominant_factor_index"].to_numpy()]
    score_df["dominant_factor_share"] = loading_shares[np.arange(len(loadings)), score_df["dominant_factor_index"].to_numpy()]

    sorted_shares = np.sort(loading_shares, axis=1)
    if loading_shares.shape[1] >= 2:
        score_df["second_factor_share"] = sorted_shares[:, -2]
    else:
        score_df["second_factor_share"] = 0.0
    score_df["dominant_factor_margin"] = score_df["dominant_factor_share"] - score_df["second_factor_share"]
    score_df["factor_count_ge_0.10"] = (loading_shares >= 0.10).sum(axis=1)
    score_df["factor_count_ge_0.15"] = (loading_shares >= 0.15).sum(axis=1)
    score_df["factor_count_ge_0.20"] = (loading_shares >= 0.20).sum(axis=1)

    factor_rank_matrix = np.argsort(np.argsort(-loadings, axis=0), axis=0) + 1
    score_df["best_factor_rank"] = factor_rank_matrix.min(axis=1)
    score_df[f"factor_in_top_{top_terms_per_factor}_count"] = (
        factor_rank_matrix <= top_terms_per_factor
    ).sum(axis=1)

    entropy_base = np.log(np.maximum(loading_shares.shape[1], 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(loading_shares > 0, loading_shares * np.log(loading_shares), 0.0), axis=1)
    score_df["factor_entropy"] = entropy / entropy_base if entropy_base > 0 else entropy

    annotated = annotate_matches(score_df, strict_gt, review_gt, negatives)
    annotated.to_csv(output_dir / "nmf_candidate_scores.csv", index=False, encoding="utf-8")

    factor_summary, top_terms = build_factor_summary(annotated, loadings, top_terms_per_factor)
    factor_summary.to_csv(output_dir / "nmf_factor_summary.csv", index=False, encoding="utf-8")
    top_terms.to_csv(output_dir / "nmf_top_terms_by_factor.csv", index=False, encoding="utf-8")

    all_metrics: list[dict[str, object]] = []
    manifest = {
        "stage": "07_nmf_filtering",
        "input_file": str(input_file.resolve()),
        "ground_truth_file": str(ground_truth_file.resolve()),
        "input_candidate_count": len(candidates),
        "embedding_model": str(embedding_model),
        "representation": representation,
        "n_components": n_components,
        "max_factor_share_thresholds": MAX_FACTOR_SHARE_THRESHOLDS,
        "multi_factor_rules": MULTI_FACTOR_RULES,
        "runs": [],
    }

    run_name = f"nmf__{representation}__{n_components}f"
    run_outputs = []
    experiments = build_nmf_filter_experiments(annotated)

    for experiment_name, keep_mask in experiments.items():
        full_experiment_name = f"{run_name}__{experiment_name}"
        metric, retained, dropped = evaluate_mask(
            full_experiment_name,
            annotated,
            keep_mask,
            strict_gt,
        )
        all_metrics.append(metric)
        run_outputs.append((metric, retained, dropped))

    run_metrics_df = pd.DataFrame(all_metrics).sort_values(
        ["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    run_metrics_df.to_csv(output_dir / "nmf_filter_metrics.csv", index=False, encoding="utf-8")

    best_experiment = run_metrics_df.iloc[0]["experiment"]
    best_tuple = next(item for item in run_outputs if item[0]["experiment"] == best_experiment)
    _, best_retained, best_dropped = best_tuple
    best_retained.to_csv(filters_dir / f"{run_name}__best_retained.csv", index=False, encoding="utf-8")
    best_dropped.to_csv(filters_dir / f"{run_name}__best_dropped.csv", index=False, encoding="utf-8")
    best_retained.to_csv(output_dir / "best_nmf_filtered_candidates.csv", index=False, encoding="utf-8")

    manifest["runs"].append(
        {
            "run": run_name,
            "status": "completed",
            "duration_seconds": round(time.time() - started, 3),
            "best_experiment": best_experiment,
            "best_f1": float(run_metrics_df.iloc[0]["f1"]),
        }
    )

    metrics_df = run_metrics_df.copy()
    metrics_df.to_csv(output_dir / "nmf_metrics.csv", index=False, encoding="utf-8")

    if metrics_df.empty:
        raise RuntimeError("No NMF filtering experiment completed successfully.")

    best_experiment = metrics_df.iloc[0]["experiment"]
    manifest["best_experiment_by_f1"] = best_experiment
    manifest["best_f1"] = float(metrics_df.iloc[0]["f1"])
    manifest["best_candidate_file"] = str(output_dir / "best_nmf_filtered_candidates.csv")

    write_report(
        annotated,
        factor_summary,
        best_experiment,
        {
            "input_file": str(input_file),
            "ground_truth_file": str(ground_truth_file),
            "embedding_model": str(embedding_model),
            "representation": representation,
            "n_components": n_components,
        },
        output_dir,
    )

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Stage 07 NMF filtering completed")
    print("=" * 80)
    print(
        metrics_df[
            [
                "experiment",
                "retained_candidate_count",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "tp_dropped",
                "fp_dropped",
            ]
        ].head(20).to_string(index=False)
    )
    print(f"\nBest experiment: {best_experiment}")
    print(f"Best retained candidates: {output_dir / 'best_nmf_filtered_candidates.csv'}")
    print(f"Markdown report: {output_dir / 'nmf_audit_report.md'}")

    manifest["all_best_retained_files"] = [str((output_dir / "best_nmf_filtered_candidates.csv").resolve())]
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest


def write_report(
    records: pd.DataFrame,
    factor_summary: pd.DataFrame,
    best_experiment: str,
    args: dict[str, object],
    outdir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# UAV Stage 07 NMF Filtering Trial")
    lines.append("")
    lines.append("## What was scored")
    lines.append("Each record is a candidate term from stage 05 plus its source sentence/context, embedded with MPNet and factorized with NMF after min-max scaling.")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Input: `{args['input_file']}`")
    lines.append(f"- Ground truth: `{args['ground_truth_file']}`")
    lines.append(f"- MPNet model: `{args['embedding_model']}`")
    lines.append(f"- Representation: `{args['representation']}`")
    lines.append(f"- NMF factors: `{args['n_components']}`")
    lines.append(f"- Concepts scored: `{len(records)}`")
    lines.append("")
    lines.append("## Important interpretation rule")
    lines.append("NMF does not directly create ontology classes. It only exposes latent factors over the embedding space. Use the factor tables to inspect which candidate terms cluster together before deciding whether a filter is useful.")
    lines.append("")
    lines.append("## Best experiment")
    lines.append(best_experiment)
    lines.append("")
    lines.append("## First 20 factor previews")
    preview_cols = [
        "factor_id",
        "top_terms_preview",
        "top_keywords_from_records",
        "structural_hint_count",
        "sensor_actuator_hint_count",
        "control_navigation_hint_count",
        "parameter_hint_count",
        "mission_hint_count",
    ]
    if not factor_summary.empty:
        lines.append(factor_summary[preview_cols].head(20).to_markdown(index=False))
    else:
        lines.append("No factor summary rows were produced.")
    lines.append("")
    lines.append("## Files to inspect first")
    lines.append("1. `nmf_candidate_scores.csv` - candidate-level factor scores and metadata.")
    lines.append("2. `nmf_factor_summary.csv` - quick factor overview.")
    lines.append("3. `nmf_top_terms_by_factor.csv` - top activating candidates per factor.")
    lines.append("4. `nmf_filter_metrics.csv` - keep/drop metrics for each NMF rule.")

    (outdir / "nmf_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config and args.config.is_file() else {}
    merged = merge_config_and_args(config, args)
    manifest = run_trial(merged)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()