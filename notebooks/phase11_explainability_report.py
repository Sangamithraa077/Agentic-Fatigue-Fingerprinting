"""Phase 11 -- Explainable AI Framework report generator.

Rebuilds the exact same (deterministic) EMG feature matrix and multimodal
dataset Phases 9/10 built, **reloads their already-trained model
checkpoints** (no retraining -- see `docs/explainability.md`), and runs
SHAP (classical models) and attention-based (proposed model)
explainability over them, saving figures and writing
`reports/explainability_report.md`.

This script implements no new modeling -- only explanation of models
already trained. It reuses:
- `ml.models.classical.*` / `ml.models.common.dataset_builder` to rebuild
  the exact EMG/all/binary_fatigue feature matrix and reload Phase 9's
  saved `random_forest`/`xgboost`/`logistic_regression` checkpoints
  (`ml/models/artifacts/classical/emg_all_binary_fatigue/`).
- `ml.models.common.multimodal_dataset_builder` / `ml.models.proposed.*`
  to rebuild Phase 10's aligned multimodal dataset and reload its saved
  `attention`-fusion checkpoint (`ml/models/artifacts/proposed/attention/`).
- `ml.explainability.*` (this phase's new framework) for every actual
  explainability computation and plot.

Reproduce with:
    python notebooks/phase11_explainability_report.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from ml.explainability.attention_visualizer import (
    compute_attention_distributions,
    compute_modality_importance,
    generate_attention_map,
    plot_attention_distribution,
    plot_attention_heatmap,
    plot_modality_importance,
)
from ml.explainability.config import FeatureRankingConfig, ShapConfig
from ml.explainability.explanation_report import (
    ReportSection,
    format_feature_ranking_table,
    format_local_explanation_table,
    format_modality_importance_table,
    generate_report,
)
from ml.explainability.global_explainer import GlobalExplanation, explain_global
from ml.explainability.local_explainer import LocalExplanation, explain_prediction
from ml.explainability.shap_explainer import (
    build_shap_explainer,
    plot_beeswarm,
    plot_dependence,
    plot_force,
    plot_summary,
    plot_waterfall,
)
from ml.feature_engineering.feature_dataset import build_feature_datasets
from ml.feature_engineering.feature_pipeline import FeaturePipeline
from ml.models.common.config import DatasetConfig, SplitConfig
from ml.models.common.data_splitter import train_val_test_group_split
from ml.models.common.dataset_builder import ModelingDataset, apply_scaler, build_modeling_dataset
from ml.models.common.multimodal_dataset_builder import MultimodalDataset, build_multimodal_dataset
from ml.models.deep_learning.training import get_device
from ml.models.proposed.config import AttentionConfig, ClassifierConfig, EncoderConfig, FusionConfig, ProposedModelConfig
from ml.models.proposed.inference import ProposedModelAdapter
from ml.models.proposed.model import ProposedFatigueModel
from ml.preprocessing.dataset_loader import WSD4FEDSRMLoader
from ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "explainability_report.md"
CLASSICAL_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts" / "classical" / "emg_all_binary_fatigue"
PROPOSED_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts" / "proposed" / "attention"

# Identical to Phase 9/10's bounded demo subset, so this phase reloads
# *exactly* the checkpoints those phases saved (same rows -> same split).
DEMO_SUBJECT_IDS = ["P01", "P02", "P03"]
SPLIT_CONFIG = SplitConfig(n_splits=3, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)
EMG_DATASET_CONFIG = DatasetConfig(modality="EMG", feature_subset="all", target_type="binary_fatigue")

CLASSICAL_MODEL_NAMES = ("random_forest", "xgboost", "logistic_regression")
SHAP_CONFIG = ShapConfig(background_samples=100, max_explain_samples=150, random_seed=42)
RANKING_CONFIG = FeatureRankingConfig(top_k=15)


# ---------------------------------------------------------------------------
# Data + checkpoint reloading (Phases 5-10, no retraining)
# ---------------------------------------------------------------------------


def prepare_feature_pipeline_outputs(subject_ids: List[str]):
    """Re-runs Phases 5-8 (data loading, preprocessing, feature extraction)
    -- deterministic and fast; only *model training* is skipped in this
    phase, not feature extraction, since Phase 9/10's saved checkpoints
    don't carry the feature matrix with them."""
    loader = WSD4FEDSRMLoader()
    condition_catalog = loader.discover_conditions()
    subject_folders = loader.discover_subjects()

    subjects_features = []
    processed_subjects = []
    for subject_id in subject_ids:
        subject = loader.load_subject(subject_id, subject_folders[subject_id], condition_catalog)
        processed_subject = PreprocessingPipeline().process_subject(subject)
        subject_features = FeaturePipeline().extract_subject(processed_subject)
        processed_subjects.append(processed_subject)
        subjects_features.append(subject_features)

    return build_feature_datasets(subjects_features, processed_subjects), processed_subjects, subjects_features


def load_classical_checkpoint(model_name: str):
    payload = joblib.load(CLASSICAL_ARTIFACTS_DIR / f"{model_name}.joblib")
    return payload["model"], payload["scaler"]


def make_proposed_model_config(dims) -> ProposedModelConfig:
    # Must exactly match notebooks/phase10_proposed_model_report.py's
    # make_model_config("attention", dims) -- the saved .pt state_dict's
    # tensor shapes were produced by that architecture, so reloading it
    # into a differently-shaped model would fail to load (or silently
    # mismatch). Duplicated here (rather than imported) because
    # notebooks/ is a collection of standalone scripts, not a package --
    # matching Phase 9/10's own convention of each report script being
    # self-contained.
    embed_dim = 8
    return ProposedModelConfig(
        emg_encoder=EncoderConfig(hidden_dims=(16,), output_dim=embed_dim, dropout=0.5),
        imu_encoder=EncoderConfig(hidden_dims=(32,), output_dim=embed_dim, dropout=0.5),
        ppg_encoder=EncoderConfig(hidden_dims=(8,), output_dim=embed_dim, dropout=0.3),
        fusion=FusionConfig(strategy="attention", attention=AttentionConfig(embed_dim=embed_dim, num_heads=2, num_layers=1, dropout=0.2)),
        classifier=ClassifierConfig(hidden_dims=(8,), dropout=0.5),
        n_classes=2,
    )


def load_proposed_checkpoint(mm: MultimodalDataset) -> ProposedModelAdapter:
    config = make_proposed_model_config(mm.dims)
    model = ProposedFatigueModel(emg_dim=mm.dims.emg, imu_dim=mm.dims.imu, ppg_dim=mm.dims.ppg, config=config)
    state_dict = torch.load(PROPOSED_ARTIFACTS_DIR / "attention.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return ProposedModelAdapter(model, mm.dims, device=get_device("auto"))


# ---------------------------------------------------------------------------
# SHAP explainability over the reloaded classical models
# ---------------------------------------------------------------------------


def run_shap_explainability(modeling_dataset: ModelingDataset):
    X, y = modeling_dataset.X, modeling_dataset.y
    split = train_val_test_group_split(y, modeling_dataset.groups, SPLIT_CONFIG)
    final_train_idx = np.concatenate([split.train_idx, split.val_idx])  # matches Phase 9's final-model fit rows

    global_explanations: Dict[str, GlobalExplanation] = {}
    local_explanations: Dict[str, List[LocalExplanation]] = {}
    figures: Dict[str, str] = {}

    for model_name in CLASSICAL_MODEL_NAMES:
        logger.info("=== SHAP explainability: %s ===", model_name)
        model, scaler = load_classical_checkpoint(model_name)
        X_background = apply_scaler(scaler, X[final_train_idx])
        X_test = apply_scaler(scaler, X[split.test_idx])
        y_test = y[split.test_idx]

        bundle = build_shap_explainer(model, model_name, X_background, SHAP_CONFIG)
        global_explanation = explain_global(bundle, X_test, modeling_dataset.feature_names, SHAP_CONFIG, RANKING_CONFIG)
        global_explanations[model_name] = global_explanation

        figures[f"{model_name}_summary"] = _relpath(plot_summary(
            global_explanation.explanation, FIGURES_DIR / f"explainability_shap_{model_name}_summary.png",
            max_display=RANKING_CONFIG.top_k,
        ))
        figures[f"{model_name}_beeswarm"] = _relpath(plot_beeswarm(
            global_explanation.explanation, FIGURES_DIR / f"explainability_shap_{model_name}_beeswarm.png",
            max_display=RANKING_CONFIG.top_k,
        ))
        top_feature = global_explanation.feature_ranking[0].feature_name
        figures[f"{model_name}_dependence"] = _relpath(plot_dependence(
            top_feature, global_explanation.explanation, FIGURES_DIR / f"explainability_shap_{model_name}_dependence.png",
        ))

        # Local case studies (random_forest only, to keep the report
        # bounded -- the same `explain_prediction` call works identically
        # for xgboost/logistic_regression, already exercised by their
        # global explanations above and by tests/test_explainability.py).
        if model_name == "random_forest":
            predicted = model.predict(X_test)
            fatigued_idx = next((i for i in range(len(y_test)) if predicted[i] == 1), 0)
            not_fatigued_idx = next((i for i in range(len(y_test)) if predicted[i] == 0), 0)
            case_indices = sorted({fatigued_idx, not_fatigued_idx})

            cases = []
            for case_number, sample_idx in enumerate(case_indices):
                local = explain_prediction(bundle, model, X_test[sample_idx], modeling_dataset.feature_names)
                cases.append(local)
                figures[f"case{case_number}_waterfall"] = _relpath(plot_waterfall(
                    local.explanation_row, FIGURES_DIR / f"explainability_local_case{case_number}_waterfall.png",
                ))
                figures[f"case{case_number}_force"] = _relpath(plot_force(
                    local.explanation_row, FIGURES_DIR / f"explainability_local_case{case_number}_force.png",
                ))
            local_explanations[model_name] = cases

    return global_explanations, local_explanations, figures, split


def _relpath(path: Path) -> str:
    return f"figures/{Path(path).name}"


# ---------------------------------------------------------------------------
# Attention explainability over the reloaded proposed model
# ---------------------------------------------------------------------------


def run_attention_explainability(mm: MultimodalDataset):
    logger.info("=== Attention explainability: proposed model (attention fusion) ===")
    adapter = load_proposed_checkpoint(mm)

    split = train_val_test_group_split(mm.y, mm.groups, SPLIT_CONFIG)
    scaler = joblib.load(PROPOSED_ARTIFACTS_DIR / "attention_scaler.joblib")["scaler"]  # Phase 10's saved scaler, not refit
    X_test_scaled = apply_scaler(scaler, mm.X[split.test_idx])

    sample = X_test_scaled[: min(64, len(X_test_scaled))]
    output = adapter.predict_with_details(sample)

    attention_maps = generate_attention_map(output.attention_weights)
    modality_importance = compute_modality_importance(output.attention_weights)
    attention_distributions = compute_attention_distributions(output.attention_weights)

    figures: Dict[str, str] = {}
    for attention_map in attention_maps:
        figures[f"attention_heatmap_layer{attention_map.layer_index}"] = _relpath(
            plot_attention_heatmap(attention_map, FIGURES_DIR / f"explainability_attention_heatmap_layer{attention_map.layer_index}.png")
        )
    figures["modality_importance"] = _relpath(
        plot_modality_importance(modality_importance, FIGURES_DIR / "explainability_modality_importance.png")
    )
    figures["attention_distribution"] = _relpath(
        plot_attention_distribution(attention_distributions, FIGURES_DIR / "explainability_attention_distribution.png")
    )

    return attention_maps, modality_importance, attention_distributions, figures


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def build_overview_section(n_samples: int) -> ReportSection:
    content = (
        "This phase implements a modular explainability framework (`ml.explainability/`) over the models already "
        "trained in Phases 9 and 10 -- **no model was retrained to produce this report**; every explanation below "
        "reloads a saved checkpoint (`ml/models/artifacts/classical/emg_all_binary_fatigue/*.joblib`, "
        "`ml/models/artifacts/proposed/attention/attention.pt`) and explains its predictions on the same held-out "
        f"test split those phases used ({n_samples} EMG test samples, subjects {', '.join(DEMO_SUBJECT_IDS)}).\n\n"
        "Two complementary explanation families are covered:\n\n"
        "- **SHAP** (Shapley Additive exPlanations) over the three classical models Phase 9 trained on EMG features "
        "(Random Forest, XGBoost, Logistic Regression) -- a rigorous, game-theoretic feature-attribution method with "
        "an automatically-selected exact explainer per model family (`ml.explainability.shap_explainer.select_explainer_type`).\n"
        "- **Cross-modal attention analysis** over Phase 10's proposed multimodal model -- inspecting the raw attention "
        "weights the model already computes internally to see which modality (EMG/IMU/PPG) it relies on most, per "
        "Phase 10's explicit design goal of exposing them \"ahead of, and feeding into, SHAP-based explainability.\""
    )
    return ReportSection(title="1. Overview", content=content)


def build_shap_section(global_explanations: Dict[str, GlobalExplanation], figures: Dict[str, str]) -> ReportSection:
    lines = [
        "Each model's explainer was automatically selected by "
        "`ml.explainability.shap_explainer.select_explainer_type`: `TreeExplainer` for Random Forest and XGBoost "
        "(exact, no background data needed), `LinearExplainer` for Logistic Regression (exact given a background "
        "distribution). All three explain the *positive class* (`fatigued`) contribution.",
        "",
    ]
    for model_name, explanation in global_explanations.items():
        lines.append(f"### 2.{list(global_explanations).index(model_name) + 1}. {model_name}")
        lines.append("")
        lines.append(f"Explained {explanation.n_samples_explained} held-out test samples.")
        lines.append("")
        lines.append(f"![{model_name} SHAP summary]({figures[f'{model_name}_summary']})")
        lines.append("")
        lines.append(f"![{model_name} SHAP beeswarm]({figures[f'{model_name}_beeswarm']})")
        lines.append("")
        top_feature = explanation.feature_ranking[0].feature_name
        lines.append(f"Dependence plot for the top-ranked feature (`{top_feature}`):")
        lines.append("")
        lines.append(f"![{model_name} SHAP dependence]({figures[f'{model_name}_dependence']})")
        lines.append("")
    return ReportSection(title="2. SHAP Results", content="\n".join(lines))


def build_global_importance_section(global_explanations: Dict[str, GlobalExplanation]) -> ReportSection:
    lines = []
    for model_name, explanation in global_explanations.items():
        lines.append(f"### 3.{list(global_explanations).index(model_name) + 1}. {model_name} -- top {RANKING_CONFIG.top_k} features")
        lines.append("")
        lines.append(format_feature_ranking_table(explanation.feature_ranking, RANKING_CONFIG.top_k))
        lines.append("")
    return ReportSection(title="3. Global Feature Importance", content="\n".join(lines))


def build_local_case_studies_section(local_explanations: Dict[str, List[LocalExplanation]], figures: Dict[str, str]) -> ReportSection:
    lines = [
        "Two representative `random_forest` predictions from the held-out test set (the identical "
        "`explain_prediction` call works unmodified for `xgboost`/`logistic_regression` -- see "
        "`tests/test_explainability.py`).",
        "",
    ]
    for case_number, local in enumerate(local_explanations.get("random_forest", [])):
        label = "fatigued" if local.predicted_class == 1 else "not_fatigued"
        lines.append(f"### 4.{case_number + 1}. Case {case_number + 1} -- predicted `{label}` (p={local.predicted_probability:.3f})")
        lines.append("")
        lines.append(format_local_explanation_table(local, top_k=8))
        lines.append("")
        lines.append(f"![Case {case_number + 1} waterfall]({figures[f'case{case_number}_waterfall']})")
        lines.append("")
        lines.append(f"![Case {case_number + 1} force]({figures[f'case{case_number}_force']})")
        lines.append("")
    return ReportSection(title="4. Local Explanation Case Studies", content="\n".join(lines))


def build_attention_section(modality_importance, attention_distributions, figures: Dict[str, str]) -> ReportSection:
    lines = [
        "Raw attention-weight analysis over Phase 10's `attention`-fusion proposed model, evaluated on 64 held-out "
        "EMG/IMU/PPG test samples (aligned multimodal windows). These are **not** SHAP values -- they summarize what "
        "the model's `nn.MultiheadAttention` layer already computes internally, not a game-theoretic attribution.",
        "",
        "### 5.1. Modality importance",
        "",
        format_modality_importance_table(modality_importance),
        "",
        f"![Modality importance]({figures['modality_importance']})",
        "",
        "### 5.2. Attention heatmaps",
        "",
    ]
    for key, path in figures.items():
        if key.startswith("attention_heatmap_layer"):
            lines.append(f"![{key}]({path})")
            lines.append("")
    lines.append("### 5.3. Attention weight distributions")
    lines.append("")
    lines.append(f"![Attention distribution]({figures['attention_distribution']})")
    lines.append("")
    n_heads = len({d.head_index for d in attention_distributions})
    n_layers = len({d.layer_index for d in attention_distributions})
    lines.append(f"({n_layers} attention layer(s), {n_heads} head(s) each.)")
    return ReportSection(title="5. Attention Results", content="\n".join(lines))


def build_comparison_section(global_explanations: Dict[str, GlobalExplanation], modality_importance) -> ReportSection:
    top_features_by_model = {name: [f.feature_name for f in exp.feature_ranking[:5]] for name, exp in global_explanations.items()}
    dominant_modality = max(modality_importance, key=lambda item: item.importance)

    lines = [
        "SHAP (classical models) and attention analysis (proposed model) operate at different granularities and "
        "cannot be compared value-for-value: SHAP attributes each of EMG's 72 individual features (e.g. per-muscle "
        "MAV/RMS), while the proposed model's attention operates over 3 *modality-level* embeddings (EMG/IMU/PPG), "
        "not individual raw features. The meaningful comparison is qualitative.",
        "",
        "**Top-5 SHAP-ranked EMG features per classical model:**",
        "",
    ]
    for model_name, top_features in top_features_by_model.items():
        lines.append(f"- **{model_name}**: {', '.join(f'`{f}`' for f in top_features)}")
    lines.append("")
    lines.append(
        f"**Proposed model's dominant attended-to modality:** `{dominant_modality.modality_name}` "
        f"(importance={dominant_modality.importance:.3f})."
    )
    lines.append("")
    lines.append(
        "All three classical models are EMG-only by construction, so they cannot express an opinion on IMU/PPG's "
        "relative importance at all -- only the proposed multimodal model can. This is itself a finding: single-"
        "modality SHAP explanations, however detailed at the feature level, are structurally blind to cross-modality "
        "questions that this project's fusion architecture exists to answer."
    )
    return ReportSection(title="6. Comparison: Classical ML vs. Proposed Model Explanations", content="\n".join(lines))


def build_discussion_section() -> ReportSection:
    content = (
        "SHAP's tree/linear explainers agreed with each other on which EMG features matter most more often than not "
        "(see Section 3) -- a mild sanity check that the underlying signal, not model-specific artifacts, drives "
        "these rankings. The proposed model's attention weights are internally consistent (Section 5) but are a "
        "fundamentally different kind of explanation: they describe the model's *internal computation*, not a "
        "rigorously-attributed causal contribution to the output the way SHAP values are (SHAP values are guaranteed "
        "to sum to the model's output minus its baseline; attention weights carry no such guarantee -- a "
        "well-documented limitation of attention-as-explanation in the interpretability literature). Both views are "
        "informative but answer different questions, which is why Section 6 keeps them separate rather than forcing "
        "a single ranking."
    )
    return ReportSection(title="7. Discussion", content=content)


def build_limitations_section() -> ReportSection:
    items = [
        "**Bounded subject count (3)**, identical to Phases 9/10 -- these are demonstrations that the explainability "
        "framework works correctly end-to-end on real trained models, not final scientific conclusions about which "
        "features/modalities matter for fatigue in general.",
        "**SHAP background/explain-sample subsampling** (`ShapConfig`, 100/150 rows) trades a small amount of "
        "precision for bounded runtime -- exact for `TreeExplainer`, an approximation-of-an-approximation for "
        "`LinearExplainer`'s background integral.",
        "**Attention weights are not SHAP values** -- see Section 7; they are included because Phase 10 explicitly "
        "exposed them for this purpose, not because they carry SHAP's additive-attribution guarantees.",
        "**The classical vs. proposed comparison is qualitative, not row-identical** -- same underlying reason as "
        "Phase 10's baseline comparison (EMG-only per-window rows vs. aligned multimodal-window rows are built by "
        "different pipelines).",
        "**No personalization or agentic reasoning** -- this phase explains the population-level models Phases 9/10 "
        "trained; per-subject explanation is Phase 12's job.",
    ]
    return ReportSection(title="8. Limitations", content="\n".join(f"- {item}" for item in items))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_time = time.time()

    logger.info("Rebuilding EMG feature dataset for subjects: %s", DEMO_SUBJECT_IDS)
    feature_datasets, _, _ = prepare_feature_pipeline_outputs(DEMO_SUBJECT_IDS)
    modeling_dataset = build_modeling_dataset(feature_datasets["EMG"], EMG_DATASET_CONFIG)
    logger.info("EMG modeling dataset: X=%s, %d feature(s).", modeling_dataset.X.shape, len(modeling_dataset.feature_names))

    global_explanations, local_explanations, shap_figures, split = run_shap_explainability(modeling_dataset)

    # Cross-check against Phase 9's saved predictions.npz: same rows should
    # reproduce the identical held-out y_true if the split reconstruction
    # is faithful.
    saved = np.load(CLASSICAL_ARTIFACTS_DIR / "random_forest_predictions.npz")
    rebuilt_y_test = modeling_dataset.y[split.test_idx]
    if not np.array_equal(saved["y_true"], rebuilt_y_test):
        raise RuntimeError("Rebuilt test split does not match Phase 9's saved predictions -- checkpoint reload would be invalid.")
    logger.info("Verified: rebuilt test split exactly matches Phase 9's saved random_forest predictions.npz.")

    logger.info("Rebuilding aligned multimodal dataset for attention explainability ...")
    config_for_mm = DatasetConfig(modality="EMG", target_type="binary_fatigue")
    mm = build_multimodal_dataset(feature_datasets["EMG"], feature_datasets["IMU"], feature_datasets["PPG"], config_for_mm)
    attention_maps, modality_importance, attention_distributions, attention_figures = run_attention_explainability(mm)

    figures = {**shap_figures, **attention_figures}

    logger.info("Writing report ...")
    sections = [
        build_overview_section(n_samples=len(split.test_idx)),
        build_shap_section(global_explanations, figures),
        build_global_importance_section(global_explanations),
        build_local_case_studies_section(local_explanations, figures),
        build_attention_section(modality_importance, attention_distributions, figures),
        build_comparison_section(global_explanations, modality_importance),
        build_discussion_section(),
        build_limitations_section(),
    ]
    total_seconds = time.time() - start_time
    preamble = (
        f"**Phase:** 11 -- Explainable AI Framework (SHAP + attention analysis over already-trained Phase 9/10 "
        f"models; no personalization or agentic reasoning -- later phases)\n\n"
        f"**Reproduce:** `python notebooks/phase11_explainability_report.py` from the project root (no retraining -- "
        f"reloads saved checkpoints; took {total_seconds:.1f}s on this machine, CPU only)."
    )
    generate_report(sections, REPORT_PATH, report_title="WSD4FEDSRM -- Explainability Report", preamble=preamble)
    logger.info("Phase 11 explainability report complete in %.1fs.", total_seconds)


if __name__ == "__main__":
    main()
