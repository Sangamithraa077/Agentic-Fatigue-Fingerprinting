"""Markdown report assembly for the explainability framework.

Split intentionally narrow: this module owns two kinds of genuinely
reusable logic --

1. `generate_report` -- assembling an ordered list of titled markdown
   sections into one file. This has no explainability-specific knowledge
   (it would work for any phase's report) and is kept generic on purpose.
2. `format_*` -- turning this package's typed result dataclasses
   (`RankedFeature`, `ModalityImportance`, `LocalExplanation`) into markdown
   tables. These are reused verbatim regardless of *which* model or sample
   produced the data, so they belong here rather than being rewritten by
   every caller.

Phase-specific narrative (which models were compared, what the results
mean, limitations discussion) is a report-authoring concern for the
notebook that has that context, not this module -- mirroring how Phase
9/10's `write_report` functions live in their notebooks, not in `ml/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ml.explainability.attention_visualizer import ModalityImportance
from ml.explainability.feature_ranker import RankedFeature
from ml.explainability.local_explainer import LocalExplanation


@dataclass(frozen=True)
class ReportSection:
    """One titled markdown section.

    Attributes:
        title: Rendered as a `##` heading.
        content: Markdown body (may itself contain further `###`
            subheadings, tables, or image references).
    """

    title: str
    content: str


def generate_report(sections: Sequence[ReportSection], output_path: Path, report_title: str, preamble: str = "") -> Path:
    """Assemble titled sections into one markdown file.

    Args:
        sections: Rendered in the given order, each as a `## title`
            heading followed by its content.
        output_path: Where to write the report (parent directories created
            if needed).
        report_title: The document's `#` title.
        preamble: Optional markdown inserted between the title and the
            first section (e.g. a "how to reproduce" line).

    Returns:
        `output_path`, for convenient chaining.
    """
    lines = [f"# {report_title}", ""]
    if preamble:
        lines.append(preamble)
        lines.append("")

    for section in sections:
        lines.append(f"## {section.title}")
        lines.append("")
        lines.append(section.content)
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def format_feature_ranking_table(ranking: Sequence[RankedFeature], top_k: int) -> str:
    """Markdown table of the top-`top_k` ranked features."""
    lines = ["| Rank | Feature | Mean Absolute SHAP Value |", "|---|---|---|"]
    for feature in ranking[:top_k]:
        lines.append(f"| {feature.rank + 1} | `{feature.feature_name}` | {feature.importance:.4f} |")
    return "\n".join(lines)


def format_modality_importance_table(importances: Sequence[ModalityImportance]) -> str:
    """Markdown table of each modality's normalized attention-based importance."""
    lines = ["| Modality | Importance |", "|---|---|"]
    for item in sorted(importances, key=lambda x: x.importance, reverse=True):
        lines.append(f"| {item.modality_name} | {item.importance:.3f} |")
    return "\n".join(lines)


def format_local_explanation_table(explanation: LocalExplanation, top_k: int = 10) -> str:
    """Markdown table of one local explanation's top signed feature contributions."""
    lines = [
        f"Predicted class: **{explanation.predicted_class}** (probability={explanation.predicted_probability:.3f}, "
        f"base value={explanation.base_value:.3f})",
        "",
        "| Feature | Value | SHAP contribution |",
        "|---|---|---|",
    ]
    for contribution in explanation.top_contributions[:top_k]:
        lines.append(f"| `{contribution.feature_name}` | {contribution.feature_value:.4f} | {contribution.shap_value:+.4f} |")
    return "\n".join(lines)
