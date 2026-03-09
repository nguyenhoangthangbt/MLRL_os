"""Standalone HTML report export for experiment results.

Generates a self-contained HTML file with embedded CSS and inline SVG charts.
No external dependencies (no CDN, no Plotly, no JavaScript libraries).
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from mlrl_os.core.experiment import (
    AlgorithmScore,
    ExperimentResult,
    FeatureImportanceEntry,
)

logger = logging.getLogger(__name__)

# Maximum number of features to display in the importance chart.
_MAX_FEATURES = 20

# SVG chart dimensions.
_SVG_WIDTH = 600
_SVG_BAR_HEIGHT = 22
_SVG_LABEL_WIDTH = 180
_SVG_BAR_MAX_WIDTH = 380
_SVG_VALUE_WIDTH = 40


def export_html_report(result: ExperimentResult, output_path: Path) -> Path:
    """Generate a standalone HTML report with embedded CSS.

    Args:
        result: Completed experiment result.
        output_path: File path to write the HTML report to.

    Returns:
        The output path (same as *output_path*).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        _render_summary(result),
        _render_metrics(result),
        _render_algorithm_comparison(result),
        _render_feature_importance(result),
    ]

    body = "\n".join(sections)
    page = _wrap_page(result.name, body)

    output_path.write_text(page, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_summary(result: ExperimentResult) -> str:
    """Render the experiment summary section."""
    duration_text = _format_duration(result.duration_seconds)
    rows = [
        ("Experiment ID", _esc(result.experiment_id)),
        ("Name", _esc(result.name)),
        ("Type", _esc(result.experiment_type.value)),
        ("Status", _esc(result.status.value)),
        ("Created", _esc(result.created_at)),
        ("Duration", duration_text),
        ("Best Algorithm", _esc(result.best_algorithm or "N/A")),
        ("Samples", str(result.sample_count or "N/A")),
        ("Features", str(result.feature_count or "N/A")),
    ]
    table_rows = "\n".join(
        f"        <tr><td class=\"label\">{label}</td><td>{value}</td></tr>"
        for label, value in rows
    )
    return f"""
    <section class="card">
      <h2>Experiment Summary</h2>
      <table class="summary-table">
{table_rows}
      </table>
    </section>"""


def _render_metrics(result: ExperimentResult) -> str:
    """Render the metrics table section."""
    if not result.metrics:
        return ""

    header = "        <tr><th>Metric</th><th>Value</th></tr>"
    rows = "\n".join(
        f"        <tr><td>{_esc(name)}</td><td>{value:.4f}</td></tr>"
        for name, value in result.metrics.items()
    )
    return f"""
    <section class="card">
      <h2>Metrics</h2>
      <table class="data-table">
{header}
{rows}
      </table>
    </section>"""


def _render_algorithm_comparison(result: ExperimentResult) -> str:
    """Render the algorithm comparison table."""
    scores = result.all_algorithm_scores
    if not scores or len(scores) < 1:
        return ""

    # Collect all metric names from the first score entry.
    metric_names = list(scores[0].metrics.keys())
    metric_headers = "".join(f"<th>{_esc(m)}</th>" for m in metric_names)
    header = f"        <tr><th>Rank</th><th>Algorithm</th>{metric_headers}</tr>"

    rows_html: list[str] = []
    for score in scores:
        metric_cells = "".join(
            f"<td>{score.metrics.get(m, 0):.4f}"
            f" <span class=\"std\">\u00b1{score.metrics_std.get(m, 0):.4f}</span></td>"
            for m in metric_names
        )
        rows_html.append(
            f"        <tr><td>#{score.rank}</td>"
            f"<td>{_esc(score.algorithm)}</td>{metric_cells}</tr>"
        )

    rows = "\n".join(rows_html)
    return f"""
    <section class="card">
      <h2>Algorithm Comparison</h2>
      <table class="data-table">
{header}
{rows}
      </table>
    </section>"""


def _render_feature_importance(result: ExperimentResult) -> str:
    """Render feature importance as an inline SVG horizontal bar chart."""
    entries = result.feature_importance
    if not entries:
        return ""

    # Limit to top N features.
    entries = entries[:_MAX_FEATURES]
    max_importance = max(e.importance for e in entries) if entries else 1.0
    if max_importance == 0:
        max_importance = 1.0

    chart_height = len(entries) * _SVG_BAR_HEIGHT + 10
    bars: list[str] = []

    for idx, entry in enumerate(entries):
        y = idx * _SVG_BAR_HEIGHT + 5
        bar_width = (entry.importance / max_importance) * _SVG_BAR_MAX_WIDTH
        text_y = y + _SVG_BAR_HEIGHT * 0.65

        # Label
        bars.append(
            f'    <text x="{_SVG_LABEL_WIDTH - 5}" y="{text_y}" '
            f'text-anchor="end" class="bar-label">{_esc(entry.feature)}</text>'
        )
        # Bar
        bars.append(
            f'    <rect x="{_SVG_LABEL_WIDTH}" y="{y + 2}" '
            f'width="{bar_width:.1f}" height="{_SVG_BAR_HEIGHT - 6}" '
            f'class="bar" rx="2" />'
        )
        # Value
        bars.append(
            f'    <text x="{_SVG_LABEL_WIDTH + bar_width + 5}" y="{text_y}" '
            f'class="bar-value">{entry.importance:.3f}</text>'
        )

    svg_content = "\n".join(bars)
    return f"""
    <section class="card">
      <h2>Feature Importance (Top {len(entries)})</h2>
      <svg width="{_SVG_WIDTH}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">
{svg_content}
      </svg>
    </section>"""


# ---------------------------------------------------------------------------
# Page wrapper
# ---------------------------------------------------------------------------


def _wrap_page(title: str, body: str) -> str:
    """Wrap body sections in a complete HTML document with embedded CSS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_esc(title)} — ML/RL OS Report</title>
  <style>
    :root {{
      --bg: #f8f9fa;
      --card-bg: #ffffff;
      --text: #212529;
      --muted: #6c757d;
      --border: #dee2e6;
      --accent: #0d6efd;
      --accent-light: #cfe2ff;
      --bar-color: #4361ee;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem;
      max-width: 900px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 1.5rem;
      margin-bottom: 0.25rem;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 0.875rem;
      margin-bottom: 1.5rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
    }}
    .card h2 {{
      font-size: 1.1rem;
      margin-bottom: 0.75rem;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .summary-table td {{
      padding: 0.35rem 0.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .summary-table td.label {{
      font-weight: 600;
      width: 160px;
      color: var(--muted);
    }}
    .data-table th,
    .data-table td {{
      padding: 0.4rem 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    .data-table th {{
      background: var(--accent-light);
      font-weight: 600;
      font-size: 0.85rem;
    }}
    .std {{
      color: var(--muted);
      font-size: 0.8rem;
    }}
    .bar-label {{
      font-size: 12px;
      fill: var(--text);
    }}
    .bar {{
      fill: var(--bar-color);
    }}
    .bar-value {{
      font-size: 11px;
      fill: var(--muted);
    }}
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.75rem;
      margin-top: 2rem;
    }}
  </style>
</head>
<body>
  <h1>{_esc(title)}</h1>
  <p class="subtitle">ML/RL OS Experiment Report</p>
{body}
  <footer>Generated by ML/RL OS</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Escape text for safe HTML inclusion."""
    return html.escape(str(text))


def _format_duration(seconds: float | None) -> str:
    """Format duration in a human-readable form."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}m {remaining:.1f}s"
