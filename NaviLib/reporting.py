"""Portable, self-contained HTML reports for exploratory analysis."""

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
import base64
import io

import numpy as np
import pandas as pd

from . import eda
from .quality import audit_data, _check_frame
from .theme import get_theme, theme_context

__all__ = ["DataReport", "create_report"]


@dataclass
class DataReport:
    """Analysis tables and figures with notebook and offline HTML rendering.

    Attributes
    ----------
    title : str
        Report heading, escaped as text in HTML.
    summary : dict
        Full-data row/column counts, missing percentage and memory size.
    tables : dict of pandas.DataFrame
        Full-data analysis tables, available for filtering and export.
    figures : dict of matplotlib.figure.Figure
        Closed figure handles; can still be saved or inspected.
    theme : dict
        Configuration captured when the report was created.
    notes : list of str
        Sampling and scope information shown in the report.
    """
    title: str
    summary: dict
    tables: dict
    figures: dict
    theme: dict
    notes: list[str] = field(default_factory=list)

    def to_html(self, path=None, *, max_rows: int = 50) -> str:
        """Render a standalone HTML string, optionally writing it to UTF-8.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination file. Parent directories are created as needed.
        max_rows : int, default 50
            Per-table display cap. Complete results remain in ``tables``.

        Returns
        -------
        str
            HTML with embedded PNG figures and CSS; no remote assets or scripts.
            Dataset values, headers and titles are escaped before insertion.
        """
        if not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer.")
        t = self.theme
        css = f"""
        :root {{color-scheme:{'dark' if t['name'] == 'dark' else 'light'};}}
        * {{box-sizing:border-box}} body {{margin:0;background:{t['background']};
        color:{t['text']};font:15px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}}
        main {{max-width:1240px;margin:auto;padding:48px 28px}} h1 {{font-size:36px;
        letter-spacing:-1px;line-height:1.2;margin:8px 0 20px}} h2 {{font-size:21px;margin:0 0 16px}}
        .eyebrow {{color:{t['palette'][0]};font-weight:700;letter-spacing:2px;font-size:12px}}
        .cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:28px 0}}
        .card,section {{background:{t['surface']};border:1px solid {t['grid']};border-radius:16px;padding:24px}}
        .card strong {{font-size:28px;display:block}} .card span {{opacity:.75;font-size:13px}}
        section {{margin-bottom:24px}} .scroll {{overflow-x:auto}} table {{border-collapse:collapse;
        width:100%;font-size:13px;text-align:left}} th,td {{padding:10px 14px;
        border-bottom:1px solid {t['grid']};white-space:nowrap}} th {{background:{t['background']};text-align:left}}
        td {{font-variant-numeric:tabular-nums}} img {{width:100%;height:auto;border-radius:10px}}
        .note,footer {{opacity:.72;font-size:13px}} nav {{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
        a {{color:{t['palette'][0]}}} details summary {{cursor:pointer;font-weight:600}}
        @media print {{main {{padding:0}} section,.card {{break-inside:avoid}} nav {{display:none}}}}
        """
        parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                 f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                 f'<title>{escape(self.title)}</title><style>{css}</style></head><body><main>',
                 '<div class="eyebrow">NAVILIB / DATA EXPLORER</div>', f'<h1>{escape(self.title)}</h1>',
                 '<p>Understand your data, inspect its quality, and choose the next step.</p>', '<div class="cards">']
        for key, value in self.summary.items():
            parts.append(f'<div class="card"><span>{escape(str(key))}</span><strong>{escape(str(value))}</strong></div>')
        parts.append('</div>')
        for note in self.notes:
            parts.append(f'<p class="note">{escape(note)}</p>')
        names = list(self.tables) + list(self.figures)
        parts.append('<nav>' + ''.join(f'<a href="#section-{i}">{escape(name)}</a>' for i, name in enumerate(names)) + '</nav>')
        for i, (name, table) in enumerate(self.tables.items()):
            parts.append(f'<section id="section-{i}"><h2>{escape(name)}</h2><div class="scroll">')
            if table.empty:
                parts.append('<p>No findings for this section.</p>')
            else:
                parts.append(table.head(max_rows).to_html(escape=True, border=0,
                             float_format=lambda x: f"{x:,.3f}", na_rep="—"))
            parts.append('</div>')
            if len(table) > max_rows:
                parts.append(f'<p class="note">Showing {max_rows} of {len(table)} rows; full results are in report.tables.</p>')
            parts.append('</section>')
        for i, (name, fig) in enumerate(self.figures.items(), len(self.tables)):
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            parts.append(f'<section id="section-{i}"><h2>{escape(name)}</h2>'
                         f'<img alt="{escape(name, quote=True)}" src="data:image/png;base64,{encoded}"></section>')
        parts.append('<footer>Generated by NaviLib · Descriptive findings are not evidence of causation.</footer></main></body></html>')
        html = '\n'.join(parts)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(html, encoding="utf-8")
        return html

    def _repr_html_(self):
        return self.to_html()


def create_report(df: pd.DataFrame, *, target: str | None = None,
                  title: str = "Dataset overview", theme: str | None = None,
                  include_plots: bool = True, sample_size: int = 5000,
                  max_columns: int = 12, random_state: int = 42) -> DataReport:
    """Build an actionable profile with full-data tables and bounded-size plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with unique column names. Input values are never changed.
    target : str, optional
        Target for label-quality checks; no model is fitted by this report.
    title : str, default 'Dataset overview'
        Heading for notebook and HTML output.
    theme : {'light', 'dark', 'paper'}, optional
        Temporary report theme; None preserves the current custom theme.
    include_plots : bool, default True
        Generate numeric distribution and missingness figures.
    sample_size : int, default 5000
        Maximum randomly sampled rows for plots. Tables use the full dataset.
    max_columns : int, default 12
        Maximum numeric columns plotted/profiled, and categorical columns profiled.
        The quality audit still checks every column.
    random_state : int, default 42
        Reproducible plot sampling seed.

    Returns
    -------
    DataReport
        Reusable tables, figures and ``to_html(path)`` export. Infinite numeric
        values are reported in the audit and treated as missing in profiles/plots.

    Examples
    --------
    >>> import NaviLib as nv
    >>> report = nv.create_report(pd.DataFrame({'x': [1., 2., 3.]}))
    >>> html = report.to_html()
    """
    from contextlib import nullcontext
    _check_frame(df)
    if any(not isinstance(x, int) or x < 1 for x in (sample_size, max_columns)):
        raise ValueError("sample_size and max_columns must be positive integers.")
    with theme_context(theme) if theme is not None else nullcontext():
        tables = {"Quality and next steps": audit_data(df, target=target)}
        num = list(df.select_dtypes(include="number").columns[:max_columns])
        cat = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns[:max_columns])
        clean = df.copy()
        if num:
            clean[num] = clean[num].replace([np.inf, -np.inf], np.nan)
            tables["Numeric profile"] = eda.describe_numeric(clean, columns=num)
        if cat:
            tables["Categorical profile"] = eda.describe_categorical(clean, columns=cat)
        figures, notes = {}, []
        if len(df) > sample_size:
            notes.append(f"Figures use a reproducible sample of {sample_size:,} rows. Tables use all {len(df):,} rows.")
        notes.append(f"Profiles and distribution plots show at most {max_columns} columns per type. Quality checks cover every column.")
        if include_plots and len(df):
            sample = clean.sample(min(len(clean), sample_size), random_state=random_state)
            plottable = [c for c in num if c != target and sample[c].notna().any()]
            if plottable:
                figures["Distributions"] = eda.plot_distribution(sample, columns=plottable,
                    kind="hist", max_cols=max_columns, log_x=False, show=False)
            if sample.isna().any().any():
                figures["Missingness"] = eda.plot_missing(sample, show=False)
            if target is not None and 0 < sample[target].nunique() <= 20:
                figures["Target distribution"] = eda.plot_categorical(sample, target, show=False)
        summary = {"Rows": f"{len(df):,}", "Columns": f"{len(df.columns):,}",
                   "Missing cells": f"{df.isna().to_numpy().mean() * 100:.1f}%" if df.size else "—",
                   "Memory": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"}
        return DataReport(title, summary, tables, figures, get_theme(), notes)
