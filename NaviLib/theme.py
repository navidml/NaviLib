"""Consistent plotting and table themes without changing global rcParams."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

__all__ = ["set_theme", "get_theme", "theme_context", "available_themes", "style_table"]

_THEMES = {
    "light": dict(background="#F5F7FB", surface="#FFFFFF", text="#172B4D",
                  grid="#DCE3EE", palette=["#356AE6", "#E58B35", "#149E8E", "#D65378",
                                          "#8461C5", "#64748B", "#A77C35", "#40A9CF"],
                  cmap="viridis"),
    "dark": dict(background="#101827", surface="#182438", text="#E6EDF7",
                 grid="#34445D", palette=["#77A5FF", "#F4B36A", "#52D3B5", "#F58DAA",
                                         "#B89AF0", "#B6C6DC", "#DFCF86", "#72D5EF"],
                 cmap="viridis"),
    "paper": dict(background="#FFFFFF", surface="#FFFFFF", text="#202020",
                  grid="#DDDDDD", palette=["#0072B2", "#E69F00", "#009E73", "#D55E00",
                                          "#CC79A7", "#56B4E9", "#8B7B18", "#555555"],
                  cmap="cividis"),
}
_ACTIVE = ContextVar("navilib_theme", default="light")


def available_themes() -> list[str]:
    """Return built-in theme names: light, dark and paper.

    Returns
    -------
    pandas.DataFrame
        Structured results with named columns; see the measures and
        interpretation described above.
    """
    return list(_THEMES)


def _resolve(name, palette=None, font_scale=1.0, rc=None):
    if name not in _THEMES:
        raise ValueError(f"Unknown theme {name!r}; choose from {available_themes()}.")
    from matplotlib import colors, rc_context
    if not isinstance(font_scale, (int, float)) or not 0 < font_scale < 10:
        raise ValueError("font_scale must be between 0 and 10.")
    result = deepcopy(_THEMES[name])
    result.update(name=name, font_scale=font_scale, rc=dict(rc or {}))
    if palette is not None:
        if isinstance(palette, str) or len(palette) < 2:
            raise ValueError("palette must be a sequence of at least two colors.")
        result["palette"] = [colors.to_hex(c) for c in palette]
    with rc_context(result["rc"]):
        pass  # validate overrides before changing the active setting
    return result


def set_theme(name: str = "light", *, palette=None, font_scale: float = 1.0,
              rc: dict | None = None) -> dict:
    """Choose the theme for subsequent NaviLib figures and styled tables.

    Parameters
    ----------
    name : {'light', 'dark', 'paper'}, default 'light'
        Light dashboard, dark dashboard, or print-friendly colors.
    palette : sequence of matplotlib colors, optional
        At least two colors; replaces the categorical color cycle.
    font_scale : float, default 1.0
        Multiplier for chart label and title sizes.
    rc : dict, optional
        Matplotlib rcParams overrides, applied only while drawing.

    Returns
    -------
    dict
        A copy of the selected configuration. Existing figures are unchanged.

    Examples
    --------
    >>> import NaviLib as nv
    >>> config = nv.set_theme('dark', font_scale=1.1)
    """
    _ACTIVE.set(_resolve(name, palette, font_scale, rc))
    return get_theme()


def get_theme() -> dict:
    """Return an independent copy of the current NaviLib theme configuration.

    Returns
    -------
    pandas.DataFrame
        Structured results with named columns; see the measures and
        interpretation described above.
    """
    current = _ACTIVE.get()
    return _resolve(current) if isinstance(current, str) else deepcopy(current)


@contextmanager
def theme_context(name: str = "light", *, palette=None, font_scale: float = 1.0,
                  rc: dict | None = None):
    """Temporarily choose a theme; restore it even if plotting raises an error.

    Parameters match :func:`set_theme`. Nested contexts restore their parent's
    settings. Yields a copy of the temporary configuration.

    Parameters
    ----------
    name : str, default 'light'
        Built-in theme name: light, dark or paper.
    palette : optional, default None
        Sequence of at least two matplotlib-compatible colors replacing the
        categorical cycle.
    font_scale : float, default 1.0
        Positive multiplier for figure title and label sizes.
    rc : dict | None, default None
        Optional matplotlib rcParams overrides, scoped to NaviLib plotting
        calls.

    Returns
    -------
    context manager
        Yields the temporary theme configuration and restores the previous one
        on exit.
    """
    token = _ACTIVE.set(_resolve(name, palette, font_scale, rc))
    try:
        yield get_theme()
    finally:
        _ACTIVE.reset(token)


@contextmanager
def _plot_context():
    import matplotlib as mpl
    from cycler import cycler
    t = get_theme()
    scale = t["font_scale"]
    params = {
        "figure.facecolor": t["background"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["background"], "text.color": t["text"],
        "axes.labelcolor": t["text"], "axes.edgecolor": t["grid"],
        "xtick.color": t["text"], "ytick.color": t["text"],
        "grid.color": t["grid"], "grid.alpha": .55, "axes.grid": True,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10 * scale, "axes.titlesize": 13 * scale,
        "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlepad": 14,
        "axes.labelsize": 10 * scale, "legend.frameon": False,
        "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 180,
        "axes.prop_cycle": cycler(color=t["palette"]), "image.cmap": t["cmap"],
    }
    params.update(t["rc"])
    with mpl.rc_context(params):
        yield


class _Palette:
    """Resolve colors at use time, including in modules that imported PALETTE."""
    def __getitem__(self, index):
        values = get_theme()["palette"]
        return values[index] if isinstance(index, slice) else values[index % len(values)]

    def __iter__(self):
        return iter(get_theme()["palette"])

    def __len__(self):
        return len(get_theme()["palette"])


def style_table(df, *, precision: int = 3, caption: str | None = None):
    """Return a themed pandas Styler without modifying data or rounding values.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to display in a notebook or export with ``.to_html()``.
    precision : int, default 3
        Display precision for floating-point values.
    caption : str, optional
        Plain-text table caption, escaped before HTML rendering.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled view; requires the ``notebook`` extra (Jinja2).
    """
    from html import escape
    t = get_theme()
    result = (df.style.format(precision=precision, na_rep="—", escape="html")
              .format_index(escape="html", axis=0).format_index(escape="html", axis=1)
              .set_properties(**{"background-color": t["surface"], "color": t["text"],
                                 "padding": "10px 14px", "border-bottom": f"1px solid {t['grid']}"})
              .set_table_styles([
                  {"selector": "th", "props": [("background", t["background"]),
                     ("color", t["text"]), ("padding", "12px 14px"), ("text-align", "left")]},
                  {"selector": "caption", "props": [("color", t["text"]),
                     ("font-size", "18px"), ("padding", "14px"), ("text-align", "left")]},
              ]))
    return result.set_caption(escape(caption)) if caption is not None else result
