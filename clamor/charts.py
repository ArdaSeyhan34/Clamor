"""Plotly figures shared by the HTML report and the Streamlit app.

Colors follow a validated categorical palette (colorblind-safe for adjacent series in
light and dark mode) and a separate, reserved status palette for verdicts. Identity is
never carried by color alone: every multi-series chart has a legend and hover labels,
and every chart has a table next to it in the report and the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}
INK_2 = {"light": "#52514e", "dark": "#c3c2b7"}
MUTED = "#898781"
GRID = {"light": "#e1e0d9", "dark": "#2c2c2a"}
STATUS = {"good": "#0ca30c", "critical": "#d03b3b", "neutral": MUTED}
VERDICT_STYLE = {  # verdict -> (status color, icon); icon + label, never color alone
    "Resolved": ("good", "✓"),
    "Improved": ("good", "↘"),
    "Worse": ("critical", "▲"),
    "No detectable change": ("neutral", "–"),
    "Inconclusive": ("neutral", "?"),
}
COMPONENTS = ["Reach", "Revenue", "Severity", "Momentum"]


def _layout(fig: go.Figure, mode: str, height: int, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 20, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "size": 13,
            "color": INK_2[mode],
        },
        hoverlabel={"font": {"family": "system-ui, sans-serif"}},
        legend=kwargs.pop(
            "legend", {"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": None}
        ),
        **kwargs,
    )
    fig.update_xaxes(
        gridcolor=GRID[mode],
        zerolinecolor=GRID[mode],
        linecolor=GRID[mode],
        tickfont={"color": MUTED},
    )
    fig.update_yaxes(
        gridcolor=GRID[mode],
        zerolinecolor=GRID[mode],
        linecolor=GRID[mode],
        tickfont={"color": INK_2[mode]},
    )
    return fig


def _short(text: str, n: int = 38) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def priority_chart(
    scored: pd.DataFrame, parts: pd.DataFrame, top_n: int = 12, mode: str = "light"
) -> go.Figure:
    """Stacked horizontal bars: how many points each signal contributes to the score."""
    road = scored[scored["score"].notna()].head(top_n)
    parts = parts.loc[road.index]
    labels = [f"{_short(n)}  " for n in road["name"]]
    fig = go.Figure()
    for comp, color in zip(COMPONENTS, SERIES[mode], strict=True):
        fig.add_bar(
            y=labels,
            x=parts[comp],
            name=comp,
            orientation="h",
            marker={"color": color, "line": {"color": SURFACE[mode], "width": 2}},
            customdata=np.stack([road["theme_id"], road["score"]], axis=1),
            hovertemplate=f"<b>%{{y}}</b><br>{comp}: %{{x:.1f}} pts"
            "<br>Total score: %{customdata[1]:.1f}<extra>%{customdata[0]}</extra>",
        )
    for label, score in zip(labels, road["score"], strict=True):
        fig.add_annotation(
            x=score,
            y=label,
            text=f"{score:.0f}",
            showarrow=False,
            xanchor="left",
            xshift=6,
            font={"color": INK[mode], "size": 12},
        )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Priority score (points by signal)", range=[0, 105])
    return _layout(
        fig,
        mode,
        height=90 + 30 * len(road),
        barmode="stack",
        bargap=0.35,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "traceorder": "normal"},
    )


def rank_shift_chart(scored: pd.DataFrame, top_n: int = 10, mode: str = "light") -> go.Figure:
    """Slope chart: rank by raw mention count vs rank by Clamor's priority score."""
    road = scored[scored["score"].notna()]
    road = road[(road["rank"] <= top_n) | (road["vote_rank"] <= top_n)]
    movers = road.reindex(road["rank_shift"].abs().sort_values(ascending=False).index).head(4)
    fig = go.Figure()
    for tid, r in road.iterrows():
        shift = r["rank_shift"]
        highlighted = tid in movers.index and shift != 0
        color = (SERIES[mode][0] if shift > 0 else STATUS["critical"]) if highlighted else MUTED
        fig.add_scatter(
            x=[0, 1],
            y=[r["vote_rank"], r["rank"]],
            mode="lines+markers",
            line={"color": color, "width": 2 if highlighted else 1},
            marker={"size": 9, "color": color, "line": {"color": SURFACE[mode], "width": 2}},
            opacity=1 if highlighted else 0.55,
            showlegend=False,
            hovertemplate=f"<b>{r['name']}</b><br>By mentions: #{int(r['vote_rank'])}"
            f"<br>By priority: #{int(r['rank'])}<extra></extra>",
        )
        weight = "<b>{}</b>" if highlighted else "{}"
        fig.add_annotation(
            x=1,
            y=r["rank"],
            text=weight.format(_short(r["name"], 30)),
            xanchor="left",
            xshift=10,
            showarrow=False,
            font={"color": INK[mode] if highlighted else INK_2[mode], "size": 12},
        )
        fig.add_annotation(
            x=0,
            y=r["vote_rank"],
            text=weight.format(_short(r["name"], 30)),
            xanchor="right",
            xshift=-10,
            showarrow=False,
            font={"color": INK[mode] if highlighted else INK_2[mode], "size": 12},
        )
    n = int(max(road["rank"].max(), road["vote_rank"].max()))
    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=["Rank by mention count", "Rank by Clamor priority"],
        range=[-0.9, 1.9],
        showgrid=False,
        side="top",
    )
    fig.update_yaxes(autorange="reversed", range=[n + 0.5, 0.5], showgrid=False, visible=False)
    return _layout(fig, mode, height=80 + 30 * n)


def timeline_chart(
    weekly: pd.DataFrame,
    theme_names: dict[str, str],
    theme_ids: list[str],
    releases: pd.DataFrame | None = None,
    mode: str = "light",
    normalize: bool = True,
) -> go.Figure:
    """Weekly mentions for up to four themes, with release dates marked."""
    fig = go.Figure()
    total = weekly["__total__"].replace(0, np.nan)
    for tid, color in zip(theme_ids[:4], SERIES[mode], strict=False):
        if tid not in weekly:
            continue
        y = weekly[tid] / total * 100 if normalize else weekly[tid]
        fig.add_scatter(
            x=weekly.index,
            y=y,
            mode="lines",
            name=_short(theme_names.get(tid, tid), 34),
            line={"color": color, "width": 2, "shape": "spline", "smoothing": 0.4},
            hovertemplate="%{y:.1f}"
            + ("% of feedback" if normalize else " mentions")
            + "<extra>%{fullData.name}</extra>",
        )
    if releases is not None and len(releases):
        for _, r in releases.iterrows():
            fig.add_vline(x=pd.Timestamp(r["date"]), line={"color": MUTED, "width": 1})
            fig.add_annotation(
                x=pd.Timestamp(r["date"]),
                y=1,
                yref="paper",
                text=r["version"],
                showarrow=False,
                yanchor="bottom",
                font={"color": MUTED, "size": 11},
            )
    fig.update_yaxes(
        title="% of all feedback" if normalize else "mentions per week", rangemode="tozero"
    )
    fig.update_layout(hovermode="x unified")
    return _layout(fig, mode, height=380, legend={"orientation": "h", "y": -0.18, "x": 0})


def release_chart(
    radar: pd.DataFrame, theme_names: dict[str, str], mode: str = "light"
) -> go.Figure:
    """Rate ratio after vs before each release with a 95% interval (log scale)."""
    data = radar[radar["rate_ratio"].notna()].copy() if "rate_ratio" in radar else radar.iloc[:0]
    fig = go.Figure()
    if data.empty:
        return _layout(fig, mode, height=200)
    data["label"] = [
        f"{VERDICT_STYLE.get(v, ('neutral', ''))[1]} {ver} · {_short(theme_names.get(t, t), 28)}"
        for v, ver, t in zip(data["verdict"], data["version"], data["theme_id"], strict=True)
    ]
    for verdict, g in data.groupby("verdict", sort=False):
        color = STATUS[VERDICT_STYLE.get(verdict, ("neutral", ""))[0]]
        fig.add_scatter(
            x=g["rate_ratio"],
            y=g["label"],
            mode="markers",
            name=verdict,
            marker={"size": 11, "color": color, "line": {"color": SURFACE[mode], "width": 2}},
            error_x={
                "type": "data",
                "symmetric": False,
                "color": color,
                "thickness": 2,
                "array": (g["ci_high"].clip(upper=50) - g["rate_ratio"]).to_numpy(),
                "arrayminus": (g["rate_ratio"] - g["ci_low"].clip(lower=0.02)).to_numpy(),
            },
            customdata=np.stack(
                [g["pre_mentions"], g["post_mentions"], g["ci_low"], g["ci_high"]], axis=1
            ),
            hovertemplate="<b>%{y}</b><br>Rate after / before: x%{x:.2f}"
            "<br>95% CI x%{customdata[2]:.2f} - x%{customdata[3]:.2f}"
            "<br>Mentions before %{customdata[0]}, after %{customdata[1]}"
            f"<extra>{verdict}</extra>",
        )
    fig.add_vline(x=1, line={"color": MUTED, "width": 1})
    fig.update_xaxes(
        type="log",
        title="Mention rate after vs before the release (log scale)",
        tickvals=[0.1, 0.25, 0.5, 1, 2, 4, 10, 25],
        ticktext=["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "10x", "25x"],
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, mode, height=110 + 42 * len(data))
