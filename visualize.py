import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prepare_data import check_data_is_fresh, OUTPUT_CSV, STT_PATH, copy_files

if not check_data_is_fresh():
    print("Data out of date. Copying...")
    copy_files()
    print("Done")

st.set_page_config(page_title="Sleep & Time Dashboard", layout="wide")


def get_theme_config():
    """Detect Streamlit theme and return appropriate colors."""
    manual_theme = st.session_state.get("chart_theme", "Auto")

    if manual_theme == "Dark":
        is_dark = True
    elif manual_theme == "Light":
        is_dark = False
    else:
        is_dark = True
        try:
            theme = st.theme()
            if theme and theme.get("base") == "dark":
                is_dark = True
        except Exception:
            try:
                if st.get_option("theme.base") == "dark":
                    is_dark = True
            except Exception:
                pass

    return {
        "text": "#ffffff" if is_dark else "#000000",
        "text_secondary": "rgba(255,255,255,0.7)" if is_dark else "rgba(0,0,0,0.6)",
        "grid": "rgba(255,255,255,0.15)" if is_dark else "rgba(0,0,0,0.1)",
        "bg_transparent": "rgba(0,0,0,0)",
        "is_dark": is_dark,
        "zero_line": "rgba(255,255,255,0.5)" if is_dark else "rgba(0,0,0,0.5)",
    }


def apply_max_width_css(max_px: int):
    st.markdown(
        f"""
<style>
.block-container {{
    max-width: {max_px}px;
    margin-left: auto;
    margin-right: auto;
}}
div[data-testid="stPyplotFigure"] {{
    width: 100%;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")


def split_interval_by_day(start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    cur = start_dt
    while cur.date() < end_dt.date():
        next_midnight = pd.Timestamp(cur.date()) + pd.Timedelta(days=1)
        mins = (next_midnight - cur).total_seconds() / 60.0
        yield pd.Timestamp(cur.date()), mins
        cur = next_midnight
    mins = (end_dt - cur).total_seconds() / 60.0
    yield pd.Timestamp(cur.date()), mins


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()


def pre_post_shift(
    df: pd.DataFrame, cols: list[str], split_date: pd.Timestamp, window_days: int
):
    split_date = pd.Timestamp(split_date).floor("D")
    pre = df[
        (df.index >= split_date - pd.Timedelta(days=window_days))
        & (df.index < split_date)
    ]
    post = df[
        (df.index >= split_date)
        & (df.index < split_date + pd.Timedelta(days=window_days))
    ]
    pre_m = pre[cols].mean()
    post_m = post[cols].mean()
    delta = post_m - pre_m
    return pre_m, post_m, delta, len(pre), len(post)


@st.cache_data(show_spinner=False)
def load_sleep_csv(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    if "date" not in df.columns:
        raise ValueError("Sleep CSV must have a 'date' column.")
    df["date"] = to_datetime_safe(df["date"]).dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

    for c in ["total_sleep_m", "sleep_score", "hrv_ms", "rem_m", "awake_m"]:
        if c not in df.columns:
            df[c] = np.nan

    df["watch_sleep_h"] = df["total_sleep_m"] / 60.0
    return df.set_index("date")


@st.cache_data(show_spinner=False)
def load_time_csv(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    required_cols = ["activity name", "time started", "time ended", "categories"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Time CSV missing required column: {c}")
    df = df.rename(
        columns={
            "activity name": "activity_name",
            "time started": "time_started",
            "time ended": "time_ended",
            "record tags": "record_tags",
            "duration minutes": "duration_minutes",
        }
    )
    df["time_started"] = to_datetime_safe(df["time_started"])
    df["time_ended"] = to_datetime_safe(df["time_ended"])
    df = df.dropna(subset=["time_started", "time_ended"])
    df = df[df["time_ended"] > df["time_started"]].copy()
    df["activity_name"] = df["activity_name"].astype(str)
    df["categories"] = df["categories"].fillna("").astype(str)
    return df


@st.cache_data(show_spinner=False)
def build_daily_time_aggregates(tt: pd.DataFrame) -> pd.DataFrame:
    daily = defaultdict(lambda: defaultdict(float))
    buckets = {"Required work", "Beneficial", "Rest", "Sleep"}

    for row in tt.itertuples(index=False):
        s = row.time_started
        e = row.time_ended
        cats = str(row.categories).strip()
        cat_list = [c.strip() for c in cats.split(",") if c.strip()] or [
            "(uncategorized)"
        ]

        is_sleep_activity = str(row.activity_name).lower() == "sleep"
        has_sleep_cat = any(c == "Sleep" for c in cat_list)

        if is_sleep_activity or has_sleep_cat:
            use_cats = ["Sleep"]
        else:
            use_cats = [c for c in cat_list if c in buckets]

        if not use_cats:
            continue

        per_cat = 1.0 / len(use_cats)
        for d, mins in split_interval_by_day(s, e):
            for c in use_cats:
                daily[d][c] += mins * per_cat

    out = []
    for d in sorted(daily.keys()):
        rec = {"date": d}
        for c in buckets:
            rec[f"{c}_h"] = daily[d].get(c, 0.0) / 60.0
        out.append(rec)

    return pd.DataFrame(out).set_index("date").sort_index()


def weekday_metric_grid_plotly(
    df: pd.DataFrame,
    cols: list[str],
    good_high: dict[str, bool],
    title: str,
):
    """Interactive weekday heatmap with theme-aware colors."""
    if df.empty or not cols:
        st.caption("No data available for weekday grid.")
        return

    theme = get_theme_config()
    tmp = df.copy()
    tmp["weekday"] = tmp.index.day_name()
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    wk = tmp.groupby("weekday")[cols].mean().reindex(order)

    z_vals = []
    text_vals = []
    hover_texts = []

    for metric in cols:
        row = wk[metric]
        mn, mx = row.min(), row.max()
        if pd.notna(mn) and pd.notna(mx) and mx > mn:
            norm = (row - mn) / (mx - mn)
            if not good_high.get(metric, True):
                norm = 1 - norm
            z_row = norm.values.tolist()
        else:
            z_row = [np.nan] * len(order)

        text_row = []
        hover_row = []
        for day, val in zip(order, row.values):
            if pd.isna(val):
                text_row.append("—")
                hover_row.append(f"<b>{metric}</b><br>{day}: No data")
            else:
                if metric.endswith("_h") or metric == "watch_sleep_h":
                    val_str = f"{val:.2f}"
                elif metric == "hrv_ms":
                    val_str = f"{val:.0f}"
                elif metric == "sleep_score":
                    val_str = f"{val:.1f}"
                else:
                    val_str = f"{val:.0f}"
                text_row.append(val_str)
                goodness = (
                    "Higher=better" if good_high.get(metric, True) else "Lower=better"
                )
                hover_row.append(
                    f"<b>{metric}</b><br>{day}: {val_str}<br><i>({goodness})</i>"
                )

        z_vals.append(z_row)
        text_vals.append(text_row)
        hover_texts.append(hover_row)

    colorscale = [
        [0.0, "rgb(255, 255, 255)"],
        [0.5, "rgb(153, 217, 153)"],
        [1.0, "rgb(51, 179, 51)"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=[d[:3] for d in order],
            y=cols,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#000000"},
            colorscale=colorscale,
            showscale=True,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(
                    text="Relative<br>Score",
                    side="right",
                    font=dict(color=theme["text"]),
                ),
                tickmode="array",
                tickvals=[0, 0.5, 1],
                ticktext=["Low", "Med", "High"],
                tickfont=dict(color=theme["text"]),
                len=0.6,
                thickness=20,
                x=1.02,
                bgcolor=theme["bg_transparent"],
            ),
            hoverongaps=False,
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color=theme["text"], size=14)),
        paper_bgcolor=theme["bg_transparent"],
        plot_bgcolor=theme["bg_transparent"],
        font=dict(color=theme["text"]),
        xaxis=dict(
            side="top",
            color=theme["text"],
            showgrid=False,
            tickfont=dict(color=theme["text"], size=12),
        ),
        yaxis=dict(
            autorange="reversed",
            color=theme["text"],
            showgrid=False,
            tickfont=dict(color=theme["text"], size=12),
        ),
        height=120 + 45 * len(cols),
        margin=dict(l=140, r=80, t=80, b=40),
    )

    st.plotly_chart(fig, width="stretch")


def plot_sleep_dual_axis_plotly(df: pd.DataFrame, smooth_window: int, split_date=None):
    theme = get_theme_config()
    left_cols = ["sleep_score", "hrv_ms", "rem_m", "awake_m"]
    right_col = "watch_sleep_h"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors_left = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, c in enumerate(left_cols):
        if c in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=smooth_series(df[c], smooth_window),
                    name=c,
                    mode="lines",
                    line=dict(color=colors_left[i % len(colors_left)], width=2),
                    hovertemplate=f"<b>{c}</b>: %{{y:.1f}}<extra></extra>",
                ),
                secondary_y=False,
            )

    if right_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=smooth_series(df[right_col], smooth_window),
                name=right_col,
                mode="lines",
                line=dict(width=3, color="#9467bd"),
                hovertemplate=f"<b>{right_col}</b>: %{{y:.2f}}h<extra></extra>",
            ),
            secondary_y=True,
        )

    if split_date is not None:
        fig.add_vline(
            x=pd.Timestamp(split_date),
            line_dash="dash",
            line_color=theme["text_secondary"],
            opacity=0.7,
        )

    fig.update_layout(
        hovermode="x unified",
        title=dict(text="Sleep metrics over time", font=dict(color=theme["text"])),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=theme["text"]),
        ),
        paper_bgcolor=theme["bg_transparent"],
        plot_bgcolor=theme["bg_transparent"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=60, t=80, b=40),
        height=450,
        xaxis=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            showgrid=True,
            tickfont=dict(color=theme["text"]),
        ),
        yaxis=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            title=dict(
                text="Score / ms / minutes", font=dict(color=theme["text_secondary"])
            ),
            tickfont=dict(color=theme["text"]),
        ),
        yaxis2=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            title=dict(text="Hours", font=dict(color=theme["text_secondary"])),
            tickfont=dict(color=theme["text"]),
        ),
    )

    st.plotly_chart(fig, width="stretch")


def plot_diff_tracker_vs_watch_plotly(
    merged_df: pd.DataFrame, smooth_window: int, split_date=None
):
    theme = get_theme_config()
    if "Sleep_h" not in merged_df.columns or "watch_sleep_h" not in merged_df.columns:
        st.caption("Tracker sleep or watch sleep missing for the difference plot.")
        return

    diff_min = (merged_df["Sleep_h"] - merged_df["watch_sleep_h"]) * 60.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=smooth_series(diff_min, smooth_window),
            name="Difference",
            mode="lines",
            line=dict(color="#17becf", width=2),
            fill="tozeroy",
            fillcolor="rgba(23, 190, 207, 0.2)",
            hovertemplate="<b>Diff</b>: %{y:.1f} min<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1.5, line_color=theme["zero_line"])

    if split_date is not None:
        fig.add_vline(
            x=pd.Timestamp(split_date),
            line_dash="dash",
            line_color=theme["text_secondary"],
            opacity=0.7,
        )

    fig.update_layout(
        hovermode="x unified",
        title=dict(
            text="Tracked sleep vs watch sleep (difference in minutes)",
            font=dict(color=theme["text"]),
        ),
        xaxis=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            title=dict(text="Date", font=dict(color=theme["text_secondary"])),
            tickfont=dict(color=theme["text"]),
        ),
        yaxis=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            title=dict(
                text="Minutes (Tracker - Watch)",
                font=dict(color=theme["text_secondary"]),
            ),
            tickfont=dict(color=theme["text"]),
        ),
        paper_bgcolor=theme["bg_transparent"],
        plot_bgcolor=theme["bg_transparent"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=30, t=80, b=40),
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def plot_time_plotly(df: pd.DataFrame, smooth_window: int, split_date=None):
    theme = get_theme_config()
    fig = go.Figure()
    colors = {
        "Required work_h": "#e74c3c",
        "Beneficial_h": "#27ae60",
        "Rest_h": "#3498db",
    }

    for c in ["Required work_h", "Beneficial_h", "Rest_h"]:
        if c in df.columns:
            clean_name = c.replace("_h", "")
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=smooth_series(df[c], smooth_window),
                    name=clean_name,
                    mode="lines",
                    line=dict(width=2.5, color=colors.get(c, "#333")),
                    hovertemplate=f"<b>{clean_name}</b>: %{{y:.2f}}h<extra></extra>",
                )
            )

    if split_date is not None:
        fig.add_vline(
            x=pd.Timestamp(split_date),
            line_dash="dash",
            line_color=theme["text_secondary"],
            opacity=0.7,
        )

    fig.update_layout(
        hovermode="x unified",
        title=dict(
            text="Time categories (hours/day, smoothed)", font=dict(color=theme["text"])
        ),
        xaxis=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            title=dict(text="Date", font=dict(color=theme["text_secondary"])),
            tickfont=dict(color=theme["text"]),
        ),
        yaxis=dict(
            color=theme["text"],
            gridcolor=theme["grid"],
            title=dict(text="Hours/day", font=dict(color=theme["text_secondary"])),
            tickfont=dict(color=theme["text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=theme["text"]),
        ),
        paper_bgcolor=theme["bg_transparent"],
        plot_bgcolor=theme["bg_transparent"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=30, t=80, b=40),
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def plot_shift_delta_plotly(
    delta: pd.Series, pre: pd.Series, post: pd.Series, title: str, ylabel: str
):
    """Interactive bar chart showing pre/post shift with hover details."""
    theme = get_theme_config()

    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in delta.values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=delta.index,
            y=delta.values,
            marker=dict(color=colors, line=dict(color=theme["text"], width=0.5)),
            text=[f"{v:+.2f}" for v in delta.values],
            textposition="outside",
            textfont=dict(color=theme["text"], size=12),
            hovertemplate="<b>%{x}</b><br>"
            + "<b>Delta:</b> %{y:.3f}<br>"
            + "<b>Pre:</b> %{customdata[0]:.3f}<br>"
            + "<b>Post:</b> %{customdata[1]:.3f}<extra></extra>",
            customdata=np.column_stack((pre.values, post.values)),
        )
    )

    fig.add_hline(y=0, line_width=2, line_color=theme["zero_line"], layer="below")

    fig.update_layout(
        title=dict(text=title, font=dict(color=theme["text"])),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color=theme["text_secondary"])),
            color=theme["text"],
            gridcolor=theme["grid"],
            zeroline=True,
            zerolinecolor=theme["zero_line"],
            zerolinewidth=2,
            tickfont=dict(color=theme["text"]),
        ),
        xaxis=dict(
            tickangle=45,
            color=theme["text"],
            gridcolor=theme["grid"],
            tickfont=dict(color=theme["text"]),
        ),
        paper_bgcolor=theme["bg_transparent"],
        plot_bgcolor=theme["bg_transparent"],
        font=dict(color=theme["text"]),
        showlegend=False,
        height=400,
        margin=dict(l=60, r=30, t=80, b=80),
        bargap=0.3,
    )

    st.plotly_chart(fig, width="stretch")


st.sidebar.header("Layout")
max_width_px = st.sidebar.slider(
    "Dashboard max width (CSS px)", 700, 1600, 1200, step=50
)
apply_max_width_css(max_width_px)

st.sidebar.header("Chart Theme")
if "chart_theme" not in st.session_state:
    st.session_state["chart_theme"] = "Auto"

chart_theme = st.sidebar.radio(
    "Select chart text color",
    ["Auto", "Light", "Dark"],
    index=["Auto", "Light", "Dark"].index(st.session_state["chart_theme"]),
    help="If text appears black on dark background, select 'Dark' to override.",
)
st.session_state["chart_theme"] = chart_theme

st.sidebar.header("Data source")
use_local = st.sidebar.checkbox(
    "Load CSVs from local paths (instead of upload)", value=True
)

sleep_source = None
time_source = None

if use_local:
    sleep_source = st.sidebar.text_input("Local sleep CSV path", value=OUTPUT_CSV)
    time_source = st.sidebar.text_input("Local time CSV path", value=STT_PATH)
else:
    sleep_file = st.sidebar.file_uploader(
        "Sleep CSV (watch)", type=["csv"], key="sleep"
    )
    time_file = st.sidebar.file_uploader("Time tracker CSV", type=["csv"], key="time")
    sleep_source = sleep_file
    time_source = time_file

if not sleep_source or not time_source:
    st.info("Provide both CSV sources (local paths or uploads).")
    st.stop()

try:
    sleep_df = load_sleep_csv(sleep_source)
    time_df = load_time_csv(time_source)
except Exception as e:
    st.error(f"Failed to load CSVs: {e}")
    st.stop()

daily_time = build_daily_time_aggregates(time_df)
merged = sleep_df.join(daily_time, how="inner")

st.sidebar.header("Filters")
min_date = min(sleep_df.index.min(), daily_time.index.min())
max_date = max(sleep_df.index.max(), daily_time.index.max())

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
d0 = pd.Timestamp(date_range[0]).floor("D")
d1 = pd.Timestamp(date_range[1]).floor("D")

smooth = st.sidebar.slider("Smoothing window (days)", 1, 21, 11)

st.sidebar.header("Week view")
week_n_days = st.sidebar.slider("Use last N days for weekday averages", 14, 180, 56)

st.sidebar.header("Before/After analysis")
split_date = st.sidebar.date_input("Split date", value=max_date.date())
shift_window = st.sidebar.slider("Window size around split date (days)", 7, 56, 28)
split_ts = pd.Timestamp(split_date).floor("D")

sleep_f = sleep_df[(sleep_df.index >= d0) & (sleep_df.index <= d1)].copy()
time_f = daily_time[(daily_time.index >= d0) & (daily_time.index <= d1)].copy()
merged_f = merged[(merged.index >= d0) & (merged.index <= d1)].copy()

st.title("Sleep & Time Dashboard")

st.header("Sleep over time (dual axis)")
st.caption(
    "Hover to see all values at that date. Unified tooltip shows all metrics simultaneously."
)
plot_sleep_dual_axis_plotly(sleep_f, smooth_window=smooth, split_date=split_ts)

st.subheader("Tracked sleep vs watch sleep (difference)")
plot_diff_tracker_vs_watch_plotly(merged_f, smooth_window=smooth, split_date=split_ts)

st.header("Sleep weekday profile")
sleep_week_base = sleep_df[
    sleep_df.index >= (sleep_df.index.max() - pd.Timedelta(days=week_n_days))
].copy()
sleep_cols = [
    c
    for c in ["sleep_score", "watch_sleep_h", "hrv_ms", "rem_m", "awake_m"]
    if c in sleep_week_base.columns
]
sleep_good_high = {
    "sleep_score": True,
    "watch_sleep_h": True,
    "hrv_ms": True,
    "rem_m": True,
    "awake_m": False,
}
weekday_metric_grid_plotly(
    sleep_week_base,
    cols=sleep_cols,
    good_high=sleep_good_high,
    title=f"Sleep by weekday (last {week_n_days} days) — white→green indicates relative performance",
)

st.header("Sleep shift around selected date")
sleep_shift_cols = [
    c
    for c in ["sleep_score", "watch_sleep_h", "hrv_ms", "rem_m", "awake_m"]
    if c in sleep_df.columns
]
pre_m, post_m, delta, npre, npost = pre_post_shift(
    sleep_df, sleep_shift_cols, split_ts, shift_window
)

plot_shift_delta_plotly(delta, pre_m, post_m, "Sleep: post - pre (means)", "Delta")

st.header("Time over time")
st.caption("Hover to see exact hours for all categories at any date.")
plot_time_plotly(time_f, smooth_window=smooth, split_date=split_ts)

st.header("Time weekday profile")
time_week_base = daily_time[
    daily_time.index >= (daily_time.index.max() - pd.Timedelta(days=week_n_days))
].copy()
time_cols = [
    c
    for c in ["Required work_h", "Beneficial_h", "Rest_h"]
    if c in time_week_base.columns
]
time_good_high = {"Required work_h": True, "Beneficial_h": True, "Rest_h": True}
weekday_metric_grid_plotly(
    time_week_base,
    cols=time_cols,
    good_high=time_good_high,
    title=f"Time categories by weekday (last {week_n_days} days) — white→green indicates relative performance",
)

st.header("Time shift around selected date")
time_shift_cols = [
    c for c in ["Required work_h", "Beneficial_h", "Rest_h"] if c in daily_time.columns
]
pre_m2, post_m2, delta2, npre2, npost2 = pre_post_shift(
    daily_time, time_shift_cols, split_ts, shift_window
)

plot_shift_delta_plotly(
    delta2, pre_m2, post_m2, "Time: post - pre (means)", "Delta (hours/day)"
)

with st.expander("Notes / assumptions"):
    st.markdown(
        """
- **Theme Aware**: All charts automatically adapt to Streamlit's light/dark mode. Use the "Chart Theme" selector in the sidebar if auto-detection fails.
- **Interactive**: Hover over any chart to see exact values. Heatmaps show relative performance (white=low/worst, green=high/best).
- **Shift Analysis**: Bar charts show delta (post-pre) with green for positive changes, red for negative. Hover to see absolute pre/post values.
- **Data Handling**: Multi-category records split evenly; midnight-spanning records split across days.
"""
    )
