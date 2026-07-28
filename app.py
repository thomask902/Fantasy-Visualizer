import datetime

import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import kurtosis, skew

st.set_page_config(page_title="NFL Fantasy Player Explorer", layout="wide")

FIRST_SEASON = 1999
TODAY = datetime.date.today()
# The NFL season named e.g. "2025" runs Sep 2025 - Feb 2026, so before September
# the most recently *completed* season is still last calendar year's.
LATEST_SEASON = TODAY.year if TODAY.month >= 9 else TODAY.year - 1
YEARS = list(range(LATEST_SEASON, FIRST_SEASON - 1, -1))

FANTASY_COLS = ["fantasy_points", "fantasy_points_half_ppr", "fantasy_points_ppr"]
SCORING_MODES = {
    "Standard": "fantasy_points",
    "Half PPR": "fantasy_points_half_ppr",
    "PPR": "fantasy_points_ppr",
}
BOOM_OR_BUST_THRESHOLD = 0.555
BUST_THRESHOLD = 5.0
CHART_BLUE = "#2a78d6"
MIN_BIN_WIDTH = 1.0
MAX_BIN_WIDTH = 3.0

# Skill positions with meaningful fantasy scoring - nflverse's fantasy_points
# formula doesn't cover O-line, IDP, or kicking/punting, so those positions are
# all-zero and not useful to search for here.
VISUALIZER_POSITIONS = ["QB", "RB", "WR", "TE", "FB"]

PASSING_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "sack_yards_lost", "sack_fumbles", "sack_fumbles_lost",
    "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
    "passing_10", "passing_16", "passing_20", "passing_40",
    "passing_epa", "passing_cpoe", "passing_2pt_conversions", "pacr",
]
RUSHING_COLS = [
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
    "rushing_fumbles_lost", "rushing_first_downs",
    "rushing_10", "rushing_12", "rushing_20", "rushing_40",
    "rushing_epa", "rushing_2pt_conversions",
]
RECEIVING_COLS = [
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "receiving_fumbles", "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs",
    "receiving_10", "receiving_16", "receiving_20", "receiving_40",
    "receiving_epa", "receiving_2pt_conversions", "racr",
    "target_share", "air_yards_share", "wopr",
]
MISC_COLS = ["special_teams_tds"]
# Pure clutter, no analytical value - drop when building a display table.
DROP_COLS_EARLY = ["player_id", "player_name", "position_group", "headshot_url", "game_id"]

# Stat-category order per position, mirroring how stat sites lay out a player's page.
POSITION_STAT_ORDER = {
    "QB": [PASSING_COLS, RUSHING_COLS, RECEIVING_COLS],
    "RB": [RUSHING_COLS, RECEIVING_COLS, PASSING_COLS],
    "FB": [RUSHING_COLS, RECEIVING_COLS, PASSING_COLS],
    "WR": [RECEIVING_COLS, RUSHING_COLS, PASSING_COLS],
    "TE": [RECEIVING_COLS, RUSHING_COLS, PASSING_COLS],
}
DEFAULT_STAT_ORDER = [RUSHING_COLS, RECEIVING_COLS, PASSING_COLS]

ACRONYMS = {
    "ppr": "PPR", "epa": "EPA", "pacr": "PACR", "racr": "RACR", "wopr": "WOPR",
    "td": "TD", "tds": "TDs", "2pt": "2PT", "cpoe": "CPOE",
}


def prettify(col: str) -> str:
    words = [ACRONYMS.get(w.lower(), w.capitalize()) for w in col.split("_")]
    return " ".join(words)


@st.cache_data(ttl=3600, show_spinner="Fetching data from nflverse...")
def load_data(year: int, summary_level: str) -> pd.DataFrame:
    df = nfl.load_player_stats([year], summary_level=summary_level).to_pandas()
    df = df.rename(columns={"team": "recent_team"})  # "week" level names it differently than "reg"/"reg+post"
    df["fantasy_points_half_ppr"] = df["fantasy_points"] + 0.5 * df["receptions"].fillna(0)
    return df


@st.cache_data(ttl=3600, show_spinner="Loading all-time player index...")
def load_alltime_index() -> pd.DataFrame:
    df = nfl.load_player_stats(True, summary_level="reg+post").to_pandas()
    return df[["player_id", "player_display_name", "position", "recent_team", "season"]]


@st.cache_data(ttl=86400, show_spinner=False)
def load_team_logos() -> dict:
    teams = nfl.load_teams().to_pandas()
    return dict(zip(teams["team_abbr"], teams["team_logo_espn"]))


@st.cache_data(ttl=3600, show_spinner=False)
def build_player_options(idx: pd.DataFrame) -> pd.DataFrame:
    idx = idx[idx["position"].isin(VISUALIZER_POSITIONS)]
    latest = idx.sort_values("season").groupby("player_id", as_index=False).last()
    dupe_counts = latest.groupby(["player_display_name", "position"])["player_id"].transform("count")
    needs_team = dupe_counts > 1
    latest["label"] = latest["player_display_name"] + " (" + latest["position"] + ")"
    latest.loc[needs_team, "label"] = (
        latest.loc[needs_team, "player_display_name"] + " (" + latest.loc[needs_team, "position"]
        + ", " + latest.loc[needs_team, "recent_team"] + ")"
    )
    return latest[["player_id", "player_display_name", "label", "position", "recent_team"]].sort_values("label")


def build_display(df: pd.DataFrame, position: str) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS_EARLY if c in df.columns])
    lead_cols = [c for c in ["player_display_name", "position", "recent_team", "opponent_team", "week", "games"] if c in df.columns]
    fantasy_cols = [c for c in FANTASY_COLS if c in df.columns]
    stat_groups = POSITION_STAT_ORDER.get(position, DEFAULT_STAT_ORDER)
    stat_cols = [c for group in stat_groups for c in group if c in df.columns]
    stat_cols += [c for c in MISC_COLS if c in df.columns]
    ordered = lead_cols + fantasy_cols + stat_cols
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]
    return df.rename(columns={c: prettify(c) for c in df.columns})


def load_or_stop(year: int, summary_level: str) -> pd.DataFrame:
    try:
        return load_data(year, summary_level)
    except Exception as e:
        if "404" in str(e):
            st.error(
                f"nflverse hasn't published {year} season stats yet. Try the most "
                f"recent year with data, e.g. {LATEST_SEASON - 1}."
            )
        else:
            st.error(f"Couldn't load {year} data from nflverse: {e}")
        st.stop()


def fmt(value) -> str:
    return "N/A" if value is None or np.isnan(value) else f"{value:.1f}"


def bimodality_coefficient(values: np.ndarray):
    n = len(values)
    if n <= 3 or np.std(values) == 0:
        return None
    g1 = skew(values, bias=False)
    g2 = kurtosis(values, fisher=True, bias=False)
    denom = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if not denom:
        return None
    bc = (g1 ** 2 + 1) / denom
    return None if np.isnan(bc) else bc


def histogram_bin_width(values: np.ndarray, iqr: float) -> float:
    """Freedman-Diaconis-style bin width (halved to fit the 1-3 point range for
    typical NFL sample sizes), clamped to 1-3 points."""
    n = len(values)
    if n < 2 or iqr <= 0:
        return MIN_BIN_WIDTH
    width = iqr / (n ** (1 / 3))
    return min(MAX_BIN_WIDTH, max(MIN_BIN_WIDTH, width))


def render_data_tab():
    st.caption("Live player stats streamed from nflverse (via nflreadpy)")

    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Year", YEARS, key="data_year")
    with col2:
        include_playoffs = st.checkbox("Include playoffs", value=False, key="data_playoffs")

    weekly_df = load_or_stop(year, "week")
    if not include_playoffs:
        weekly_df = weekly_df[weekly_df["season_type"] == "REG"]

    if weekly_df.empty:
        st.warning(f"No data available for {year} yet.")
        st.stop()

    position_counts = weekly_df["position"].value_counts()
    positions = position_counts.index.tolist()
    default_pos_index = positions.index("QB") if "QB" in positions else 0

    col3, col4 = st.columns(2)
    with col3:
        position = st.selectbox("Position", positions, index=default_pos_index, key="data_position")
    with col4:
        weeks = sorted(weekly_df.loc[weekly_df["position"] == position, "week"].unique())
        period = st.selectbox("Period", ["Full Season"] + [f"Week {w}" for w in weeks], key="data_period")

    if period == "Full Season":
        # Season totals come pre-aggregated from nflverse (not summed from weekly rows here),
        # since rate stats like target_share can't just be added across weeks.
        season_level = "reg+post" if include_playoffs else "reg"
        season_df = load_or_stop(year, season_level)
        display_df = season_df[season_df["position"] == position]
    else:
        week_num = int(period.split(" ")[1])
        pos_df = weekly_df[weekly_df["position"] == position]
        display_df = pos_df[pos_df["week"] == week_num].drop(columns=["season_type"])

    display_df = build_display(display_df, position)

    col5, col6 = st.columns([3, 1])
    with col5:
        default_sort = prettify("fantasy_points_ppr")
        sort_col = st.selectbox("Sort by", display_df.columns, index=list(display_df.columns).index(default_sort), key="data_sort")
    with col6:
        ascending = st.checkbox("Ascending", value=False, key="data_ascending")

    display_df = display_df.sort_values(by=sort_col, ascending=ascending, na_position="last")

    st.caption(f"{len(display_df)} players — {year} {period}{' (incl. playoffs)' if include_playoffs else ''}")
    st.dataframe(display_df, use_container_width=True, height=700, hide_index=True)


def render_visualizer_tab():
    st.caption("Single-game fantasy performances, career or single season")

    idx = load_alltime_index()
    options = build_player_options(idx)
    logos = load_team_logos()

    selected_label = st.selectbox("Search for a player", options["label"].tolist(), key="viz_player")
    player_row = options[options["label"] == selected_label].iloc[0]
    player_id = player_row["player_id"]

    seasons = sorted(idx.loc[idx["player_id"] == player_id, "season"].unique().tolist(), reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        scoring_label = st.selectbox("Scoring", list(SCORING_MODES.keys()), key="viz_scoring")
    with col2:
        timeframe = st.selectbox("Timeframe", ["Career"] + [str(s) for s in seasons], key="viz_timeframe")

    scoring_col = SCORING_MODES[scoring_label]
    years_to_fetch = seasons if timeframe == "Career" else [int(timeframe)]

    with st.spinner(f"Loading {len(years_to_fetch)} season(s) of game logs..."):
        frames = []
        for y in years_to_fetch:
            wk = load_data(y, "week")
            wk = wk[wk["season_type"] == "REG"]
            frames.append(wk[wk["player_id"] == player_id])
        games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if games.empty:
        st.warning("No game logs found for this player in that timeframe.")
        return

    team_season = seasons[0] if timeframe == "Career" else int(timeframe)
    team_match = idx.loc[(idx["player_id"] == player_id) & (idx["season"] == team_season), "recent_team"]
    team_abbr = team_match.iloc[0] if not team_match.empty else None
    logo_url = logos.get(team_abbr)

    header_logo, header_name = st.columns([1, 8])
    with header_logo:
        if logo_url:
            st.image(logo_url, width=70)
    with header_name:
        st.subheader(f"{player_row['player_display_name']} ({player_row['position']})")

    values = games[scoring_col].dropna().to_numpy()
    n = len(values)
    avg = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1)) if n > 1 else None
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    q05, q95 = np.percentile(values, [5, 95])
    bc = bimodality_coefficient(values)
    bust_count = int((values < BUST_THRESHOLD).sum())
    bust_pct = 100 * bust_count / n

    left, right = st.columns(2)

    with left:
        bin_width = histogram_bin_width(values, iqr)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=values,
            xbins=dict(size=bin_width),
            marker_color=CHART_BLUE,
            hovertemplate="%{x} pts<br>%{y} game(s)<extra></extra>",
            name="",
        ))
        fig.add_vline(
            x=avg, line_dash="dot", line_width=2, line_color="#52514e",
            annotation_text=f"Avg: {avg:.1f}", annotation_position="top",
        )
        fig.update_layout(
            xaxis_title=f"{scoring_label} Fantasy Points (per game)",
            yaxis_title="Games",
            bargap=0.08,
            showlegend=False,
            margin=dict(t=60, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    with right:
        avg_std_display = f"{avg:.1f} ± {std:.1f}" if std is not None else fmt(avg)

        c1, c2, c3 = st.columns(3)
        c1.metric("Games", str(n))
        c2.metric("Average ± Std Dev", avg_std_display)
        c3.metric("Median", fmt(median))

        c4, c5 = st.columns(2)
        c4.metric("50% of Games Within", f"[{q1:.1f}-{q3:.1f}]")
        c5.metric("90% of Games Within", f"[{q05:.1f}-{q95:.1f}]")

        if bc is None:
            bc_display = "N/A"
        else:
            bc_display = f"{bc:.3f}" + (" (Boom or Bust)" if bc > BOOM_OR_BUST_THRESHOLD else "")

        c6, c7, c8 = st.columns(3)
        c6.metric(f"Bust Games (<{BUST_THRESHOLD:.0f} pts)", f"{bust_count}/{n}")
        c7.metric("Bust Game %", f"{bust_pct:.1f}%")
        c8.metric("Boom or Bust Coefficient", bc_display)

        st.caption(f"{n} game(s) — {timeframe}")


st.title("🏈 NFL Fantasy Player Explorer")

visualizer_tab, data_tab = st.tabs(["Visualizer", "Data"])
with visualizer_tab:
    render_visualizer_tab()
with data_tab:
    render_data_tab()
