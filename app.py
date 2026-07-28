import datetime

import nflreadpy as nfl
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Fantasy Player Explorer", layout="wide")

FIRST_SEASON = 1999
TODAY = datetime.date.today()
# The NFL season named e.g. "2025" runs Sep 2025 - Feb 2026, so before September
# the most recently *completed* season is still last calendar year's.
LATEST_SEASON = TODAY.year if TODAY.month >= 9 else TODAY.year - 1
YEARS = list(range(LATEST_SEASON, FIRST_SEASON - 1, -1))

FANTASY_COLS = ["fantasy_points", "fantasy_points_half_ppr", "fantasy_points_ppr"]

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
# Pure clutter, no analytical value - drop as soon as data is loaded.
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
    return df.drop(columns=[c for c in DROP_COLS_EARLY if c in df.columns])


def build_display(df: pd.DataFrame, position: str) -> pd.DataFrame:
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


st.title("🏈 NFL Fantasy Player Explorer")
st.caption("Live player stats streamed from nflverse (via nflreadpy)")

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("Year", YEARS)
with col2:
    include_playoffs = st.checkbox("Include playoffs", value=False)

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
    position = st.selectbox("Position", positions, index=default_pos_index)
with col4:
    weeks = sorted(weekly_df.loc[weekly_df["position"] == position, "week"].unique())
    period = st.selectbox("Period", ["Full Season"] + [f"Week {w}" for w in weeks])

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
    sort_col = st.selectbox("Sort by", display_df.columns, index=list(display_df.columns).index(default_sort))
with col6:
    ascending = st.checkbox("Ascending", value=False)

display_df = display_df.sort_values(by=sort_col, ascending=ascending, na_position="last")

st.caption(f"{len(display_df)} players — {year} {period}{' (incl. playoffs)' if include_playoffs else ''}")
st.dataframe(display_df, use_container_width=True, height=700, hide_index=True)
