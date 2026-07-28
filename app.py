import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="NFL Fantasy Player Explorer", layout="wide")

DATA_ROOT = Path(__file__).resolve().parent.parent / "NFL-Data" / "NFL-data-Players"
YEARS = [2022, 2021]
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DB", "DL", "LB"]
WEEKS = list(range(1, 18))


@st.cache_data
def load_season(year: int, position: str) -> pd.DataFrame:
    path = DATA_ROOT / str(year) / f"{position}_season.csv"
    return pd.read_csv(path)


@st.cache_data
def load_week(year: int, week: int, position: str) -> pd.DataFrame:
    path = DATA_ROOT / str(year) / str(week) / f"{position}.csv"
    return pd.read_csv(path)


st.title("🏈 NFL Fantasy Player Explorer")
st.caption("Browse 2021 & 2022 player stats from the NFL-Data archive")

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Year", YEARS)
with col2:
    position = st.selectbox("Position", POSITIONS)
with col3:
    period = st.selectbox("Period", ["Full Season"] + [f"Week {w}" for w in WEEKS])

if period == "Full Season":
    df = load_season(year, position)
else:
    week_num = int(period.split(" ")[1])
    df = load_week(year, week_num, position)

col4, col5 = st.columns([3, 1])
with col4:
    default_col = "TotalPoints" if "TotalPoints" in df.columns else df.columns[0]
    sort_col = st.selectbox("Sort by", df.columns, index=list(df.columns).index(default_col))
with col5:
    ascending = st.checkbox("Ascending", value=False)

df_sorted = df.sort_values(by=sort_col, ascending=ascending, na_position="last")

st.caption(f"{len(df_sorted)} players — {year} {period}")
st.dataframe(df_sorted, use_container_width=True, height=700, hide_index=True)
