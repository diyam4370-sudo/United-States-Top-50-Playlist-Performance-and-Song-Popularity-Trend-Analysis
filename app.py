import pandas as pd
import plotly.express as px
import streamlit as st
from data_prep import load_clean, song_table, artist_table, movements

st.set_page_config(page_title="US Top 50 Analytics", layout="wide")


@st.cache_data
def get_data():
    return load_clean()


df_all, report = get_data()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(dates) != 2:
    st.stop()
rank_lo, rank_hi = st.sidebar.slider("Rank range", 1, 50, (1, 50))
album_types = st.sidebar.multiselect("Album type", sorted(df_all["album_type"].unique()),
                                     default=sorted(df_all["album_type"].unique()))
artists = st.sidebar.multiselect("Artist", sorted(df_all["primary_artist"].unique()))
songs = st.sidebar.multiselect("Song", sorted(df_all["song_id"].unique()))

df = df_all[df_all["date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
            & df_all["position"].between(rank_lo, rank_hi)
            & df_all["album_type"].isin(album_types)]
if artists:
    df = df[df["primary_artist"].isin(artists)]
if songs:
    df = df[df["song_id"].isin(songs)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.title("United States Top 50 – Playlist Performance & Popularity Trends")

# ---------------- KPIs ----------------
songs_t = song_table(df)
art_t = artist_table(df)
c = st.columns(5)
c[0].metric("Unique songs", f"{songs_t.shape[0]:,}")
c[1].metric("Median days on chart", f"{songs_t['days_on_chart'].median():.0f}")
c[2].metric("Avg rank volatility (σ)", f"{songs_t['rank_volatility'].mean():.1f}")
c[3].metric("Avg popularity", f"{df['popularity'].mean():.1f}")
c[4].metric("Explicit share", f"{df['is_explicit'].mean() * 100:.1f}%")

tabs = st.tabs(["Timeline explorer", "Song trends", "Artist dominance", "Popularity vs rank",
                "Content attributes", "Movers", "Data quality"])

# ---------------- Timeline explorer ----------------
with tabs[0]:
    day = st.select_slider("Snapshot date", options=sorted(df["date"].dt.date.unique()),
                           value=sorted(df["date"].dt.date.unique())[-1])
    snap = df[df["date"].dt.date == day].sort_values("position")
    st.dataframe(snap[["position", "song", "artist", "popularity", "album_type", "is_explicit"]],
                 use_container_width=True, hide_index=True)
    st.caption("Chart entries per day (rows in selected filters)")
    st.plotly_chart(px.line(df.groupby("date").size().reset_index(name="entries"), x="date", y="entries"),
                    use_container_width=True)

# ---------------- Song trends ----------------
with tabs[1]:
    top_n = st.slider("Songs shown (by days on chart)", 3, 15, 6)
    default = songs_t.sort_values("days_on_chart", ascending=False).head(top_n)["song_id"].tolist()
    chosen = st.multiselect("Songs to chart", sorted(df["song_id"].unique()), default=default)
    sub = df[df["song_id"].isin(chosen)]
    fig = px.line(sub, x="date", y="position", color="song", title="Rank over time")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.line(sub, x="date", y="pop_trend", color="song", title="Popularity trend (rolling mean)"),
                    use_container_width=True)
    st.subheader("Peak rank vs longevity")
    st.plotly_chart(px.scatter(songs_t, x="best_rank", y="days_on_chart", color="album_type",
                               hover_data=["song", "artist", "avg_rank"]), use_container_width=True)
    st.dataframe(songs_t.sort_values("days_on_chart", ascending=False).head(25)[
        ["song", "artist", "days_on_chart", "avg_rank", "best_rank", "rank_volatility", "avg_popularity"]
    ].round(2), use_container_width=True, hide_index=True)

# ---------------- Artist dominance ----------------
with tabs[2]:
    n = st.slider("Artists shown", 5, 30, 15)
    top = art_t.head(n)
    st.plotly_chart(px.bar(top, x="dominance_index", y="primary_artist", orientation="h",
                           labels={"dominance_index": "Share of Top-50 slots (%)"})
                    .update_yaxes(autorange="reversed"), use_container_width=True)
    st.dataframe(top.round(2), use_container_width=True, hide_index=True)
    over_time = df[df["primary_artist"].isin(top.head(5)["primary_artist"])]
    m = over_time.groupby([pd.Grouper(key="date", freq="MS"), "primary_artist"]).size().reset_index(name="slots")
    st.plotly_chart(px.line(m, x="date", y="slots", color="primary_artist",
                            title="Monthly chart slots – top 5 artists"), use_container_width=True)

# ---------------- Popularity vs rank ----------------
with tabs[3]:
    sample = df.sample(min(len(df), 5000), random_state=0)
    st.plotly_chart(px.scatter(sample, x="position", y="popularity", opacity=0.3,
                               hover_data=["song", "artist"]), use_container_width=True)
    corr = df[["position", "popularity"]].corr(method="spearman").iloc[0, 1]
    st.write(f"Spearman correlation (rank vs popularity): **{corr:.3f}** (negative = better rank, higher popularity)")
    bucket = pd.cut(df["position"], [0, 10, 20, 50], labels=["Top 10", "Top 20", "Top 50"])
    st.plotly_chart(px.box(df.assign(bucket=bucket), x="bucket", y="popularity"), use_container_width=True)
    st.plotly_chart(px.scatter(songs_t, x="pop_std", y="rank_volatility", hover_data=["song", "artist"],
                               title="Popularity stability vs rank volatility"), use_container_width=True)

# ---------------- Content attributes ----------------
with tabs[4]:
    a, b = st.columns(2)
    ex = df.groupby("is_explicit").agg(entries=("position", "size"), avg_rank=("position", "mean"),
                                       avg_popularity=("popularity", "mean")).reset_index()
    a.subheader("Explicit vs non-explicit"); a.dataframe(ex.round(2), hide_index=True)
    at = df.groupby("album_type").agg(entries=("position", "size"), avg_rank=("position", "mean"),
                                      avg_popularity=("popularity", "mean")).reset_index()
    b.subheader("Album type"); b.dataframe(at.round(2), hide_index=True)
    st.plotly_chart(px.scatter(songs_t, x="duration_min", y="avg_popularity", color="album_type",
                               hover_data=["song"], title="Duration vs popularity"), use_container_width=True)
    st.plotly_chart(px.scatter(songs_t, x="total_tracks", y="avg_rank", hover_data=["song"],
                               title="Album size vs average rank"), use_container_width=True)
    share = df.groupby(pd.Grouper(key="date", freq="MS"))["is_explicit"].mean().mul(100).reset_index()
    st.plotly_chart(px.line(share, x="date", y="is_explicit", title="Explicit share by month (%)"),
                    use_container_width=True)

# ---------------- Movers ----------------
with tabs[5]:
    mv = movements(df_all[df_all["date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))])
    st.write(f"Average absolute daily rank change: **{mv['rank_change'].abs().mean():.2f}** positions")
    l, r = st.columns(2)
    l.subheader("Biggest one-day risers")
    l.dataframe(mv.nlargest(15, "rank_change")[["date", "song", "artist", "position", "rank_change"]], hide_index=True)
    r.subheader("Biggest one-day drops")
    r.dataframe(mv.nsmallest(15, "rank_change")[["date", "song", "artist", "position", "rank_change"]], hide_index=True)
    entries = df_all.groupby("song_id")["date"].agg(["min", "max"])
    st.write("Entries per month:")
    st.bar_chart(entries["min"].dt.to_period("M").value_counts().sort_index().rename(lambda p: str(p)))

# ---------------- Data quality ----------------
with tabs[6]:
    st.json(report)
