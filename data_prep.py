"""Data ingestion, validation and feature engineering for the US Top 50 analysis."""
import pandas as pd
import numpy as np

CSV = "Atlantic_United_States.csv"


def load_clean(path=CSV):
    df = pd.read_csv(path)
    report = {"raw_rows": len(df)}

    # Dates in the file are dd-mm-yyyy
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    # Validate rank range
    report["rank_out_of_range"] = int((~df["position"].between(1, 50)).sum())
    df = df[df["position"].between(1, 50)]

    # 2025-03-01 contains two different 50-row snapshots; keep the first per (date, position)
    report["duplicate_date_position"] = int(df.duplicated(["date", "position"]).sum())
    df = df.drop_duplicates(["date", "position"], keep="first")

    # Standardise text
    for c in ["song", "artist"]:
        df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)
    df["primary_artist"] = df["artist"].str.split(r" & |, ", n=1, regex=True).str[0]

    # Song identity = title + artist (some titles are reused by different artists)
    df["song_id"] = df["song"] + " | " + df["artist"]
    report["duplicate_date_song"] = int(df.duplicated(["date", "song_id"]).sum())
    df = df.drop_duplicates(["date", "song_id"], keep="first")

    report["missing_values"] = int(df.isna().sum().sum())
    all_days = pd.date_range(df["date"].min(), df["date"].max())
    report["missing_days"] = [d.date().isoformat() for d in all_days.difference(df["date"].unique())]
    report["clean_rows"] = len(df)

    df["duration_min"] = df["duration_ms"] / 60000
    df = df.sort_values(["song_id", "date"])
    # Popularity Trend Score: 7-observation rolling mean per song
    df["pop_trend"] = df.groupby("song_id")["popularity"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return df.sort_values(["date", "position"]).reset_index(drop=True), report


def song_table(df):
    g = df.groupby("song_id").agg(
        song=("song", "first"), artist=("artist", "first"),
        days_on_chart=("date", "nunique"), avg_rank=("position", "mean"),
        best_rank=("position", "min"), rank_volatility=("position", "std"),
        avg_popularity=("popularity", "mean"), pop_std=("popularity", "std"),
        first_seen=("date", "min"), last_seen=("date", "max"),
        duration_min=("duration_min", "first"), is_explicit=("is_explicit", "first"),
        album_type=("album_type", "first"), total_tracks=("total_tracks", "first"),
    ).reset_index()
    g["rank_volatility"] = g["rank_volatility"].fillna(0)
    return g


def artist_table(df):
    total_slots = df["date"].nunique() * 50
    g = df.groupby("primary_artist").agg(
        unique_songs=("song_id", "nunique"), days_present=("date", "nunique"),
        chart_slots=("position", "size"), avg_rank=("position", "mean"),
        avg_popularity=("popularity", "mean"),
    ).reset_index()
    g["dominance_index"] = g["chart_slots"] / total_slots * 100  # % of all Top-50 slots held
    return g.sort_values("dominance_index", ascending=False)


def movements(df):
    """Day-over-day rank change per song (positive = moved up)."""
    d = df.sort_values(["song_id", "date"]).copy()
    gap = d.groupby("song_id")["date"].diff().dt.days
    d["rank_change"] = (d.groupby("song_id")["position"].shift(1) - d["position"]).where(gap == 1)
    return d
