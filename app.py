import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import glob
import json
import plotly.express as px
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib
import io
import os
import requests
import plotly.graph_objects as go
import base64
import spotipy
from spotipy.oauth2 import SpotifyOAuth

CACHE_FILE = "artist_genres_cache.json"

matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()

st.set_page_config(layout="wide")
st.set_page_config(page_title="All-Time Spotify Recap", page_icon="🎵")

@st.cache_data(show_spinner=False)
def get_top_plot_data(filtered_df, topic, minutes, stacked):
  
  if topic == "Artists":
        if stacked:
            agg = filtered_df.groupby(["master_metadata_album_artist_name", "year"])["ms_played"].sum().unstack(fill_value=0) if minutes else filtered_df[filtered_df["ms_played"] >= 30000].groupby("master_metadata_album_artist_name").size().sort_values(ascending=False)
            agg["total"] = agg.sum(axis=1)
            agg = agg.sort_values("total", ascending=False)
            agg = agg.drop(columns="total")
        else:
            agg = filtered_df.groupby("master_metadata_album_artist_name")["ms_played"].sum().sort_values(ascending=False) if minutes else filtered_df[filtered_df["ms_played"] >= 30000].groupby("master_metadata_album_artist_name").size().sort_values(ascending=False)
  elif topic == "Songs":
        if stacked:
            agg = filtered_df.groupby(["master_metadata_album_artist_name","master_metadata_track_name", "year"])["ms_played"].sum().unstack(fill_value=0) if minutes else filtered_df[filtered_df["ms_played"] >= 30000].groupby(["master_metadata_album_artist_name","master_metadata_track_name", "year"]).size().unstack(fill_value=0)
            agg["total"] = agg.sum(axis=1)
            agg = agg.sort_values("total", ascending=False)
            agg = agg.drop(columns="total")
        else:
            agg = filtered_df.groupby(["master_metadata_album_artist_name","master_metadata_track_name"])["ms_played"].sum().sort_values(ascending=False) if minutes else filtered_df[filtered_df["ms_played"] >= 30000].groupby(["master_metadata_album_artist_name","master_metadata_track_name"]).size().sort_values(ascending=False)
  elif topic == "Albums":
        if stacked:
            agg = filtered_df.groupby(["master_metadata_album_artist_name","master_metadata_album_album_name", "year"])["ms_played"].sum().unstack(fill_value=0) if minutes else filtered_df[filtered_df["ms_played"] >= 30000].groupby(["master_metadata_album_artist_name","master_metadata_album_album_name", "year"]).size().unstack(fill_value=0)
            agg["total"] = agg.sum(axis=1)
            agg = agg.sort_values("total", ascending=False)
            agg = agg.drop(columns="total")
        else:
            agg = filtered_df.groupby(["master_metadata_album_artist_name","master_metadata_album_album_name"])["ms_played"].sum().sort_values(ascending=False) if minutes else filtered_df[filtered_df["ms_played"] >= 30000].groupby(["master_metadata_album_artist_name","master_metadata_album_album_name"]).size().sort_values(ascending=False)

  if minutes:
    agg = agg / 60000  # Convert to minutes

  return agg

def get_top_plot(agg_top, topic, minutes, stacked):    
    height = len(agg_top) * 0.55

    if (topic in ["Songs", "Albums"]):
      agg_top.index = agg_top.index.map(lambda x: f"{truncate_name(x[1], x[0])}")

    if stacked:
        fig, ax = plt.subplots(figsize=(12, height))
        agg_top.plot(kind="barh", stacked=True, cmap="tab20", width=0.8, ax=ax)
        plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.figure(figsize=(12, height))
        ax = agg_top.plot(kind="barh", stacked=False)

    # Add total values at the end of bars
    if stacked:
        totals = agg_top.sum(axis=1)
        for i, v in enumerate(totals):
            ax.text(v + 2, i, f"{int(v)}", va='center', fontsize=10)
    else:
        for i, (name, v) in enumerate(agg_top.items()):
            ax.text(v + 2, i, f"{int(v)}", va='center', fontsize=10)

    # Add rank numbers
    for i in range(len(agg_top)):
        ax.text(0.5, i, f"{i+1}", va='center', ha='left', fontsize=8, fontweight='bold', color='white')

    # Set labels and title
    unit = "Total Minutes" if minutes else "# Plays"
    plt.xlabel(unit)
    plt.ylabel(topic)
    plt.title(f"Top {topic} by {unit}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return plt

@st.cache_data(show_spinner=False)
def get_timeline_plot_data(song_df, granularity):
  playtime = defaultdict(int)
  song_df['ts'] = pd.to_datetime(song_df['ts'], format="%Y-%m-%dT%H:%M:%SZ")

  # Create 'period' column based on granularity
  if granularity == "Month":
      song_df['period'] = song_df['ts'].dt.strftime("%Y-%m")
  elif granularity == "Week":
      song_df['period'] = song_df['ts'].dt.strftime("%Y-W%V")
  elif granularity == "Year":
      song_df['period'] = song_df['ts'].dt.strftime("%Y")

  # Aggregate playtime by period
  playtime = song_df.groupby('period')['ms_played'].sum().to_dict()

  # Convert to minutes and prepare DataFrame
  periods = sorted(playtime.keys())
  minutes = [playtime[p] / 1000 / 60 for p in periods]

  df = pd.DataFrame({
      "Period": periods,
      "MinutesPlayed": minutes
  })

  # Add a 'HoverText' column for hover template
  if granularity == "Week":
      def get_week_range(week_str):
          year, week = week_str.split("-W")
          year = int(year)
          week = int(week)
          first_day = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
          last_day = first_day + timedelta(days=6)
          return f"{first_day.strftime('%d/%m/%y')} - {last_day.strftime('%d/%m/%y')}"

      df["HoverText"] = df["Period"].apply(get_week_range)
  elif granularity == "Month":
      def format_month(month_str):
          year, month = month_str.split("-")
          month_name = datetime.strptime(month, "%m").strftime("%b")
          return f"{month_name} {year}"

      df["HoverText"] = df["Period"].apply(format_month)
  else:
      df["HoverText"] = df["Period"]

  # Convert 'Period' to datetime for better x-axis handling
  if granularity == "Month":
      df["Date"] = pd.to_datetime(df["Period"] + "-01")  # Add day "01" for months
  elif granularity == "Week":
      def week_to_datetime(week_str):
          year, week = map(int, week_str.split("-W"))
          return datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")  # Monday of the week
      df["Date"] = df["Period"].apply(week_to_datetime)
  elif granularity == "Year":
      df["Date"] = pd.to_datetime(df["Period"], format="%Y")

  return df

def get_timeline_plot(song_df, granularity):
    df = get_timeline_plot_data(song_df, granularity)

    # Plot with datetime x-axis
    fig = px.bar(
        df,
        x="Date",
        y="MinutesPlayed",
        text="HoverText",  # Use HoverText for hover
        labels={"Date": "", "MinutesPlayed": "Minutes Played"}
    )

    # Customize hover template and hide bar labels
    fig.update_traces(
        hovertemplate='%{text}<br>%{y:.0f} Minutes',
        textposition='none'  # Hide text labels on bars
    )

    # Add title and adjust layout
    fig.update_layout(
        title=f"Total Minutes Played per {granularity}",
        yaxis_title="Minutes Played"
    )

    return fig

@st.cache_data(show_spinner=False)
def get_top_per_period_plot_data(song_df, entity, metric, granularity):
  # Aggregate data
    play_data = defaultdict(int)
    play_counts = defaultdict(int)

    song_df['ts'] = pd.to_datetime(song_df['ts'], format="%Y-%m-%dT%H:%M:%SZ")

    # Create 'period' column
    song_df['period'] = song_df['ts'].dt.strftime("%Y-%m" if granularity == "Month" else "%Y")

    # Create 'key' column based on entity
    if entity == "Song":
        song_df['key'] = list(zip(
            song_df["master_metadata_track_name"],
            song_df["master_metadata_album_artist_name"],
            song_df["period"]
        ))
    elif entity == "Artist":
        song_df['key'] = list(zip(
            song_df["master_metadata_album_artist_name"],
            song_df["period"]
        ))
    elif entity == "Album":
        song_df['key'] = list(zip(
            song_df["master_metadata_album_album_name"],
            song_df["master_metadata_album_artist_name"],
            song_df["period"]
        ))

    # Aggregate play_data and play_counts
    play_data = song_df.groupby('key')['ms_played'].sum().to_dict()
    play_counts = song_df[song_df['ms_played'] >= 30000].groupby('key').size().to_dict()

    # Convert to DataFrame
    if entity == "Song":
        df = pd.DataFrame([
            {"Track": k[0], "Artist": k[1], "Period": k[2], "ms": v, "Plays": play_counts.get(k, 0)}
            for k, v in play_data.items()
        ])
    elif entity == "Artist":
        df = pd.DataFrame([
            {"Artist": k[0], "Period": k[1], "ms": v, "Plays": play_counts.get(k, 0)}
            for k, v in play_data.items()
        ])
    elif entity == "Album":
        df = pd.DataFrame([
            {"Album": k[0], "Artist": k[1], "Period": k[2], "ms": v, "Plays": play_counts.get(k, 0)}
            for k, v in play_data.items()
        ])

    # Convert to minutes
    df["Minutes"] = df["ms"] / 1000 / 60

    return df
def get_top_per_period_plot(song_df, entity, metric, granularity):
    df = get_top_per_period_plot_data(song_df, entity, metric, granularity)

    # Get top entity per period
    if metric == "Total Minutes":
        top_per_period = df.loc[df.groupby("Period")["Minutes"].idxmax()].sort_values("Period")
    else:  # # Plays
        top_per_period = df.loc[df.groupby("Period")["Plays"].idxmax()].sort_values("Period")

    # Create labels
    def short(s, n=30):
        return s if len(s) <= n else s[:n-3] + "..."

    if entity == "Song":
        top_per_period["Label"] = top_per_period.apply(
            lambda row: f"{row['Period']} – {short(row['Track'])} ({short(row['Artist'])})",
            axis=1
        )
    elif entity == "Artist":
        top_per_period["Label"] = top_per_period.apply(
            lambda row: f"{row['Period']} – {short(row['Artist'])}",
            axis=1
        )
    elif entity == "Album":
        top_per_period["Label"] = top_per_period.apply(
            lambda row: f"{row['Period']} – {short(row['Album'])} ({short(row['Artist'])})",
            axis=1
        )

    # Plot with Matplotlib
    plt.figure(figsize=(12, 0.5 * len(top_per_period)))
    ax = plt.gca()

    metric_value = "Minutes" if metric == "Total Minutes" else "Plays"
    ax.barh(top_per_period["Label"], top_per_period[metric_value], color='skyblue')

    # Add value labels
    for i, v in enumerate(top_per_period[metric_value]):
        ax.text(v + 0.5, i, f"{int(v)}", va='center', fontsize=8)

    # Customize plot
    plt.xlabel(metric)
    plt.title(f"Top {entity} each {granularity} by {metric}")
    ax.set_ylim(-0.5, len(top_per_period) - 0.5)
    plt.gca().invert_yaxis()  # Highest on top
    plt.tight_layout()

    return plt

def get_week_range(week_str):
    """Convert week string (e.g., '2025-W41') to date range string (e.g., '12-18/10/25')"""
    year, week = map(int, week_str.split("-W"))
    # Find Monday of the week
    first_day = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
    # Find Sunday of the week
    last_day = first_day + timedelta(days=6)
    # Format as "DD-DD/MM/YY"
    return f"{first_day.day:01d}-{last_day.day:01d}/{first_day.month:01d}/{str(first_day.year)[-2:]}"

@st.cache_data(show_spinner=False)
def get_top_listening_combos_data(song_df, entries, metric, granularity, entity_type):
  play_data = defaultdict(int)
  play_counts = defaultdict(int)

  song_df['ts'] = pd.to_datetime(song_df['ts'], format="%Y-%m-%dT%H:%M:%SZ")

  # Create 'period' column based on granularity
  if granularity == "Year":
      song_df['period'] = song_df['ts'].dt.strftime("%Y")
  elif granularity == "Month":
      song_df['period'] = song_df['ts'].dt.strftime("%Y-%m")
  elif granularity == "Week":
      song_df['period'] = song_df['ts'].dt.strftime("%Y-W%V")
  elif granularity == "Day":
      song_df['period'] = song_df['ts'].dt.strftime("%Y-%m-%d")

  # Create 'key' column based on entity_type
  if entity_type == "Song":
      song_df['key'] = list(zip(
          song_df["master_metadata_track_name"],
          song_df["master_metadata_album_artist_name"],
          song_df["period"]
      ))
  elif entity_type == "Artist":
      song_df['key'] = list(zip(
          song_df["master_metadata_album_artist_name"],
          song_df["period"]
      ))
  elif entity_type == "Album":
      song_df['key'] = list(zip(
          song_df["master_metadata_album_album_name"],
          song_df["master_metadata_album_artist_name"],
          song_df["period"]
      ))

  # Group by 'key' and aggregate
  play_data_df = song_df.groupby('key')['ms_played'].sum()
  play_counts_df = song_df[song_df['ms_played'] >= 30000].groupby('key').size()

  # Convert to dictionaries if needed
  play_data = play_data_df.to_dict()
  play_counts = play_counts_df.to_dict()

  # Convert to DataFrame based on entity type
  if entity_type == "Song":
      df = pd.DataFrame([
          {"Track": k[0], "Artist": k[1], "Period": k[2], "ms": v, "Plays": play_counts.get(k, 0)}
          for k, v in play_data.items()
      ])
  elif entity_type == "Artist":
      df = pd.DataFrame([
          {"Artist": k[0], "Period": k[1], "ms": v, "Plays": play_counts.get(k, 0)}
          for k, v in play_data.items()
      ])
  elif entity_type == "Album":
      df = pd.DataFrame([
          {"Album": k[0], "Artist": k[1], "Period": k[2], "ms": v, "Plays": play_counts.get(k, 0)}
          for k, v in play_data.items()
      ])

  return df
def get_top_listening_combos(song_df, entries, metric, granularity, entity_type):

    df = get_top_listening_combos_data(song_df, entries, metric, granularity, entity_type)
    # Convert to minutes if needed
    if metric == "Total Minutes":
        df["value"] = df["ms"] / 1000 / 60
        metric_label = "Minutes Played"
    else:  # plays
        df["value"] = df["Plays"]
        metric_label = "# Plays"

    # Format period for display
    def format_period(period, granularity):
        if granularity == "Week":
            return get_week_range(period)
        elif granularity == "Month":
            month_name = datetime.strptime(period, "%Y-%m").strftime("%b %Y")
            return month_name
        elif granularity == "Day":
            return datetime.strptime(period, "%Y-%m-%d").strftime("%d/%m/%y")
        else:  # year
            return period

    # Add formatted period for labels
    df["FormattedPeriod"] = df.apply(lambda row: format_period(row["Period"], granularity), axis=1)

    # Sort by value in descending order (highest to lowest)
    top_entries = df.sort_values("value", ascending=False).head(entries)

    # Create labels based on entity type
    def short(s, n=40):
        return s if len(s) <= n else s[:n-3] + "..."

    if entity_type == "Song":
        top_entries["Label"] = top_entries.apply(
            lambda row: f"{row['FormattedPeriod']} - {short(row['Track'])} ({short(row['Artist'])})",
            axis=1
        )
    elif entity_type == "Artist":
        top_entries["Label"] = top_entries.apply(
            lambda row: f"{row['FormattedPeriod']} - {short(row['Artist'])}",
            axis=1
        )
    elif entity_type == "Album":
        top_entries["Label"] = top_entries.apply(
            lambda row: f"{row['FormattedPeriod']} - {short(row['Album'])} ({short(row['Artist'])})",
            axis=1
        )

    # Plot
    plt.figure(figsize=(12, 0.4 * len(top_entries)))
    ax = plt.gca()

    # Plot bars in reverse order (so the highest value is at the top)
    y_pos = range(len(top_entries)-1, -1, -1)
    ax.barh(y_pos, top_entries["value"], color='skyblue')

    # Add value labels (in reverse order)
    max_value = top_entries["value"].max()
    for i, v in enumerate(top_entries["value"]):
        ax.text(v + max_value * 0.02, y_pos[i], f"{int(v)}", va='center', fontsize=8)

    # Add rank numbers (in reverse order)
    for i in range(len(top_entries)):
        ax.text(max_value * 0.01, y_pos[i], f"{i+1}", va='center', ha='left',
                fontsize=8, fontweight='bold', color='white')

    # Set y-ticks to show labels in the correct order
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_entries["Label"])

    # Remove extra space at top and bottom
    ax.set_ylim(-0.5, len(top_entries) - 0.5)

    # Customize plot
    plt.xlabel(metric_label)
    plt.title(f"{entity_type}-{granularity} combos by {metric}")
    plt.tight_layout()

    return plt

def truncate_name(x0, x1, max_len=60):
  if len(x0 + x1) > max_len:
    position = max_len - len(x1) - 3
    return x0[:(position if position > 0 else 1)] + "..." + " (" + x1 + ")"
  else:
    return f"{x0} ({x1})"

def load_genre_cache():
    """Load the genre cache from the JSON file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_genre_cache(cache):
    """Save the genre cache to the JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

@st.cache_data(show_spinner=False)
def get_top_genres_plot_data(song_df):
    access_token = get_access_token()

    artist_names = song_df["master_metadata_album_artist_name"].unique().tolist()

    cache = load_genre_cache()
    missing_artist_names = [name for name in artist_names if name not in cache]

    if missing_artist_names:

        artist_track_uris = {}
        for name in missing_artist_names:

            track_uri = song_df[song_df["master_metadata_album_artist_name"] == name]["spotify_track_uri"].iloc[0]
            artist_track_uris[name] = track_uri.split(':')[-1]

        track_uris = list(artist_track_uris.values())
        artist_ids = {}

        batch_size = 50
        for i in range(0, len(track_uris), batch_size):
            batch = track_uris[i:i + batch_size]
            tracks_url = f"https://api.spotify.com/v1/tracks?ids={','.join(batch)}"
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(tracks_url, headers=headers)

            if response.status_code == 200:
                tracks = response.json()["tracks"]
                for track in tracks:
                    if track and "artists" in track and track["artists"]:
                        artist_id = track["artists"][0]["id"]
                        artist_name = next(
                            name for name, uri in artist_track_uris.items()
                            if uri == f"{track['id']}"
                        )
                        artist_ids[artist_name] = artist_id

        if artist_ids:
            batch_size = 50
            for i in range(0, len(artist_ids), batch_size):
                batch = list(artist_ids.values())[i:i + batch_size]
                artists_url = f"https://api.spotify.com/v1/artists?ids={','.join(batch)}"
                headers = {"Authorization": f"Bearer {access_token}"}
                response = requests.get(artists_url, headers=headers)

                if response.status_code == 200:
                    artists = response.json()["artists"]
                    for artist in artists:
                        if artist and "genres" in artist:
                            artist_name = next(
                                name for name, aid in artist_ids.items()
                                if aid == artist["id"]
                            )
                            cache[artist_name] = artist["genres"]

        save_genre_cache(cache)

    artist_genres = {name: cache.get(name, []) for name in artist_names}

    artist_listening_time = song_df.groupby("master_metadata_album_artist_name")["ms_played"].sum().to_dict()

    genre_data = defaultdict(lambda: {"total_ms": 0, "artists": defaultdict(int)})
    for artist_name in artist_names:
        total_ms = artist_listening_time.get(artist_name, 0)
        genres = artist_genres.get(artist_name, [])
        for genre in genres:
            genre_data[genre]["total_ms"] += total_ms
            genre_data[genre]["artists"][artist_name] += total_ms

    sorted_genres = sorted(genre_data.items(), key=lambda x: x[1]["total_ms"], reverse=True)

    return sorted_genres

def get_top_genres_plot(song_df, entries, selected_year):

    if selected_year != "All":
        filtered_df = song_df[song_df["year"] == selected_year]
    else:
        filtered_df = song_df
    sorted_genres = get_top_genres_plot_data(filtered_df)

    genres = []
    total_minutes = []
    hover_texts = []
    for genre, data in sorted_genres[:entries]:
        genres.append(genre)
        total_minutes.append(data["total_ms"] / 60000)
        top_artists = sorted(data["artists"].items(), key=lambda x: x[1], reverse=True)[:5]
        artist_list ="<b>Top 5</b> <br>" + "<br>".join([f"{artist}: {ms / 60000:.1f} min" for artist, ms in top_artists])
        hover_texts.append(f"{artist_list}")
    max_minutes = max(total_minutes)
    x_range = [0, max_minutes * 1.2]

    num_entries = len(genres)
    bar_height = 30
    fig_height = num_entries * bar_height + 200
    ranking_text = [f"{i}" for i in range(1, num_entries + 1)]

    fig = go.Figure(
        go.Bar(
            x=total_minutes,
            y=genres,
            orientation="h",
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(color="skyblue"),
            text=ranking_text,
            insidetextanchor="start",
            textfont=dict(size=10, color="white"),
        )
    )

    for i, (genre, minutes) in enumerate(zip(genres, total_minutes)):
        fig.add_annotation(
            xref="x", yref="y",
            x=minutes,
            y=genre,
            text=f"{minutes:.1f}",
            showarrow=False,
            font=dict(size=12, color="white"),
            xanchor="left",
            xshift=10,
        )

    fig.update_layout(
        title=f"Top Genres by Total Minutes ({'All Years' if selected_year == 'All' else selected_year})",
        xaxis_title="Total Minutes",
        yaxis_title="Genre",
        yaxis=dict(
            autorange="reversed",
            fixedrange=False,
        ),
        xaxis=dict(range=x_range),
        margin=dict(r=150),
        height=fig_height,
    )

    return fig

def get_map_plot(song_df):

  def alpha2_to_alpha3(alpha2_code):
      alpha2_to_alpha3_map = {
      "AF": "AFG",  # Afghanistan
      "AX": "ALA",  # Åland Islands
      "AL": "ALB",  # Albania
      "DZ": "DZA",  # Algeria
      "AS": "ASM",  # American Samoa
      "AD": "AND",  # Andorra
      "AO": "AGO",  # Angola
      "AI": "AIA",  # Anguilla
      "AQ": "ATA",  # Antarctica
      "AG": "ATG",  # Antigua and Barbuda
      "AR": "ARG",  # Argentina
      "AM": "ARM",  # Armenia
      "AW": "ABW",  # Aruba
      "AU": "AUS",  # Australia
      "AT": "AUT",  # Austria
      "AZ": "AZE",  # Azerbaijan
      "BS": "BHS",  # Bahamas
      "BH": "BHR",  # Bahrain
      "BD": "BGD",  # Bangladesh
      "BB": "BRB",  # Barbados
      "BY": "BLR",  # Belarus
      "BE": "BEL",  # Belgium
      "BZ": "BLZ",  # Belize
      "BJ": "BEN",  # Benin
      "BM": "BMU",  # Bermuda
      "BT": "BTN",  # Bhutan
      "BO": "BOL",  # Bolivia
      "BQ": "BES",  # Bonaire, Sint Eustatius, and Saba
      "BA": "BIH",  # Bosnia and Herzegovina
      "BW": "BWA",  # Botswana
      "BV": "BVT",  # Bouvet Island
      "BR": "BRA",  # Brazil
      "IO": "IOT",  # British Indian Ocean Territory
      "BN": "BRN",  # Brunei Darussalam
      "BG": "BGR",  # Bulgaria
      "BF": "BFA",  # Burkina Faso
      "BI": "BDI",  # Burundi
      "CV": "CPV",  # Cabo Verde
      "KH": "KHM",  # Cambodia
      "CM": "CMR",  # Cameroon
      "CA": "CAN",  # Canada
      "KY": "CYM",  # Cayman Islands
      "CF": "CAF",  # Central African Republic
      "TD": "TCD",  # Chad
      "CL": "CHL",  # Chile
      "CN": "CHN",  # China
      "CX": "CXR",  # Christmas Island
      "CC": "CCK",  # Cocos (Keeling) Islands
      "CO": "COL",  # Colombia
      "KM": "COM",  # Comoros
      "CG": "COG",  # Congo (Brazzaville)
      "CD": "COD",  # Congo (Kinshasa)
      "CK": "COK",  # Cook Islands
      "CR": "CRI",  # Costa Rica
      "CI": "CIV",  # Côte d'Ivoire
      "HR": "HRV",  # Croatia
      "CU": "CUB",  # Cuba
      "CW": "CUW",  # Curaçao
      "CY": "CYP",  # Cyprus
      "CZ": "CZE",  # Czech Republic
      "DK": "DNK",  # Denmark
      "DJ": "DJI",  # Djibouti
      "DM": "DMA",  # Dominica
      "DO": "DOM",  # Dominican Republic
      "EC": "ECU",  # Ecuador
      "EG": "EGY",  # Egypt
      "SV": "SLV",  # El Salvador
      "GQ": "GNQ",  # Equatorial Guinea
      "ER": "ERI",  # Eritrea
      "EE": "EST",  # Estonia
      "SZ": "SWZ",  # Eswatini
      "ET": "ETH",  # Ethiopia
      "FK": "FLK",  # Falkland Islands
      "FO": "FRO",  # Faroe Islands
      "FJ": "FJI",  # Fiji
      "FI": "FIN",  # Finland
      "FR": "FRA",  # France
      "GF": "GUF",  # French Guiana
      "PF": "PYF",  # French Polynesia
      "TF": "ATF",  # French Southern Territories
      "GA": "GAB",  # Gabon
      "GM": "GMB",  # Gambia
      "GE": "GEO",  # Georgia
      "DE": "DEU",  # Germany
      "GH": "GHA",  # Ghana
      "GI": "GIB",  # Gibraltar
      "GR": "GRC",  # Greece
      "GL": "GRL",  # Greenland
      "GD": "GRD",  # Grenada
      "GP": "GLP",  # Guadeloupe
      "GU": "GUM",  # Guam
      "GT": "GTM",  # Guatemala
      "GG": "GGY",  # Guernsey
      "GN": "GIN",  # Guinea
      "GW": "GNB",  # Guinea-Bissau
      "GY": "GUY",  # Guyana
      "HT": "HTI",  # Haiti
      "HM": "HMD",  # Heard Island and McDonald Islands
      "VA": "VAT",  # Holy See
      "HN": "HND",  # Honduras
      "HK": "HKG",  # Hong Kong
      "HU": "HUN",  # Hungary
      "IS": "ISL",  # Iceland
      "IN": "IND",  # India
      "ID": "IDN",  # Indonesia
      "IR": "IRN",  # Iran
      "IQ": "IRQ",  # Iraq
      "IE": "IRL",  # Ireland
      "IM": "IMN",  # Isle of Man
      "IL": "ISR",  # Israel
      "IT": "ITA",  # Italy
      "JM": "JAM",  # Jamaica
      "JP": "JPN",  # Japan
      "JE": "JEY",  # Jersey
      "JO": "JOR",  # Jordan
      "KZ": "KAZ",  # Kazakhstan
      "KE": "KEN",  # Kenya
      "KI": "KIR",  # Kiribati
      "KP": "PRK",  # North Korea
      "KR": "KOR",  # South Korea
      "KW": "KWT",  # Kuwait
      "KG": "KGZ",  # Kyrgyzstan
      "LA": "LAO",  # Laos
      "LV": "LVA",  # Latvia
      "LB": "LBN",  # Lebanon
      "LS": "LSO",  # Lesotho
      "LR": "LBR",  # Liberia
      "LY": "LBY",  # Libya
      "LI": "LIE",  # Liechtenstein
      "LT": "LTU",  # Lithuania
      "LU": "LUX",  # Luxembourg
      "MO": "MAC",  # Macao
      "MG": "MDG",  # Madagascar
      "MW": "MWI",  # Malawi
      "MY": "MYS",  # Malaysia
      "MV": "MDV",  # Maldives
      "ML": "MLI",  # Mali
      "MT": "MLT",  # Malta
      "MH": "MHL",  # Marshall Islands
      "MQ": "MTQ",  # Martinique
      "MR": "MRT",  # Mauritania
      "MU": "MUS",  # Mauritius
      "YT": "MYT",  # Mayotte
      "MX": "MEX",  # Mexico
      "FM": "FSM",  # Micronesia
      "MD": "MDA",  # Moldova
      "MC": "MCO",  # Monaco
      "MN": "MNG",  # Mongolia
      "ME": "MNE",  # Montenegro
      "MS": "MSR",  # Montserrat
      "MA": "MAR",  # Morocco
      "MZ": "MOZ",  # Mozambique
      "MM": "MMR",  # Myanmar
      "NA": "NAM",  # Namibia
      "NR": "NRU",  # Nauru
      "NP": "NPL",  # Nepal
      "NL": "NLD",  # Netherlands
      "NC": "NCL",  # New Caledonia
      "NZ": "NZL",  # New Zealand
      "NI": "NIC",  # Nicaragua
      "NE": "NER",  # Niger
      "NG": "NGA",  # Nigeria
      "NU": "NIU",  # Niue
      "NF": "NFK",  # Norfolk Island
      "MK": "MKD",  # North Macedonia
      "MP": "MNP",  # Northern Mariana Islands
      "NO": "NOR",  # Norway
      "OM": "OMN",  # Oman
      "PK": "PAK",  # Pakistan
      "PW": "PLW",  # Palau
      "PS": "PSE",  # Palestine
      "PA": "PAN",  # Panama
      "PG": "PNG",  # Papua New Guinea
      "PY": "PRY",  # Paraguay
      "PE": "PER",  # Peru
      "PH": "PHL",  # Philippines
      "PN": "PCN",  # Pitcairn
      "PL": "POL",  # Poland
      "PT": "PRT",  # Portugal
      "PR": "PRI",  # Puerto Rico
      "QA": "QAT",  # Qatar
      "RE": "REU",  # Réunion
      "RO": "ROU",  # Romania
      "RU": "RUS",  # Russia
      "RW": "RWA",  # Rwanda
      "BL": "BLM",  # Saint Barthélemy
      "SH": "SHN",  # Saint Helena, Ascension, and Tristan da Cunha
      "KN": "KNA",  # Saint Kitts and Nevis
      "LC": "LCA",  # Saint Lucia
      "MF": "MAF",  # Saint Martin (French part)
      "PM": "SPM",  # Saint Pierre and Miquelon
      "VC": "VCT",  # Saint Vincent and the Grenadines
      "WS": "WSM",  # Samoa
      "SM": "SMR",  # San Marino
      "ST": "STP",  # Sao Tome and Principe
      "SA": "SAU",  # Saudi Arabia
      "SN": "SEN",  # Senegal
      "RS": "SRB",  # Serbia
      "SC": "SYC",  # Seychelles
      "SL": "SLE",  # Sierra Leone
      "SG": "SGP",  # Singapore
      "SX": "SXM",  # Sint Maarten (Dutch part)
      "SK": "SVK",  # Slovakia
      "SI": "SVN",  # Slovenia
      "SB": "SLB",  # Solomon Islands
      "SO": "SOM",  # Somalia
      "ZA": "ZAF",  # South Africa
      "GS": "SGS",  # South Georgia and the South Sandwich Islands
      "SS": "SSD",  # South Sudan
      "ES": "ESP",  # Spain
      "LK": "LKA",  # Sri Lanka
      "SD": "SDN",  # Sudan
      "SR": "SUR",  # Suriname
      "SJ": "SJM",  # Svalbard and Jan Mayen
      "SE": "SWE",  # Sweden
      "CH": "CHE",  # Switzerland
      "SY": "SYR",  # Syria
      "TW": "TWN",  # Taiwan
      "TJ": "TJK",  # Tajikistan
      "TZ": "TZA",  # Tanzania
      "TH": "THA",  # Thailand
      "TL": "TLS",  # Timor-Leste
      "TG": "TGO",  # Togo
      "TK": "TKL",  # Tokelau
      "TO": "TON",  # Tonga
      "TT": "TTO",  # Trinidad and Tobago
      "TN": "TUN",  # Tunisia
      "TR": "TUR",  # Turkey
      "TM": "TKM",  # Turkmenistan
      "TC": "TCA",  # Turks and Caicos Islands
      "TV": "TUV",  # Tuvalu
      "UG": "UGA",  # Uganda
      "UA": "UKR",  # Ukraine
      "AE": "ARE",  # United Arab Emirates
      "GB": "GBR",  # United Kingdom
      "US": "USA",  # United States
      "UM": "UMI",  # United States Minor Outlying Islands
      "UY": "URY",  # Uruguay
      "UZ": "UZB",  # Uzbekistan
      "VU": "VUT",  # Vanuatu
      "VE": "VEN",  # Venezuela
      "VN": "VNM",  # Vietnam
      "VG": "VGB",  # British Virgin Islands
      "VI": "VIR",  # U.S. Virgin Islands
      "WF": "WLF",  # Wallis and Futuna
      "EH": "ESH",  # Western Sahara
      "YE": "YEM",  # Yemen
      "ZM": "ZMB",  # Zambia
      "ZW": "ZWE",  # Zimbabwe
      }
      return alpha2_to_alpha3_map.get(alpha2_code, alpha2_code)

  def alpha2_to_name(alpha2_code):
    alpha2_to_name_map = {
        "AF": "Afghanistan",
        "AX": "Åland Islands",
        "AL": "Albania",
        "DZ": "Algeria",
        "AS": "American Samoa",
        "AD": "Andorra",
        "AO": "Angola",
        "AI": "Anguilla",
        "AQ": "Antarctica",
        "AG": "Antigua and Barbuda",
        "AR": "Argentina",
        "AM": "Armenia",
        "AW": "Aruba",
        "AU": "Australia",
        "AT": "Austria",
        "AZ": "Azerbaijan",
        "BS": "Bahamas",
        "BH": "Bahrain",
        "BD": "Bangladesh",
        "BB": "Barbados",
        "BY": "Belarus",
        "BE": "Belgium",
        "BZ": "Belize",
        "BJ": "Benin",
        "BM": "Bermuda",
        "BT": "Bhutan",
        "BO": "Bolivia",
        "BQ": "Bonaire, Sint Eustatius and Saba",
        "BA": "Bosnia and Herzegovina",
        "BW": "Botswana",
        "BV": "Bouvet Island",
        "BR": "Brazil",
        "IO": "British Indian Ocean Territory",
        "BN": "Brunei Darussalam",
        "BG": "Bulgaria",
        "BF": "Burkina Faso",
        "BI": "Burundi",
        "CV": "Cabo Verde",
        "KH": "Cambodia",
        "CM": "Cameroon",
        "CA": "Canada",
        "KY": "Cayman Islands",
        "CF": "Central African Republic",
        "TD": "Chad",
        "CL": "Chile",
        "CN": "China",
        "CX": "Christmas Island",
        "CC": "Cocos (Keeling) Islands",
        "CO": "Colombia",
        "KM": "Comoros",
        "CG": "Congo (Brazzaville)",
        "CD": "Congo (Kinshasa)",
        "CK": "Cook Islands",
        "CR": "Costa Rica",
        "CI": "Côte d'Ivoire",
        "HR": "Croatia",
        "CU": "Cuba",
        "CW": "Curaçao",
        "CY": "Cyprus",
        "CZ": "Czech Republic",
        "DK": "Denmark",
        "DJ": "Djibouti",
        "DM": "Dominica",
        "DO": "Dominican Republic",
        "EC": "Ecuador",
        "EG": "Egypt",
        "SV": "El Salvador",
        "GQ": "Equatorial Guinea",
        "ER": "Eritrea",
        "EE": "Estonia",
        "SZ": "Eswatini",
        "ET": "Ethiopia",
        "FK": "Falkland Islands",
        "FO": "Faroe Islands",
        "FJ": "Fiji",
        "FI": "Finland",
        "FR": "France",
        "GF": "French Guiana",
        "PF": "French Polynesia",
        "TF": "French Southern Territories",
        "GA": "Gabon",
        "GM": "Gambia",
        "GE": "Georgia",
        "DE": "Germany",
        "GH": "Ghana",
        "GI": "Gibraltar",
        "GR": "Greece",
        "GL": "Greenland",
        "GD": "Grenada",
        "GP": "Guadeloupe",
        "GU": "Guam",
        "GT": "Guatemala",
        "GG": "Guernsey",
        "GN": "Guinea",
        "GW": "Guinea-Bissau",
        "GY": "Guyana",
        "HT": "Haiti",
        "HM": "Heard Island and McDonald Islands",
        "VA": "Holy See",
        "HN": "Honduras",
        "HK": "Hong Kong",
        "HU": "Hungary",
        "IS": "Iceland",
        "IN": "India",
        "ID": "Indonesia",
        "IR": "Iran",
        "IQ": "Iraq",
        "IE": "Ireland",
        "IM": "Isle of Man",
        "IL": "Israel",
        "IT": "Italy",
        "JM": "Jamaica",
        "JP": "Japan",
        "JE": "Jersey",
        "JO": "Jordan",
        "KZ": "Kazakhstan",
        "KE": "Kenya",
        "KI": "Kiribati",
        "KP": "North Korea",
        "KR": "South Korea",
        "KW": "Kuwait",
        "KG": "Kyrgyzstan",
        "LA": "Laos",
        "LV": "Latvia",
        "LB": "Lebanon",
        "LS": "Lesotho",
        "LR": "Liberia",
        "LY": "Libya",
        "LI": "Liechtenstein",
        "LT": "Lithuania",
        "LU": "Luxembourg",
        "MO": "Macao",
        "MG": "Madagascar",
        "MW": "Malawi",
        "MY": "Malaysia",
        "MV": "Maldives",
        "ML": "Mali",
        "MT": "Malta",
        "MH": "Marshall Islands",
        "MQ": "Martinique",
        "MR": "Mauritania",
        "MU": "Mauritius",
        "YT": "Mayotte",
        "MX": "Mexico",
        "FM": "Micronesia",
        "MD": "Moldova",
        "MC": "Monaco",
        "MN": "Mongolia",
        "ME": "Montenegro",
        "MS": "Montserrat",
        "MA": "Morocco",
        "MZ": "Mozambique",
        "MM": "Myanmar",
        "NA": "Namibia",
        "NR": "Nauru",
        "NP": "Nepal",
        "NL": "Netherlands",
        "NC": "New Caledonia",
        "NZ": "New Zealand",
        "NI": "Nicaragua",
        "NE": "Niger",
        "NG": "Nigeria",
        "NU": "Niue",
        "NF": "Norfolk Island",
        "MK": "North Macedonia",
        "MP": "Northern Mariana Islands",
        "NO": "Norway",
        "OM": "Oman",
        "PK": "Pakistan",
        "PW": "Palau",
        "PS": "Palestine",
        "PA": "Panama",
        "PG": "Papua New Guinea",
        "PY": "Paraguay",
        "PE": "Peru",
        "PH": "Philippines",
        "PN": "Pitcairn",
        "PL": "Poland",
        "PT": "Portugal",
        "PR": "Puerto Rico",
        "QA": "Qatar",
        "RE": "Réunion",
        "RO": "Romania",
        "RU": "Russia",
        "RW": "Rwanda",
        "BL": "Saint Barthélemy",
        "SH": "Saint Helena, Ascension and Tristan da Cunha",
        "KN": "Saint Kitts and Nevis",
        "LC": "Saint Lucia",
        "MF": "Saint Martin (French part)",
        "PM": "Saint Pierre and Miquelon",
        "VC": "Saint Vincent and the Grenadines",
        "WS": "Samoa",
        "SM": "San Marino",
        "ST": "Sao Tome and Principe",
        "SA": "Saudi Arabia",
        "SN": "Senegal",
        "RS": "Serbia",
        "SC": "Seychelles",
        "SL": "Sierra Leone",
        "SG": "Singapore",
        "SX": "Sint Maarten (Dutch part)",
        "SK": "Slovakia",
        "SI": "Slovenia",
        "SB": "Solomon Islands",
        "SO": "Somalia",
        "ZA": "South Africa",
        "GS": "South Georgia and the South Sandwich Islands",
        "SS": "South Sudan",
        "ES": "Spain",
        "LK": "Sri Lanka",
        "SD": "Sudan",
        "SR": "Suriname",
        "SJ": "Svalbard and Jan Mayen",
        "SE": "Sweden",
        "CH": "Switzerland",
        "SY": "Syria",
        "TW": "Taiwan",
        "TJ": "Tajikistan",
        "TZ": "Tanzania",
        "TH": "Thailand",
        "TL": "Timor-Leste",
        "TG": "Togo",
        "TK": "Tokelau",
        "TO": "Tonga",
        "TT": "Trinidad and Tobago",
        "TN": "Tunisia",
        "TR": "Turkey",
        "TM": "Turkmenistan",
        "TC": "Turks and Caicos Islands",
        "TV": "Tuvalu",
        "UG": "Uganda",
        "UA": "Ukraine",
        "AE": "United Arab Emirates",
        "GB": "United Kingdom",
        "US": "United States",
        "UM": "United States Minor Outlying Islands",
        "UY": "Uruguay",
        "UZ": "Uzbekistan",
        "VU": "Vanuatu",
        "VE": "Venezuela",
        "VN": "Vietnam",
        "VG": "British Virgin Islands",
        "VI": "U.S. Virgin Islands",
        "WF": "Wallis and Futuna",
        "EH": "Western Sahara",
        "YE": "Yemen",
        "ZM": "Zambia",
        "ZW": "Zimbabwe",
    }
    return alpha2_to_name_map.get(alpha2_code, alpha2_code)

  country_data = song_df.groupby("conn_country")["ms_played"].sum().reset_index()
  country_data["total_minutes"] = country_data["ms_played"] / 60000
  country_data["alpha3_code"] = country_data["conn_country"].apply(alpha2_to_alpha3)
  country_data["country_name"] = country_data["conn_country"].apply(alpha2_to_name)
  fig = px.choropleth(
    country_data,
    locations="alpha3_code",
    locationmode="ISO-3",
    color="total_minutes",
    hover_name="country_name",
    hover_data={"total_minutes": True},
    color_continuous_scale="Viridis",
    title="Total Minutes Played by Country",
  )

  fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
  fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>"
                  "%{z:.1f} minutes<extra></extra>"
  )

  return fig

# Authenticate the user with Spotify
def get_spotify_oauth():
    redirect = "https://spotifydataanalysis.streamlit.app"
    scope = "playlist-modify-public playlist-modify-private"
    return SpotifyOAuth(
        client_id=st.secrets['spotify_client_id'],
        client_secret=st.secrets['spotify_secret_id'],
        redirect_uri=redirect,
        scope=scope,
        cache_path=".spotify_cache",
        show_dialog=True,
    )

@st.cache_data(show_spinner=False)
def get_track_uris_for_playlist(agg, filtered_df):
    if isinstance(agg.index, pd.MultiIndex):
        # Get the top song names and artists from the aggregated data
        track_names = agg.index.get_level_values(1)  # master_metadata_track_name
        artist_names = agg.index.get_level_values(0)  # master_metadata_album_artist_name

        # Filter the original DataFrame to get the track URIs
        track_uris = filtered_df[
            (filtered_df["master_metadata_track_name"].isin(track_names)) &
            (filtered_df["master_metadata_album_artist_name"].isin(artist_names))
        ]["spotify_track_uri"].unique().tolist()
    else:
        # Fallback for non-MultiIndex (shouldn't happen for Songs)
        track_uris = []

    return track_uris

# Create a playlist in the user's account
def create_spotify_playlist(agg, filtered_df, playlist_name):
    sp = authenticate_spotify()  
    track_uris = get_track_uris_for_playlist(agg, filtered_df)
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
    sp.playlist_add_items(playlist["id"], track_uris)
    return playlist["external_urls"]["spotify"]

def get_access_token():
    token_url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{st.secrets['spotify_client_id']}:{st.secrets['spotify_secret_id']}".encode()).decode()

    response = requests.post(
        token_url,
        headers={"Authorization": f"Basic {auth_header}"},
        data={"grant_type": "client_credentials"},
    )
    response.raise_for_status()
    return response.json()["access_token"]

@st.cache_data(show_spinner=False)
def extract_data(uploaded_file):
  zip_data = io.BytesIO(uploaded_file.read())
  songs = []
  podcasts = []
  target_folder = "Spotify Extended Streaming History"
  with zipfile.ZipFile(zip_data) as zip_ref:
      for file_info in zip_ref.infolist():
        if file_info.filename.startswith(target_folder + "/") and file_info.filename.endswith(".json"):
          with zip_ref.open(file_info) as file:
              data = json.load(file)
              for entry in data:
                  if entry["spotify_track_uri"] is not None:
                      songs.append(entry)
                  else:
                      podcasts.append(entry)

        # Create DataFrame in memory
  song_df = pd.DataFrame(songs)
  song_df["year"] = pd.to_datetime(song_df["ts"]).dt.year

  return song_df, podcasts

plt.rcParams['text.usetex'] = False

# Title
st.title("All-Time Spotify Recap")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["File Upload", "Your Favorites", "Your Music Timeline", "Favorites per Period", "Your Biggest Addictions", "Your Styles", "World Map", "Suggestions"])

with tab1:
  uploaded_file = st.file_uploader("Upload your ZIP file", type=["zip"])

  if uploaded_file:
    try:
      song_df, podcasts = extract_data(uploaded_file)
      st.session_state.song_df = song_df

      st.success("File uploaded successfully!")
    except Exception as e:
      st.error("Something went wrong with your file. Are you sure you are uploading the right file?")
      uploaded_file = None

  st.write("How to get your file?")
  st.write("Go to https://www.spotify.com/us/account/privacy and scroll down to \"Download Your Data\".")
  st.write("Check \"Select Extended streaming history\" and click on \"Request Data\".")
  st.write("You will receive an email confirming your request. Wait for a few days (they say 30 days but to me it took 1 day) and Spotify should send your ZIP by email!")

with tab2:
    if uploaded_file:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            years = ["All"] + sorted(st.session_state.song_df["year"].unique())
            selected_year = st.selectbox("Filter by Year:", years, key="year1")
        with col2:
            entries = st.selectbox("Number of Entries:", [10, 25, 50, 100, 200, 300, 500], key="entries1")
        with col3:
            topic = st.selectbox("Entity:", ["Songs", "Artists", "Albums"], key="entity1")
        with col4:
            style = st.selectbox("Metric:", ["# Plays", "Total Minutes"], key="metric1")

        if selected_year != "All":
            stacked = False
            st.checkbox("Stacked values per year?", value=False, disabled=True)
        else:
            stacked = st.checkbox("Stacked values per year?", value=False)

        minutes = style == "Total Minutes"
        if selected_year != "All":
          filtered_df = song_df[song_df["year"] == selected_year]
        else:
          filtered_df = song_df

        agg = get_top_plot_data(filtered_df, topic, minutes, stacked)
        agg_top = agg[0:entries]

        if topic == "Songs":
          
          params = st.query_params
          code = params.get("code")
          # STEP 1 → Button: Begin login flow
          if st.button("Create Playlist") and code is None:
              
              auth_url = get_spotify_oauth().get_authorize_url()
              st.markdown(
              f"<meta http-equiv='refresh' content='0; url={auth_url}'>",
              unsafe_allow_html=True
              )

          # STEP 2 → If redirected back with ?code=...
          elif code:
              oauth = get_spotify_oauth()
              token_info = oauth.get_cached_token(code)
              sp = spotipy.Spotify(auth=token_info["access_token"])

              track_uris = get_track_uris_for_playlist(agg_top, filtered_df)

              # Create playlist
              user_id = sp.current_user()["id"]
              playlist_name = "My Top Songs Playlist"
              playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
              sp.playlist_add_items(playlist["id"], track_uris)
      
              st.success(f"Playlist created! [Open in Spotify]({playlist['external_urls']['spotify']})")
                    # if st.button("Generate Spotify Playlist"):
          #   try:            
          #       playlist_url = create_spotify_playlist(filtered_df, agg_top, "My Top Songs Playlist")
          #       st.success(f"Playlist created! [Open in Spotify]({playlist_url})")
          #   except Exception as e:
          #       st.error(f"Failed to create playlist: {e}")
        plot_container = st.empty()
        with st.spinner("Drawing your chart..."):
          plt = get_top_plot(agg_top, topic, minutes, stacked)
        plot_container.pyplot(plt)
    else:
        st.write("Please upload your file first.")

with tab3:
    if uploaded_file:
        granularity = st.selectbox(
            "Which granularity do you want?",
            ["Week", "Month", "Year"],
            index=1, key="granularity1")

        plot_container = st.empty()
        fig = get_timeline_plot(st.session_state.song_df, granularity)
        plot_container.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Please upload your file first.")

# Apply the same pattern to tabs 4 and 5
with tab4:
    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        with col1:
            entity = st.selectbox("Entity:", ["Song", "Artist", "Album"], key="entity2")
        with col2:
            metric = st.selectbox("Metric:", ["# Plays","Total Minutes"], key="metric2")
        with col3:
            granularity = st.selectbox("Granularity:", ["Month", "Year"], key="granularity2")

        plot_container = st.empty()
        plt = get_top_per_period_plot(st.session_state.song_df, entity, metric, granularity)
        plot_container.pyplot(plt)

    else:
        st.write("Please upload your file first.")

with tab5:
    if uploaded_file:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            entity_type = st.selectbox("Entity Type:", ["Song", "Artist", "Album"], key="entity3")
        with col2:
            entries = st.selectbox("Number of entries:", [10, 25, 50, 100, 200, 300, 500], key="entries2")
        with col3:
            metric = st.selectbox("Metric:", ["# Plays", "Total Minutes"], key="metric3")
        with col4:
            granularity = st.selectbox("Granularity:", ["Day", "Week", "Month", "Year"], index=2, key="granularity3")

        plot_container = st.empty()
        plt = get_top_listening_combos(st.session_state.song_df, entries, metric, granularity, entity_type)
        plot_container.pyplot(plt)

    else:
        st.write("Please upload your file first.")

with tab6:
  if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
      entries = st.selectbox("Number of Entries:", [10, 25, 50], key="genre_entries")
    with col2:
      available_years = sorted(song_df["year"].unique().tolist())
      available_years = ["All"] + available_years
      year = st.selectbox("Filter by Year:", available_years)
    plot_container = st.empty()
    with st.spinner("Fetching genres and drawing chart..."):
        try:
            fig = get_top_genres_plot(st.session_state.song_df, entries, year)
            plot_container.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate chart: {e}")
  else:
        st.write("Please upload your file first.")

with tab7:
  if uploaded_file:
    plot_container = st.empty()
    with st.spinner("Drawing map..."):
      fig = get_map_plot(st.session_state.song_df)
      plot_container.plotly_chart(fig, use_container_width=True)
  else:
        st.write("Please upload your file first.")
with tab8:
    st.text("Did you find a bug or have any suggestion? Contact me and I'll try to do it asap!")

    with st.form("suggestions_form"):
        name = st.text_input("Your Name (Optional)")
        st.markdown("Your Comment <span style='color: red'>*</span>", unsafe_allow_html=True)
        comment = st.text_area("", key="comment", height=150, label_visibility="collapsed")

        submitted = st.form_submit_button("Submit")

        if submitted:
            if not comment:
                st.error("Please fill in the comment field!")
            else:
                try:
                    # JavaScript code to send email via EmailJS with Public Key
                    js_code = f"""
                    <script src="https://cdn.jsdelivr.net/npm/@emailjs/browser@3/dist/email.min.js"></script>
                    <script>
                        (function() {{
                            emailjs.init('{st.secrets['email_public_key']}'); // Replace with your Public Key

                            var params = {{
                                name: "{name}",
                                comment: `{comment}`,
                                reply_to: "{name}",
                            }};

                            emailjs.send('{st.secrets['email_service_id']}', '{st.secrets['email_template_id']}', params) // Replace with your Service ID and Template ID
                                .then(function(response) {{
                                    document.write('<p style="color: green;">Thank you for your feedback! 🎉</p>');
                                    console.log('SUCCESS!', response.status, response.text);
                                }}, function(error) {{
                                    document.write('<p style="color: red;">Failed to send feedback. Please try again later.</p>');
                                    console.log('FAILED...', error);
                                }});
                        }}());
                    </script>
                    """

                    st.components.v1.html(js_code, height=100)
                except Exception as e:
                    st.error(f"Error: {e}")
