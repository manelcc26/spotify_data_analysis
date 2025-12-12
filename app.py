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

CACHE_FILE = "artist_genres_cache.json"

matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()

st.set_page_config(layout="wide")
st.set_page_config(page_title="All-Time Spotify Recap", page_icon="🎵")

@st.cache_data(show_spinner=False)
def get_top_plot_data(song_df, selected_year, topic, minutes, stacked):
  if selected_year != "All":
      filtered_df = song_df[song_df["year"] == selected_year]
  else:
      filtered_df = song_df
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

def get_top_plot(song_df, selected_year, topic, entries, minutes, stacked):

    agg = get_top_plot_data(song_df, selected_year, topic, minutes, stacked)
    agg_top = agg[0:entries]
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
            ax.text(v + 1, i, f"{int(v)}", va='center', fontsize=10)
    else:
        for i, (name, v) in enumerate(agg_top.items()):
            ax.text(v + 1, i, f"{int(v)}", va='center', fontsize=10)

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["File Upload", "Your Favorites", "Your Music Timeline", "Favorites per Period", "Your Biggest Addictions", "Your Styles", "Suggestions"])

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

        plot_container = st.empty()
        with st.spinner("Drawing your chart..."):
          plt = get_top_plot(st.session_state.song_df, selected_year, topic, entries, minutes, stacked)
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
