from datetime import datetime, timedelta
import os
import json
import requests
import base64

def get_week_range(week_str):
          year, week = week_str.split("-W")
          year = int(year)
          week = int(week)
          first_day = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
          last_day = first_day + timedelta(days=6)
          return f"{first_day.strftime('%d/%m/%y')} - {last_day.strftime('%d/%m/%y')}"

def format_month(month_str):
          year, month = month_str.split("-")
          month_name = datetime.strptime(month, "%m").strftime("%b")
          return f"{month_name} {year}"

def week_to_datetime(week_str):
          year, week = map(int, week_str.split("-W"))
          return datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")  # Monday of the week
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

def truncate_name(x0, x1, max_len=60):
  if len(x0 + x1) > max_len:
    position = max_len - len(x1) - 3
    return x0[:(position if position > 0 else 1)] + "..." + " (" + x1 + ")"
  else:
    return f"{x0} ({x1})"

def load_genre_cache(filename):
    """Load the genre cache from the JSON file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def save_genre_cache(cache, filename):
    """Save the genre cache to the JSON file."""
    with open(filename, "w") as f:
        json.dump(cache, f, indent=2)

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
