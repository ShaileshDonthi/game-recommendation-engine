import requests
import pandas as pd
import time
import concurrent.futures
import os
from tqdm import tqdm
from textblob import TextBlob  # Used for sentiment scoring

STEAM_APP_LIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
STEAM_APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails?appids={}"
STEAM_REVIEWS_URL = "https://store.steampowered.com/appreviews/{}?json=1&num_per_page={}&filter=recent"
FAILED_APP_IDS_LOG = "failed_appids.log"

def get_steam_app_list():
    try:
        response = requests.get(STEAM_APP_LIST_URL)
        response.raise_for_status()
        data = response.json()
        apps = data.get('applist', {}).get('apps', [])
        return pd.DataFrame(apps)
    except requests.RequestException as e:
        print(f"Error fetching app list: {e}")
        return pd.DataFrame()

def get_app_details(appid):
    url = STEAM_APP_DETAILS_URL.format(appid)
    retries = 3
    backoff_factor = 1
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                time.sleep(retry_after * backoff_factor)
                backoff_factor *= 2
                continue
            response.raise_for_status()
            data = response.json().get(str(appid), {})
            if not data.get('success', False):
                return None
            return data.get('data', {})
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"Failed to fetch details for app {appid}: {e}")
                with open(FAILED_APP_IDS_LOG, "a") as log_file:
                    log_file.write(f"{appid}\n")
                return None
            time.sleep(5 * backoff_factor)
            backoff_factor *= 2

def get_app_reviews(appid, num_reviews=20):
    url = STEAM_REVIEWS_URL.format(appid, num_reviews)
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            data = response.json()
            if "reviews" not in data:
                return []
            return [review.get("review", "") for review in data["reviews"]]
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"Error fetching reviews for app {appid}: {e}")
                return []
            time.sleep(5)

def analyze_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0

def fetch_game_reviews(num_games=5000, num_reviews_per_game=20, max_workers=5):
    apps_df = get_steam_app_list()
    if len(apps_df) == 0:
        raise Exception("Failed to fetch Steam app list")
    app_ids = apps_df["appid"].sample(min(num_games, len(apps_df)), random_state=42).tolist()
    game_details = fetch_all_game_details(app_ids, max_workers)
    game_data = []

    for appid, details in tqdm(game_details.items(), desc="Fetching reviews"):
        if not details:
            continue
        reviews = get_app_reviews(appid, num_reviews_per_game)
        if not reviews:
            continue
        full_text = " ".join(reviews)
        sentiment = analyze_sentiment(full_text)
        game_info = {
            "appid": appid,
            "name": details.get("name", "N/A"),
            "type": details.get("type", "N/A"),
            "is_free": details.get("is_free", "N/A"),
            "release_date": details.get("release_date", {}).get("date", "N/A"),
            "genres": ", ".join([g["description"] for g in details.get("genres", [])]) if details.get("genres") else "N/A",
            "short_description": details.get("short_description", "N/A"),
            "reviews": full_text,
            "sentiment_score": sentiment
        }
        game_data.append(game_info)
    return pd.DataFrame(game_data)

def fetch_all_game_details(app_ids, max_workers=5):
    game_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_app_details, appid): appid for appid in app_ids}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching game details"):
            appid = futures[future]
            try:
                result = future.result()
                if result:
                    game_data[appid] = result
            except Exception as e:
                print(f"Error processing appid {appid}: {e}")
    return game_data

if __name__ == "__main__":
    df = fetch_game_reviews(num_games=10000, num_reviews_per_game=10)
    print(f"Fetched data for {len(df)} games")
    df.to_csv("all_steam_games_with_reviews.csv", index=False)
