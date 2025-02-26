# src/data_loader.py
"""
This module retrieves and preprocesses data from Steam using the official API.
"""

import requests
import pandas as pd
import time

def get_steam_app_list():
    """
    Retrieve the list of Steam apps using the official API.
    
    :return: DataFrame of apps.
    """
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching app list:", response.status_code)
        return None
    data = response.json()
    apps = data.get('applist', {}).get('apps', [])
    return pd.DataFrame(apps)

def get_app_details(api_key, appid):
    """
    Retrieve detailed information for a specific app using Steam's Store API.
    
    :param api_key: str, your Steam API key.
    :param appid: int, the Steam app ID.
    :return: Dictionary containing the app details.
    """
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching app details for appid {appid}: {response.status_code}")
        return None
    data = response.json()
    return data.get(str(appid), {})

def search_game_by_name(game_name, apps_df):
    """
    Search for a game by name in the provided DataFrame of Steam apps.
    
    :param game_name: str, the game name (or part of it) to search for.
    :param apps_df: DataFrame, must contain at least the 'appid' and 'name' columns.
    :return: DataFrame containing games that match the search term.
    """
    return apps_df[apps_df['name'].str.contains(game_name, case=False, na=False)]

def get_app_reviews(appid, num_reviews=20):
    """
    Retrieve reviews for a specific app using the Steam app reviews endpoint.
    
    :param appid: int, the Steam app ID.
    :param num_reviews: int, the number of reviews to fetch.
    :return: A concatenated string of reviews.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent&language=all&purchase_type=all&num_per_page={num_reviews}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching reviews for appid {appid}: {response.status_code}")
        return ""
    data = response.json()
    
    if 'reviews' not in data or not data['reviews']:
        print(f"No reviews found for appid {appid}")
        return ""

    review_texts = [review.get('review', '') for review in data['reviews']]
    return " ".join(review_texts)

def fetch_game_reviews(api_key, num_games=50, num_reviews=20):
    """
    Fetch reviews for a selection of Steam games.

    :param api_key: str, Steam API key.
    :param num_games: int, number of games to fetch reviews for.
    :param num_reviews: int, number of reviews per game.
    :return: DataFrame containing game names, app IDs, and reviews.
    """
    df_games = get_steam_app_list()
    if df_games is None:
        return None

    print(f"Total games available on Steam: {len(df_games)}")

    # Sample a subset of games (or limit to `num_games`)
    df_games = df_games.sample(min(num_games, len(df_games)))

    # Fetch reviews for each game
    review_list = []
    for index, row in df_games.iterrows():
        appid = row['appid']
        print(f"Fetching reviews for {row['name']} (appid: {appid})...")
        reviews = get_app_reviews(appid, num_reviews)
        review_list.append(reviews)
        time.sleep(1)  # Adding a delay to avoid overwhelming the Steam API

    # Add reviews to DataFrame
    df_games['Reviews'] = review_list

    # Remove games with empty reviews
    df_games = df_games[df_games['Reviews'].str.strip() != ""]

    return df_games

if __name__ == '__main__':
    # For demonstration, we're using your API key directly.
    # NOTE: In production, store your API key securely and load it from an environment variable.
    api_key = "63BF59BF95391DEA2787321623A2C77E"

    # Retrieve the app list and display the first 5 apps
    df_apps = get_steam_app_list()
    if df_apps is not None:
        print("First 5 apps from Steam:")
        print(df_apps.head())

    # Fetch a dataset of games with reviews
    df_with_reviews = fetch_game_reviews(api_key, num_games=100, num_reviews=20)
    if df_with_reviews is not None:
        print("First 5 games with reviews:")
        print(df_with_reviews.head())

    # Save the dataset for use in the recommendation model
    df_with_reviews.to_csv("all_steam_games_with_reviews.csv", index=False)
    print("Dataset saved successfully.")
