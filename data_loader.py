try:
    import pandas as pd
except Exception:
    pd = None
import random
import time
import concurrent.futures
import os
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable
import json
import threading

# Prefer `requests` when available, fall back to stdlib `urllib` when not.
try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:
    _REQUESTS_AVAILABLE = False
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error

# Prefer `TextBlob` when available, fall back to a very small rule-based
# sentiment function if it's not installed so the script can run without
# external dependencies for quick checks.
try:
    from textblob import TextBlob  # Used for sentiment scoring
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False

STEAM_APP_LIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
# Additional app-list endpoints to try (some environments or mirrors may differ)
STEAM_APP_LIST_ALTERNATIVES = [
    "https://api.steampowered.com/ISteamApps/GetAppList/v2/",
    "https://api.steampowered.com/ISteamApps/GetAppList/v2",
    "http://api.steampowered.com/ISteamApps/GetAppList/v2/",
]
STEAM_APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails?appids={}"
STEAM_REVIEWS_URL = "https://store.steampowered.com/appreviews/{}?json=1&num_per_page={}&filter=recent"
FAILED_APP_IDS_LOG = "failed_appids.log"

# When a local reviews CSV is used as a final dataset (fallback), this
# flag is set so callers know to return it directly.
_LOCAL_REVIEWS_FALLBACK = False

# Global rate-limit between outbound HTTP calls (seconds). Can be overridden
# with environment variable STEAM_REQUEST_INTERVAL. Uses a module-level lock
# and timestamp to serialize requests across threads.
_REQUEST_LOCK = threading.Lock()
try:
    _LAST_REQUEST_TIME = float(0.0)
except Exception:
    _LAST_REQUEST_TIME = 0.0
try:
    _REQUEST_MIN_INTERVAL = float(os.environ.get('STEAM_REQUEST_INTERVAL', 1.5))
except Exception:
    _REQUEST_MIN_INTERVAL = 1.5


if pd is None:
    import csv

    class SimpleDataFrame:
        def __init__(self, data):
            self._data = data or []

        def to_csv(self, path, index=False):
            if not self._data:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    f.write("")
                return
            keys = sorted({k for d in self._data for k in d.keys()})
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self._data:
                    writer.writerow(row)

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def to_dicts(self):
            return self._data



def _http_get_json(url, timeout=15):
    """Get JSON from URL using requests if available, else urllib.

    Returns a tuple (status_code, headers_dict, parsed_json) or raises.
    """
    # Enforce a minimum interval between outbound HTTP requests across
    # threads to reduce hitting remote rate limits (e.g., HTTP 429).
    try:
        with _REQUEST_LOCK:
            now = time.time()
            elapsed = now - _LAST_REQUEST_TIME
            wait = _REQUEST_MIN_INTERVAL - elapsed
            if wait > 0:
                time.sleep(wait)
            # update last request timestamp
            _LAST_REQUEST_TIME = time.time()
    except Exception:
        # if anything goes wrong with timing, continue without delaying
        pass

    if _REQUESTS_AVAILABLE:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.status_code, resp.headers, resp.json()
    # urllib fallback
    req = _urllib_request.Request(url, headers={"User-Agent": "python-urllib/3"})
    try:
        with _urllib_request.urlopen(req, timeout=timeout) as r:
            status = getattr(r, 'status', 200)
            headers = dict(r.getheaders())
            body = r.read()
            try:
                parsed = json.loads(body.decode('utf-8'))
            except Exception:
                parsed = {}
            return status, headers, parsed
    except _urllib_error.HTTPError as e:
        # mimic requests behaviour for 429
        raise



def get_steam_app_list():
    def _append_key(u, key):
        if not key:
            return u
        joiner = '&' if ('?' in u) else '?'
        return f"{u}{joiner}key={key}"

    api_key = os.environ.get('STEAM_API_KEY')
    # Try multiple endpoints until one returns the expected structure
    last_err = None
    for base in STEAM_APP_LIST_ALTERNATIVES:
        try:
            url = _append_key(base, api_key)
            status, headers, data = _http_get_json(url)
            apps = data.get('applist', {}).get('apps', [])
            if apps:
                if pd is not None:
                    return pd.DataFrame(apps)
                return apps
        except Exception as e:
            last_err = e
            continue

    print(f"Error fetching app list from network: {last_err}")
    # Try SteamSpy as a fallback source for app metadata (works without Steam API)
    try:
        steamspy_url = 'https://steamspy.com/api.php?request=all'
        status, headers, data = _http_get_json(steamspy_url)
        if isinstance(data, dict) and len(data) > 0:
            apps = []
            for k, v in data.items():
                try:
                    appid = int(v.get('appid', k))
                except Exception:
                    try:
                        appid = int(k)
                    except Exception:
                        continue
                apps.append({'appid': appid, 'name': v.get('name', '')})
            if apps:
                print('Loaded app list from SteamSpy (fallback).')
                if pd is not None:
                    return pd.DataFrame(apps)
                return apps
    except Exception:
        pass
    # Fallback: try loading local CSV files that may already contain app data
    local_paths = [
        os.path.join(os.getcwd(), 'all_steam_games_with_reviews.csv'),
        os.path.join(os.getcwd(), 'models', 'games_data.csv'),
    ]
    for p in local_paths:
        try:
            if os.path.exists(p):
                if pd is not None:
                    # First inspect raw header line to robustly detect if this CSV
                    # already contains review content (some CSVs may have messy
                    # column parsing that hides the column name in df.columns).
                    try:
                        with open(p, 'r', encoding='utf-8') as fh:
                            first = fh.readline().lower()
                    except Exception:
                        first = ''
                    if 'reviews' in first or 'sentiment_score' in first:
                        try:
                            df = pd.read_csv(p)
                        except Exception:
                            df = pd.read_csv(p, engine='python', encoding='utf-8', on_bad_lines='skip')
                        global _LOCAL_REVIEWS_FALLBACK
                        _LOCAL_REVIEWS_FALLBACK = True
                        print(f"Loaded local reviews CSV from {p} (using as final dataset)")
                        return df
                    # fallback: try reading with pandas and look for appid
                    try:
                        df = pd.read_csv(p)
                    except Exception:
                        df = pd.read_csv(p, engine='python', encoding='utf-8', on_bad_lines='skip')
                    if 'appid' in df.columns:
                        print(f"Loaded local app list from {p}")
                        return df
                    # otherwise return whatever we read
                    return df
                else:
                    # read minimal CSV without pandas
                    apps = []
                    with open(p, 'r', encoding='utf-8') as f:
                        header = f.readline().strip().split(',')
                        if 'appid' in header:
                            idx = header.index('appid')
                            for line in f:
                                parts = line.strip().split(',')
                                if len(parts) > idx:
                                    apps.append({'appid': parts[idx]})
                    if apps:
                        print(f"Loaded local app list from {p}")
                        return apps
        except Exception:
            continue

    # Final fallback: empty structure
    print("No app list available (network failed and no local CSV found).")
    if pd is not None:
        return pd.DataFrame()
    return []

def get_app_details(appid):
    url = STEAM_APP_DETAILS_URL.format(appid)
    retries = 3
    backoff_factor = 1
    for attempt in range(retries):
        try:
            status, headers, payload = _http_get_json(url)
            if status == 429 or int(headers.get("Retry-After", 0) or 0) > 0:
                retry_after = int(headers.get("Retry-After", 5) or 5)
                time.sleep(retry_after * backoff_factor)
                backoff_factor *= 2
                continue
            data = payload.get(str(appid), {})
            if not data.get('success', False):
                return None
            return data.get('data', {})
        except Exception as e:
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
            status, headers, data = _http_get_json(url)
            if status == 429:
                retry_after = int(headers.get("Retry-After", 5) or 5)
                time.sleep(retry_after)
                continue
            if "reviews" not in data:
                return []
            return [review.get("review", "") for review in data["reviews"]]
        except Exception as e:
            if attempt == retries - 1:
                print(f"Error fetching reviews for app {appid}: {e}")
                return []
            time.sleep(5)


def _sanitize_app_ids(apps_df):
    """Return a list of numeric appids extracted from apps_df.

    Accepts a pandas DataFrame (with an `appid` column) or a list of dicts
    (each with an `appid` key). Filters out non-numeric values and duplicates.
    """
    ids = []
    if apps_df is None:
        return []
    # pandas DataFrame
    if pd is not None and isinstance(apps_df, pd.DataFrame):
        if 'appid' not in apps_df.columns:
            return []
        raw = apps_df['appid'].tolist()
    else:
        # assume iterable of mappings
        try:
            raw = [a.get('appid') for a in apps_df]
        except Exception:
            return []

    seen = set()
    for v in raw:
        if v is None:
            continue
        # try to coerce to int
        try:
            iv = int(v)
        except Exception:
            # sometimes values like '2012670' are strings with whitespace
            try:
                s = str(v).strip()
                iv = int(s)
            except Exception:
                continue
        if iv in seen:
            continue
        seen.add(iv)
        ids.append(iv)
    return ids

def analyze_sentiment(text):
    if _TEXTBLOB_AVAILABLE:
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            return 0.0
    # Simple fallback: basic polarity by counting positive/negative words
    try:
        pos_words = set(["good", "great", "love", "excellent", "fun", "enjoy", "awesome", "best"]) 
        neg_words = set(["bad", "terrible", "hate", "awful", "worst", "boring", "poor"]) 
        text_l = text.lower()
        pos = sum(text_l.count(w) for w in pos_words)
        neg = sum(text_l.count(w) for w in neg_words)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    except Exception:
        return 0.0

def fetch_game_reviews(num_games=5000, num_reviews_per_game=20, max_workers=5):
    apps_df = get_steam_app_list()

    # Extract numeric app ids (from DataFrame or list). If we get none,
    # only then consider returning a local reviews CSV (fallback).
    candidate_ids = _sanitize_app_ids(apps_df)
    if not candidate_ids:
        # if get_steam_app_list returned a DataFrame that already contains
        # completed reviews (local CSV), return it as the final dataset.
        if pd is not None and isinstance(apps_df, pd.DataFrame) and any(col in apps_df.columns for col in ("reviews", "sentiment_score", "name")):
            print("Using local reviews CSV as final dataset (fallback).")
            return apps_df
        print("Warning: No numeric Steam app ids available after fetching app list.")
        if pd is not None:
            return pd.DataFrame()
        return SimpleDataFrame([])

    # Sample up to num_games from candidate_ids deterministically
    sample_count = min(num_games, len(candidate_ids))
    rnd = random.Random(42)
    app_ids = rnd.sample(candidate_ids, sample_count)
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
    if pd is not None:
        return pd.DataFrame(game_data)
    return SimpleDataFrame(game_data)

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
