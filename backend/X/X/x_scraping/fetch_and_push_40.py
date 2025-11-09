# fetch_and_push_40.py

import os
import json
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / "bearerkey.env"
SINCE_ID_PATH = BASE_DIR / "since_id.json"

# load X_BEARER_TOKEN from bearerkey.env
load_dotenv(dotenv_path=ENV_PATH)

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
if not X_BEARER_TOKEN:
    raise SystemExit("X_BEARER_TOKEN is not set. Put it in bearerkey.env")

SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
# very broad query: matches lots of English tweets, no retweets, no replies
QUERY = "(the OR to OR a OR of OR in OR for OR on OR is OR it OR that OR you OR i OR at OR this OR with OR my OR we OR be OR me OR your OR so OR not OR but OR have OR just OR from OR by OR as) lang:en -is:retweet -is:reply"
TWEETS_PER_BATCH = 40

ENGINE_INGEST_URL = "http://localhost:8000/ingest"
ENGINE_CLUSTER_URL = "http://localhost:8000/cluster_now"


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_since_id() -> str | None:
    if not SINCE_ID_PATH.exists():
        return None
    try:
        data = json.loads(SINCE_ID_PATH.read_text(encoding="utf-8"))
        return data.get("since_id")
    except Exception:
        return None


def save_since_id(since_id: str) -> None:
    SINCE_ID_PATH.write_text(
        json.dumps({"since_id": since_id}, indent=2),
        encoding="utf-8",
    )


def fetch_recent_batch():
    """
    Fetch up to TWEETS_PER_BATCH tweets newer than the last seen ID.
    """
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    since_id = load_since_id()

    params = {
        "query": QUERY,
        "max_results": TWEETS_PER_BATCH,
        "tweet.fields": "created_at,lang,public_metrics",
        "expansions": "author_id",
        "user.fields": "public_metrics,verified",
    }
    if since_id:
        params["since_id"] = since_id

    resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=20)

    if resp.status_code != 200:
        print("X API error:", resp.status_code, resp.text[:400])
        resp.raise_for_status()

    payload = resp.json()

    # update since_id using newest_id from meta, if present
    meta = payload.get("meta", {}) or {}
    newest_id = meta.get("newest_id")
    if newest_id:
        save_since_id(newest_id)

    return payload


def parse_search_result(payload: dict):
    tweets = payload.get("data", []) or []
    includes = payload.get("includes", {}) or {}
    users = includes.get("users", []) or []
    user_map = {u["id"]: u for u in users}

    result = []

    for t in tweets:
        tid = t["id"]
        author_id = t.get("author_id", "")
        user = user_map.get(author_id, {})
        pm = t.get("public_metrics", {}) or {}
        upm = user.get("public_metrics", {}) or {}

        rec = {
            "id": tid,
            "text": t.get("text", ""),
            "created_at": t.get("created_at"),
            "user_id": author_id,
            "followers": upm.get("followers_count", 0),
            "verified": user.get("verified", False),
            "likes": pm.get("like_count", 0),
            "retweets": pm.get("retweet_count", 0),
            "replies": pm.get("reply_count", 0),
            "quotes": pm.get("quote_count", 0),
        }
        result.append(rec)

    return result


def push_to_engine(tweets):
    resp = requests.post(ENGINE_INGEST_URL, json={"tweets": tweets}, timeout=30)
    if resp.status_code != 200:
        print("Engine ingest error:", resp.status_code, resp.text[:400])
        resp.raise_for_status()
    return resp.json()


def trigger_cluster():
    """
    Ask the engine to recluster orphans and check resolutions.
    """
    try:
        resp = requests.post(ENGINE_CLUSTER_URL, timeout=30)
        if resp.status_code != 200:
            print("Cluster_now error:", resp.status_code, resp.text[:200])
            return
        data = resp.json()
        mkts = data.get("markets", [])
        buf = data.get("buffer_size")
        print(
            f"Cluster_now ok: markets={len(mkts)}, buffer_size={buf}"
        )
    except Exception as e:
        print("Error calling cluster_now:", repr(e))


def main():
    print("Starting continuous fetch and push loop (40 tweets per minute)...")
    while True:
        try:
            print("\n[loop] Fetching up to 40 new recent tweets...")
            payload = fetch_recent_batch()
            tweets = parse_search_result(payload)
            print(f"[loop] Fetched {len(tweets)} tweets from X API")

            if tweets:
                # show a few samples
                for t in tweets[:3]:
                    print("  sample:", t["id"], "-", t["text"][:120])

                print("[loop] Sending tweets to engine service...")
                result = push_to_engine(tweets)
                print("[loop] Engine ingest result:", result)

                print("[loop] Triggering clustering...")
                trigger_cluster()
            else:
                print("[loop] No new tweets this round.")

        except Exception as e:
            print("[loop] Error in loop:", repr(e))

        print("[loop] Sleeping 60 seconds...")
        time.sleep(60.0)


if __name__ == "__main__":
    main()
