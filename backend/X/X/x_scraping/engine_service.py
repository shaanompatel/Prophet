import os
import threading
import time
from datetime import datetime, timezone

from flask import Flask, request, jsonify

from dotenv import load_dotenv
from pathlib import Path

# load envs before importing MarketEngine so os.getenv works there
env_path = Path(__file__).parent / "bearerkey.env"
load_dotenv(dotenv_path=env_path)

from X.x_scraping.market_engine import MarketEngine, LOG_FILE
from X.x_scraping.market_engine import MarketEngine, LOG_FILE

engine = MarketEngine()
app = Flask(__name__)


def background_worker():
    """
    Periodically:
      - try to form new markets from orphans
      - ask Grok if any markets are resolved
    """
    while True:
        try:
            engine.maybe_cluster_orphans()
            engine.maybe_check_resolutions()
        except Exception as e:
            print("[background_worker] error:", repr(e))
        time.sleep(30.0)


@app.route("/health", methods=["GET"])
def health():
    snap = engine.snapshot()
    return jsonify(
        {
            "ok": True,
            "buffer_size": snap["buffer_size"],
            "markets": snap["markets"],
            "log_file": LOG_FILE,
        }
    )
@app.route("/ingest", methods=["POST"])
def ingest_manual():
    """
    Accept JSON:
      either a list of tweets
      or {"tweets": [ ... ]}

    Each tweet can provide:
      id (optional), text, created_at, followers, verified,
      likes, retweets, replies, quotes
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({"error": "invalid_json", "detail": str(e)}), 400

    if isinstance(data, list):
        tweets = data
    elif isinstance(data, dict):
        tweets = data.get("tweets", [])
    else:
        return jsonify({"error": "bad_payload_type", "got": str(type(data))}), 400

    ingested = 0
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        for t in tweets:
            tweet = {
                "id": t.get("id") or f"manual-{int(time.time() * 1000)}-{ingested}",
                "text": t.get("text", ""),
                "created_at": t.get("created_at", now_iso),
                "user_id": t.get("user_id", "manual"),
                "followers": t.get("followers", 0),
                "verified": t.get("verified", False),
                "likes": t.get("likes", 0),
                "retweets": t.get("retweets", 0),
                "replies": t.get("replies", 0),
                "quotes": t.get("quotes", 0),
            }
            engine.ingest_tweet(tweet, source="manual")
            ingested += 1
    except Exception as e:
        # this is the part that was causing a 500
        # log to stdout so you see a traceback in the engine terminal
        import traceback
        print("ERROR during ingest_manual:\n", traceback.format_exc())
        return jsonify({"error": "ingest_failed", "detail": str(e)}), 500

    return jsonify({"ingested": ingested})
@app.route("/cluster_now", methods=["POST"])
def cluster_now():
    """
    Force a clustering and resolution check right now.
    Returns current markets and buffer size.
    """
    try:
        engine.maybe_cluster_orphans()
        engine.maybe_check_resolutions()
        snap = engine.snapshot()
        return jsonify(
            {
                "ok": True,
                "buffer_size": snap["buffer_size"],
                "markets": snap["markets"],
            }
        )
    except Exception as e:
        import traceback
        print("ERROR during cluster_now:\n", traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 500



def main():
    t = threading.Thread(target=background_worker, daemon=True)
    t.start()

    print("Engine service on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()

