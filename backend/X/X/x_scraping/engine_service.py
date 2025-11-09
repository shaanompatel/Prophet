# engine_service.py

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv

import asyncio
import websockets

from market_engine import MarketEngine, LOG_FILE

# load envs before importing MarketEngine so os.getenv works there
env_path = Path(__file__).parent / "bearerkey.env"
load_dotenv(dotenv_path=env_path)

engine = MarketEngine()
app = Flask(__name__)

ACTION_SERVER_HOST = "0.0.0.0"
ACTION_SERVER_PORT = 8766

ACTION_CLIENTS = set()


# ------------- websocket action server -------------

async def action_ws_handler(websocket):
    """
    Websocket handler for the action server.

    Any client that connects and sends a message will have that message
    broadcast to all other connected clients. MarketEngine connects as a
    client and sends JSON actions. Listeners (like sample_market_listener.py)
    connect and receive those actions.
    """
    print(f"[ACTION_SERVER] Client connected: {websocket.remote_address}")
    ACTION_CLIENTS.add(websocket)
    try:
        async for message in websocket:
            dead = []
            for client in list(ACTION_CLIENTS):
                # do not echo back to sender
                if client is websocket:
                    continue
                try:
                    await client.send(message)
                except Exception as e:
                    print(f"[ACTION_SERVER] Error sending to {getattr(client, 'remote_address', '?')}: {e}")
                    dead.append(client)

            for d in dead:
                ACTION_CLIENTS.discard(d)
    except Exception as e:
        print(f"[ACTION_SERVER] Handler error: {e}")
    finally:
        print(f"[ACTION_SERVER] Client disconnected: {websocket.remote_address}")
        ACTION_CLIENTS.discard(websocket)


async def run_action_server():
    async with websockets.serve(
        action_ws_handler,
        ACTION_SERVER_HOST,
        ACTION_SERVER_PORT,
    ):
        print(f"[ACTION_SERVER] listening on ws://localhost:{ACTION_SERVER_PORT}")
        await asyncio.Future()


def start_action_server_thread():
    def runner():
        asyncio.run(run_action_server())

    t = threading.Thread(target=runner, daemon=True)
    t.start()


# ------------- background worker for clustering and resolution -------------

def background_worker():
    while True:
        try:
            engine.maybe_cluster_orphans()
            engine.maybe_check_resolutions()
        except Exception as e:
            print("[background_worker] error:", repr(e))
        time.sleep(30.0)


# ------------- HTTP routes -------------

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
        import traceback
        print("ERROR during ingest_manual:\n", traceback.format_exc())
        return jsonify({"error": "ingest_failed", "detail": str(e)}), 500

    return jsonify({"ingested": ingested})


@app.route("/cluster_now", methods=["POST"])
def cluster_now():
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


@app.route("/test_grok", methods=["GET"])
def test_grok():
    system_prompt = "You are a test bot that replies briefly."
    user_prompt = "Say hello in one short sentence."

    content = engine._grok_chat(system_prompt, user_prompt, max_tokens=32)
    return jsonify({"result": content})


def main():
    t = threading.Thread(target=background_worker, daemon=True)
    t.start()

    start_action_server_thread()

    print("Engine service on http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()
