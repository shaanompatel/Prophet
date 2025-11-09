import asyncio
import websockets
import json
import time
import os
import re
from collections import deque
from dotenv import load_dotenv

from xai_sdk import Client
from xai_sdk.chat import system, user

# env variables
load_dotenv()  # reads from .env file

# --- CONFIGURATION ---
TWEET_SERVER_URI = "ws://localhost:8765"
ACTION_SERVER_HOST = "localhost"
ACTION_SERVER_PORT = 8766
BUFFER_SIZE = 10
HISTORY_SIZE = 100
XAI_MODEL = 'grok-3-mini'

AI_SYSTEM_PROMPT = """
You are an autonomous prediction market manager. Your job is to read
real-time text data ("NEW TWEETS") and compare it against a list of
"ACTIVE MARKETS" and "RECENT HISTORY" to make two decisions:

1.  **CREATE:** Detect *new*, verifiable, binary-outcome (YES/NO) events in
    the "NEW TWEETS" that are not in the "ACTIVE MARKETS" list.
2.  **RESOLVE:** Detect if any of the "ACTIVE MARKETS" have been *conclusively
    resolved* by information in the "NEW TWEETS".

-   Respond ONLY with a single, minified JSON object containing a list of actions.
-   If no new markets or resolutions are found, respond with: {"actions": []}

--- JSON Response Format ---
{
  "actions": [
    {
      "action": "CREATE",
      "market_name": "ShortCamelCaseName",
      "probability": 0.65,
      "reason": "The new tweets confirm X, which was only speculation."
    },
    {
      "action": "RESOLVE",
      "market_name": "ProjectChimeraLaunchNov10",
      "outcome": "NO",
      "reason": "Official @ProjectChimera tweet confirms delay."
    }
  ]
}

"probability" MUST be a float (0.0-1.0) for the 'YES' outcome.
"outcome" MUST be either "YES" or "NO".
"""

# --- Global "Memory" ---
ACTIVE_MARKETS = {}
HISTORICAL_CONTEXT = deque(maxlen=HISTORY_SIZE)
CONNECTED_ACTION_CLIENTS = set()


# --- WEBSOCKET SERVER LOGIC (Unchanged) ---

async def broadcast_action(action_json: str):
    if CONNECTED_ACTION_CLIENTS:
        print(f"[ACTION_BROADCAST] Sending to {len(CONNECTED_ACTION_CLIENTS)} clients: {action_json}")
        await asyncio.gather(*[client.send(action_json) for client in CONNECTED_ACTION_CLIENTS])

async def action_client_handler(websocket):
    print(f"[ACTION_SERVER] Client connected: {websocket.remote_address}")
    CONNECTED_ACTION_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        print(f"[ACTION_SERVER] Client disconnected: {websocket.remote_address}")
        CONNECTED_ACTION_CLIENTS.remove(websocket)


# --- MODIFIED: XAI SDK AI FUNCTIONS ---

def configure_xai_client():
    """Configures and returns the XAI SDK client."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY environment variable not set.")
        return None
    # Per the docs, a longer timeout is good for reasoning models
    return Client(api_key=api_key, timeout=3600)

def format_tweet(tweet: dict) -> str:
    user = tweet.get('user', {}).get('username', 'unknown')
    followers = tweet.get('user', {}).get('followers', 0)
    text = tweet.get('text', '')
    return f"@{user} ({followers} followers): \"{text}\""

def format_buffer_for_ai(current_buffer: list, historical_context: deque, active_markets: dict) -> str:
    history_lines = ["--- RECENT HISTORY (Last 100 Processed Tweets) ---"] + [format_tweet(t) for t in historical_context]
    new_lines = ["\n--- NEW TWEETS (To Analyze) ---"] + [format_tweet(t) for t in current_buffer]
    active_market_names = [name for name, data in active_markets.items() if data["status"] == "OPEN"]
    market_lines = ["\n--- ACTIVE MARKETS (To Check for Resolution) ---"] + (active_market_names if active_market_names else ["None"])
    return "\n".join(history_lines + market_lines + new_lines)

def clean_json_response(text: str) -> str:
    """
    Cleans the AI's response to extract only the JSON, in case it adds
    surrounding text like "Here is the JSON you requested:".
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

async def get_ai_decision(xai_client: Client, current_buffer: list, historical_context: deque, active_markets: dict) -> dict:
    """
    Sends the tweet buffer and context to the XAI API using the stateful chat pattern.
    """
    if not xai_client:
        return {"actions": []}

    buffer_text = format_buffer_for_ai(current_buffer, historical_context, active_markets)
    print(f"\n--- Sending buffer of {len(current_buffer)} (with {len(historical_context)} history) to XAI ---")

    try:
       
        
        # 1. Create a new chat session with the system prompt
        chat = xai_client.chat.create(
            model=XAI_MODEL,
            messages=[system(AI_SYSTEM_PROMPT)],
        )
        
        # 2. Append the user's data payload
        chat.append(user(buffer_text))

        # 3. Get the response from the model
        response = chat.sample()

        # 4. Clean and parse the response
        response_text = response.content
        cleaned_json = clean_json_response(response_text)
        
        if not cleaned_json:
            print(f"  XAI Error: No JSON found in response: {response_text}")
            return {"actions": []}

        decision = json.loads(cleaned_json)
        return decision

    except Exception as e:
        print(f"  XAI API Error: {e}")
        return {"actions": []}


# --- MAIN AGENT LOOP (Connects to Tweet Stream) ---

async def listen_to_tweet_stream(xai_client: Client):
    tweet_buffer = []
    print(f"Prophet Agent connecting to Tweet Stream at {TWEET_SERVER_URI}...")

    while True:
        try:
            async with websockets.connect(TWEET_SERVER_URI) as tweet_ws:
                print("Successfully connected to Tweet Stream! Listening...")

                async for message in tweet_ws:
                    try:
                        tweet = json.loads(message)
                        tweet_buffer.append(tweet)
                        user = tweet.get('user', {}).get('username', 'unknown')
                        print(f"  Buffer={len(tweet_buffer)}/{BUFFER_SIZE} | Read from @{user}")

                        if len(tweet_buffer) >= BUFFER_SIZE:
                            ai_decision = await get_ai_decision(
                                xai_client, tweet_buffer, HISTORICAL_CONTEXT, ACTIVE_MARKETS
                            )
                            HISTORICAL_CONTEXT.extend(tweet_buffer)
                            tweet_buffer = []

                            actions = ai_decision.get("actions", [])
                            if not actions:
                                print("  AI Decision: No new actions.")

                            for action in actions:
                                market_name = action.get("market_name")
                                if not market_name: continue

                                if action.get("action") == "CREATE":
                                    if market_name in ACTIVE_MARKETS:
                                        print(f"  AI Decision: Market '{market_name}' already exists. Ignoring.")
                                    else:
                                        print("\n" + "="*50)
                                        print(f">>>> [AGENT ACTION] CREATING NEW MARKET <<<<")
                                        print(f"  Market Name: {market_name}")
                                        print(f"  Initial Probability (YES): {action.get('probability')}")
                                        print(f"  AI Reason: {action.get('reason')}")
                                        print("="*50 + "\n")
                                        ACTIVE_MARKETS[market_name] = {"status": "OPEN", "probability": action.get('probability')}
                                        await broadcast_action(json.dumps(action))

                                elif action.get("action") == "RESOLVE":
                                    if market_name in ACTIVE_MARKETS and ACTIVE_MARKETS[market_name]["status"] == "OPEN":
                                        print("\n" + "="*50)
                                        print(f">>>> [AGENT ACTION] RESOLVING MARKET <<<<")
                                        print(f"  Market Name: {market_name}")
                                        print(f"  Outcome: {action.get('outcome')}")
                                        print(f"  AI Reason: {action.get('reason')}")
                                        print("="*50 + "\n")
                                        ACTIVE_MARKETS[market_name]["status"] = "RESOLVED"
                                        await broadcast_action(json.dumps(action))
                                    else:
                                        print(f"  AI Decision: Market '{market_name}' already resolved or unknown. Ignoring.")
                    except Exception as e:
                        print(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Tweet Stream connection lost ({e.code}). Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"An error occurred in listener: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

async def main():
    xai_client = configure_xai_client()
    if not xai_client:
        return

    print("--- Prophet Agent v4 (Brain + Action Server) ---")
    asyncio.create_task(listen_to_tweet_stream(xai_client))
    print(f"Action Server listening on ws://{ACTION_SERVER_HOST}:{ACTION_SERVER_PORT}")
    async with websockets.serve(action_client_handler, ACTION_SERVER_HOST, ACTION_SERVER_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())