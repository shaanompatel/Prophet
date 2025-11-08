import asyncio
import websockets
import json
import time
import os
import re
from collections import deque
from dotenv import load_dotenv

# Gemini AI Imports
import google.generativeai as genai


# env variables
load_dotenv()  # reads from .env file

# --- CONFIGURATION ---
TWEET_SERVER_URI = "ws://localhost:8765"  # Listen to tweets from here
ACTION_SERVER_HOST = "localhost"         # Host our action server here
ACTION_SERVER_PORT = 8766                # on this port
BUFFER_SIZE = 10
HISTORY_SIZE = 100

# --- NEW, UPGRADED AI PROMPT ---
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

# --- Global list of connected clients (e.g., your Solana code) ---
CONNECTED_ACTION_CLIENTS = set()


# --- WEBSOCKET SERVER LOGIC (for broadcasting actions) ---

async def broadcast_action(action_json: str):
    """Broadcasts an action to all connected clients."""
    if CONNECTED_ACTION_CLIENTS:
        print(f"[ACTION_BROADCAST] Sending to {len(CONNECTED_ACTION_CLIENTS)} clients: {action_json}")
        tasks = [
            asyncio.create_task(client.send(action_json))
            for client in CONNECTED_ACTION_CLIENTS
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

async def action_client_handler(websocket):
    """Handles a new client connecting to *this* server."""
    print(f"[ACTION_SERVER] Client connected: {websocket.remote_address}")
    CONNECTED_ACTION_CLIENTS.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        print(f"[ACTION_SERVER] Client disconnected: {websocket.remote_address}")
        CONNECTED_ACTION_CLIENTS.remove(websocket)


# --- GEMINI AI FUNCTION (Unchanged) ---

def configure_gemini():
    """Configures the Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        'gemini-2.5-flash-preview-09-2025',
        system_instruction=AI_SYSTEM_PROMPT
    )

def format_tweet(tweet: dict) -> str:
    """Converts a single tweet dict into a simple text line."""
    user = tweet.get('user', {}).get('username', 'unknown')
    followers = tweet.get('user', {}).get('followers', 0)
    text = tweet.get('text', '')
    return f"@{user} ({followers} followers): \"{text}\""

def format_buffer_for_ai(current_buffer: list, historical_context: deque, active_markets: dict) -> str:
    """Converts all our data into a single text block for the AI."""
    history_lines = [
        "--- RECENT HISTORY (Last 100 Processed Tweets) ---"
    ] + [format_tweet(tweet) for tweet in historical_context]
    
    new_lines = [
        "\n--- NEW TWEETS (To Analyze) ---"
    ] + [format_tweet(tweet) for tweet in current_buffer]
    
    active_market_names = [name for name, data in active_markets.items() if data["status"] == "OPEN"]
    market_lines = [
        "\n--- ACTIVE MARKETS (To Check for Resolution) ---"
    ] + (active_market_names if active_market_names else ["None"])
    
    return "\n".join(history_lines + market_lines + new_lines)

def clean_json_response(text: str) -> str:
    """Cleans the AI's response to extract only the JSON."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

async def get_ai_decision(ai_model, current_buffer: list, historical_context: deque, active_markets: dict) -> dict:
    """
    Sends the tweet buffer, history, and active markets to Gemini.
    """
    if not ai_model:
        return {"actions": []}

    buffer_text = format_buffer_for_ai(current_buffer, historical_context, active_markets)
    print(f"\n--- Sending buffer of {len(current_buffer)} (with {len(historical_context)} history) to AI ---")
    
    try:
        retries = 3
        delay = 1
        for i in range(retries):
            try:
                response = await ai_model.generate_content_async(buffer_text)
                json_text = clean_json_response(response.text)
                
                if not json_text:
                    print(f"  AI Error: No JSON found in response: {response.text}")
                    return {"actions": []}
                    
                decision = json.loads(json_text)
                return decision
            
            except Exception as e:
                print(f"  AI API call failed (attempt {i+1}/{retries}): {e}")
                if i < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise e
        
    except Exception as e:
        print(f"  AI Error after retries: {e}")
        return {"actions": []}


# --- MAIN AGENT LOOP (Connects to Tweet Stream) ---

async def listen_to_tweet_stream(ai_model):
    """
    Connects to the TWEET stream and runs the main agent logic.
    """
    tweet_buffer = []
    print(f"Prophet Agent connecting to Tweet Stream at {TWEET_SERVER_URI}...")
    
    while True:
        try:
            # Connect to the *tweet* server
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
                                ai_model, 
                                tweet_buffer, 
                                HISTORICAL_CONTEXT,
                                ACTIVE_MARKETS
                            )
                            
                            HISTORICAL_CONTEXT.extend(tweet_buffer)
                            tweet_buffer = [] 
                            
                            actions = ai_decision.get("actions", [])
                            if not actions:
                                print("  AI Decision: No new actions.")
                            
                            for action in actions:
                                market_name = action.get("market_name")
                                if not market_name:
                                    continue

                                # --- PROCESS "CREATE" ACTION ---
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
                                        
                                        ACTIVE_MARKETS[market_name] = {
                                            "status": "OPEN",
                                            "probability": action.get('probability')
                                        }
                                        # *** MODIFIED: Broadcast to our *own* clients ***
                                        await broadcast_action(json.dumps(action))

                                # --- PROCESS "RESOLVE" ACTION ---
                                elif action.get("action") == "RESOLVE":
                                    if market_name in ACTIVE_MARKETS and ACTIVE_MARKETS[market_name]["status"] == "OPEN":
                                        print("\n" + "="*50)
                                        print(f">>>> [AGENT ACTION] RESOLVING MARKET <<<<")
                                        print(f"  Market Name: {market_name}")
                                        print(f"  Outcome: {action.get('outcome')}")
                                        print(f"  AI Reason: {action.get('reason')}")
                                        print("="*50 + "\n")
                                        
                                        ACTIVE_MARKETS[market_name]["status"] = "RESOLVED"
                                        # *** MODIFIED: Broadcast to our *own* clients ***
                                        await broadcast_action(json.dumps(action))
                                    else:
                                        print(f"  AI Decision: Market '{market_name}' already resolved or unknown. Ignoring.")

                    except json.JSONDecodeError:
                        print(f"Error: Received invalid JSON from stream: {message}")
                    except Exception as e:
                        print(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Tweet Stream connection lost ({e.code}). Reconnecting in 5 seconds...")
        except Exception as e:
            print(f"An error occurred in listener: {e}. Retrying in 5 seconds...")
        
        await asyncio.sleep(5)

async def main():
    ai_model = configure_gemini()
    if not ai_model:
        return

    print("--- Prophet Agent v4 (Brain + Action Server) ---")
    
    # Start the tweet listener as a background task
    asyncio.create_task(listen_to_tweet_stream(ai_model))

    # Start the Action Server to accept clients (e.g., Solana code)
    print(f"Action Server listening on ws://{ACTION_SERVER_HOST}:{ACTION_SERVER_PORT}")
    async with websockets.serve(
        action_client_handler, 
        ACTION_SERVER_HOST, 
        ACTION_SERVER_PORT
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())