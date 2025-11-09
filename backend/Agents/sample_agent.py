import asyncio
import websockets
import json
import os
import re
from collections import deque, defaultdict
from dotenv import load_dotenv

# Gemini AI Imports
import google.generativeai as genai

# env variables
load_dotenv() 

# --- CONFIGURATION ---
AGENT_ID = "trader_3_Stateful" # Changed ID to avoid collisions
EXCHANGE_URL = "ws://localhost:8767"    
TWEET_SERVER_URI = "ws://localhost:8765" 
BUFFER_SIZE = 5
HISTORY_SIZE = 50
# TRADE_QUANTITY = 10.0 # --- REMOVED: AI will decide quantity ---

# --- MODIFICATION: Entirely new, smarter prompt ---
AI_SYSTEM_PROMPT_TRADER = """
You are an autonomous prediction market trader operating in a live market
with other AI traders. Your goal is to make profitable trades — but also
to stay active in the market, providing liquidity and reacting to price
movements even if it’s not perfectly optimal.

You will receive:
1. "NEW TWEETS": Real-time event signals and context.
2. "ACTIVE MARKETS": Current order book prices (best bids and asks).
3. "CURRENT_PORTFOLIO": Your current holdings (USD and tokens).

Your objectives:
1. Estimate your own "true_probability" for each event.
2. Compare that probability to current market prices.
3. Decide whether to BUY, SELL, or HOLD based on relative value and momentum.
4. Optionally post both buy and sell orders to provide liquidity.

DECISION GUIDELINES:
- **BUY Opportunity:** If your `true_probability` > `best_ask` (market underpriced).
- **SELL Opportunity:** If your `true_probability` < `best_bid` (market overpriced).
- **Liquidity Behavior:** Occasionally place both BUY and SELL limit orders near the market price, even if your belief is neutral — this helps stimulate trading between agents.
- **Reactive Behavior:** If another trader sells at a low price you think is too cheap, BUY some.
- **Profit Taking:** If a position gains value, SELL part of it to lock in profits.
- **Contrarian Moves:** Occasionally sell into buying pressure or buy into selling pressure if prices move too far.

RISK MANAGEMENT:
- Avoid using all your capital at once.
- Use partial position sizing (small trade_size_pct per action).
- Maintain cash for future opportunities.

RESPONSE RULES:
- Respond ONLY with one minified JSON object.
- If no clear opportunities, you can still place small liquidity-providing orders near the current price.
- If absolutely no action, respond with: {"actions": []}

--- JSON Response Format ---
{
  "actions": [
    {
      "action": "PLACE_ORDER",
      "side": "BUY",
      "market_name": "ProjectChimeraLaunch_YES",
      "trade_size_pct": 0.20,
      "reason": "Tweets suggest high chance; price 0.55, my belief 0.7. Buying modestly."
    },
    {
      "action": "PLACE_ORDER",
      "side": "SELL",
      "market_name": "ProjectChimeraLaunch_YES",
      "trade_size_pct": 0.15,
      "reason": "Price spiked above my belief (0.75 vs 0.65). Taking profit."
    }
  ]
}

- "trade_size_pct" is a float from 0.0 to 1.0.
  - For "BUY": Percentage of available USD to spend.
  - For "SELL": Percentage of tokens for that asset to sell.
"""

# --- Global State for the Agent ---
AGENT_PORTFOLIO = defaultdict(float)
ACTIVE_MARKETS_STATE = {}
HISTORICAL_CONTEXT = deque(maxlen=HISTORY_SIZE)


# --- GEMINI AI FUNCTIONS (configure_gemini, format_tweet, clean_json_response are unchanged) ---

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(f"[{AGENT_ID}] Error: GEMINI_API_KEY not set.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        'gemini-2.5-flash-preview-09-2025',
        system_instruction=AI_SYSTEM_PROMPT_TRADER
    )

def format_tweet(tweet: dict) -> str:
    user = tweet.get('user', {}).get('username', 'unknown')
    followers = tweet.get('user', {}).get('followers', 0)
    text = tweet.get('text', '')
    return f"@{user} ({followers} followers): \"{text}\""

# --- MODIFICATION: Now accepts and formats the portfolio ---
def format_prompt_for_ai(current_buffer: list, agent_portfolio: dict) -> str:
    """Combines portfolio, market state, and tweets into a prompt for the AI."""
    
    # 1. Format portfolio
    portfolio_lines = ["\n--- CURRENT_PORTFOLIO ---"]
    if not agent_portfolio:
        portfolio_lines.append("USD: 10000.0 (Initial)") # Show default
    else:
        for asset, quantity in agent_portfolio.items():
            portfolio_lines.append(f"{asset}: {quantity:.2f}")

    # 2. Format active markets
    market_lines = ["\n--- ACTIVE MARKETS (Current Prices) ---"]
    if not ACTIVE_MARKETS_STATE:
        market_lines.append("None")
    else:
        for market, state in ACTIVE_MARKETS_STATE.items():
            market_lines.append(
                f"- {market}: "
                f"best_ask (buy at): {state.get('best_ask', 'N/A')}, "
                f"best_bid (sell at): {state.get('best_bid', 'N/A')}"
            )
            
    # 3. Format tweets
    history_lines = [
        "\n--- RECENT HISTORY (Last 50 Processed Tweets) ---"
    ] + [format_tweet(tweet) for tweet in HISTORICAL_CONTEXT]
    
    new_lines = [
        "\n--- NEW TWEETS (To Analyze) ---"
    ] + [format_tweet(tweet) for tweet in current_buffer]
    
    # Combine all parts
    return "\n".join(portfolio_lines + market_lines + history_lines + new_lines)

def clean_json_response(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match: return match.group(0)
    return ""

# --- MODIFICATION: Now passes portfolio to the formatting function ---
async def get_ai_decision(ai_model, current_buffer: list, agent_portfolio: dict) -> dict:
    """Sends portfolio, market state, and tweets to Gemini."""
    if not ai_model: return {"actions": []}

    # Pass the agent's portfolio to the prompt formatter
    prompt_text = format_prompt_for_ai(current_buffer, agent_portfolio)
    
    print(f"\n[{AGENT_ID}] --- Sending prompt to AI for analysis ---")
    # print(prompt_text) # Uncomment for debugging the prompt
    
    try:
        response = await ai_model.generate_content_async(prompt_text)
        json_text = clean_json_response(response.text)
        if not json_text:
            print(f"   [{AGENT_ID}] AI Error: No JSON found in response: {response.text}")
            return {"actions": []}
        
        decision = json.loads(json_text)
        return decision
    
    except Exception as e:
        print(f"   [{AGENT_ID}] AI Error: {e}")
        return {"actions": []}


# --- AGENT'S CORE LOGIC ---

# --- MODIFICATION: listen_to_exchange is now much simpler ---
# It only updates our local state. The "decide" logic is moved
# to listen_to_tweets.
async def listen_to_exchange(ws_exchange):
    """
    Listens FOR messages FROM the exchange.
    This is our "Observe" loop for market state.
    """
    print(f"[{AGENT_ID}] Exchange listener started.")
    try:
        async for message in ws_exchange:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == 'registered':
                print(f"[{AGENT_ID}] Successfully registered with exchange.")
            
            elif msg_type == 'account_update':
                # Update our portfolio
                print(f"[{AGENT_ID}] Portfolio Update: {data['balances']}")
                AGENT_PORTFOLIO.clear()
                AGENT_PORTFOLIO.update(data['balances'])

            elif msg_type == 'new_market':
                print(f"[{AGENT_ID}] New Market Seen: {data['market_name']}")
                ACTIVE_MARKETS_STATE[data['market_yes']] = {"status": "OPEN"}
                ACTIVE_MARKETS_STATE[data['market_no']] = {"status": "OPEN"}

            elif msg_type == 'market_resolved':
                print(f"[{AGENT_ID}] Market Resolved: {data['market_name']}")
                market_yes = f"{data['market_name']}_YES"
                market_no = f"{data['market_name']}_NO"
                if market_yes in ACTIVE_MARKETS_STATE: del ACTIVE_MARKETS_STATE[market_yes]
                if market_no in ACTIVE_MARKETS_STATE: del ACTIVE_MARKETS_STATE[market_no]

            elif msg_type == 'order_book_update':
                market = data['market']
                if market in ACTIVE_MARKETS_STATE:
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    
                    if bids:
                        ACTIVE_MARKETS_STATE[market]['best_bid'] = bids[0][0]
                    else:
                        ACTIVE_MARKETS_STATE[market]['best_bid'] = None
                        
                    if asks:
                        ACTIVE_MARKETS_STATE[market]['best_ask'] = asks[0][0]
                    else:
                        ACTIVE_MARKETS_STATE[market]['best_ask'] = None

            elif msg_type == 'trade_executed':
                if data['buyer'] == AGENT_ID:
                    print(f"*** [{AGENT_ID}] => Our BUY order was filled! Bought {data['quantity']} {data['market']} ***")
                if data['seller'] == AGENT_ID:
                    print(f"*** [{AGENT_ID}] => Our SELL order was filled! Sold {data['quantity']} {data['market']} ***")

            elif msg_type == 'error':
                print(f"[{AGENT_ID}] Exchange Error: {data['message']}")

    except websockets.exceptions.ConnectionClosed:
        print(f"[{AGENT_ID}] Connection to exchange lost.")
    except Exception as e:
        print(f"[{AGENT_ID}] Error in exchange listener: {e}")

async def listen_to_tweets(ws_exchange, ai_model):
    """
    Listens FOR messages FROM the tweet stream.
    This is our "Orient, Decide, Act" loop.
    """
    print(f"[{AGENT_ID}] Tweet listener connecting to {TWEET_SERVER_URI}...")
    tweet_buffer = []
    
    while True:
        try:
            async with websockets.connect(TWEET_SERVER_URI) as tweet_ws:
                print(f"[{AGENT_ID}] Tweet listener connected!")
                
                async for message in tweet_ws:
                    tweet = json.loads(message)
                    tweet_buffer.append(tweet)
                    
                    if len(tweet_buffer) >= BUFFER_SIZE:
                        # 1. Get AI decision
                        ai_decision = await get_ai_decision(
                            ai_model, 
                            tweet_buffer, 
                            AGENT_PORTFOLIO
                        )
                        
                        HISTORICAL_CONTEXT.extend(tweet_buffer)
                        tweet_buffer = []
                        
                        # 2. Act on decision
                        actions = ai_decision.get("actions", [])
                        if not actions:
                            print(f"   [{AGENT_ID}] AI: No actions to take.")
                        
                        else:
                            # --- MODIFICATION: Earmarking Logic ---
                            # Get a snapshot of our funds *before* this batch
                            available_usd_for_batch = AGENT_PORTFOLIO.get("USD", 0)
                            # Create a temporary copy of token balances for this batch
                            available_tokens_for_batch = AGENT_PORTFOLIO.copy()
                            
                            print(f"   [{AGENT_ID}] Starting batch with ${available_usd_for_batch:.2f} available.")
                            
                            for action in actions:
                                if action.get("action") == "PLACE_ORDER":
                                    
                                    # Pass the *remaining* funds to the processor
                                    usd_spent, token_market, tokens_sold = await process_ai_trade(
                                        ws_exchange, 
                                        action,
                                        available_usd_for_batch,
                                        available_tokens_for_batch
                                    )
                                    
                                    # Earmark the funds for the *next* action in this batch
                                    available_usd_for_batch -= usd_spent
                                    if token_market:
                                        available_tokens_for_batch[token_market] -= tokens_sold

        except websockets.exceptions.ConnectionClosed:
            print(f"[{AGENT_ID}] Tweet stream connection lost. Reconnecting in 5s...")
        except Exception as e:
            print(f"[{AGENT_ID}] Error in tweet listener: {e}. Retrying in 5s...")
        
        await asyncio.sleep(5)

# --- REBUILT: _process_ai_trade for AMM (with Percentage Caps) ---
async def _process_ai_trade(self, action: dict, available_usd: float, available_tokens: dict) -> (float, str, float):
    """
    Calculates quantity from percentage and sends AMM trade order,
    but caps the trade to a maximum size to prevent price collapse.
    Returns (usd_spent, token_market, tokens_sold) to earmark funds.
    """
    
    # --- NEW: Define maximums for a single trade ---
    # This is the "sanity check" to prevent market collapse.
    MAX_USD_PER_TRADE = 500.0   # Cap any single BUY at $500
    MAX_TOKENS_PER_TRADE = 500.0 # Cap any single SELL at 500 tokens
    
    # --- 1. Get All AI Parameters ---
    side = action.get("side")
    market_asset = action.get("market_asset") # e.g., "Market_YES"
    trade_size_pct = action.get("trade_size_pct", 0.1) # Default to 10%
    
    print(f"--- [{self.agent_id}] AI ACTION: {side} {market_asset} ({trade_size_pct*100}%) ---")
    print(f"     Reason: {action.get('reason')}")

    current_price = self.active_markets.get(market_asset, {}).get('price')
    if current_price is None:
        print(f"     ACTION FAILED: No price data for {market_asset}.")
        return (0, None, 0)

    # --- 2. Process BUY Logic (with new cap) ---
    if side == "BUY":
        # 1. Calculate desired spend based on AI confidence
        desired_usd_to_spend = available_usd * trade_size_pct
        
        # 2. Apply the hard cap
        usd_to_spend = min(desired_usd_to_spend, MAX_USD_PER_TRADE)
        
        if usd_to_spend < 1.0: # Don't trade if less than $1
            print(f"     ACTION FAILED: Trade size too small (${usd_to_spend:.2f}).")
            return (0, None, 0)
            
        if current_price <= 0:
            print(f"     ACTION FAILED: Market price is {current_price}. Cannot calculate quantity.")
            return (0, None, 0)

        # 3. Calculate quantity based on the *capped* amount
        quantity_to_buy = round(usd_to_spend / current_price, 3) 

        if quantity_to_buy < 0.001:
            print(f"     ACTION FAILED: Calculated quantity too small (Qty: {quantity_to_buy}).")
            return (0, None, 0)
            
        await self.ws_exchange.send(json.dumps({
            "action": "trade",
            "market": market_asset,
            "quantity": quantity_to_buy  # Positive for BUY
        }))
        print(f"     > Sent BUY for {quantity_to_buy} {market_asset}. (Capped spend at ${usd_to_spend:.2f})")
        return (usd_to_spend, None, 0) # Earmark the capped amount

    # --- 3. Process SELL Logic (with new cap) ---
    elif side == "SELL":
        available_tokens_for_market = available_tokens.get(market_asset, 0)
        
        # 1. Calculate desired sell amount based on AI confidence
        desired_tokens_to_sell = available_tokens_for_market * trade_size_pct
        
        # 2. Apply the hard cap
        quantity_to_sell = min(desired_tokens_to_sell, MAX_TOKENS_PER_TRADE)
        
        if quantity_to_sell < 0.001:
            print(f"     ACTION FAILED: No tokens to sell ({available_tokens_for_market}) or trade size too small.")
            return (0, None, 0)
        
        await self.ws_exchange.send(json.dumps({
            "action": "trade",
            "market": market_asset,
            "quantity": -quantity_to_sell # Negative for SELL
        }))
        print(f"     > Sent SELL for {quantity_to_sell} {market_asset}. (Capped at {MAX_TOKENS_PER_TRADE} tokens)")
        return (0, market_asset, quantity_to_sell) # Earmark the capped tokens

    return (0, None, 0)

# --- Main function is unchanged ---
async def main():
    ai_model = configure_gemini()
    if not ai_model: return

    print(f"--- Intelligent Trader Agent [{AGENT_ID}] v2.0 (Stateful) ---")
    
    while True:
        try:
            async with websockets.connect(EXCHANGE_URL) as ws_exchange:
                print(f"[{AGENT_ID}] Connected to exchange {EXCHANGE_URL}")
                
                await ws_exchange.send(json.dumps({
                    "action": "register",
                    "agent_id": AGENT_ID
                }))
                
                exchange_listener_task = asyncio.create_task(
                    listen_to_exchange(ws_exchange)
                )
                
                tweet_listener_task = asyncio.create_task(
                    listen_to_tweets(ws_exchange, ai_model)
                )
                
                await asyncio.gather(exchange_listener_task, tweet_listener_task)

        except websockets.exceptions.ConnectionClosed:
            print(f"[{AGENT_ID}] Main connection to exchange failed. Retrying in 5s...")
        except Exception as e:
            print(f"[{AGENT_ID}] Main loop error: {e}. Retrying in 5s...")
        
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())