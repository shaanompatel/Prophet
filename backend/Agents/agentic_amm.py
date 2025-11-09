import asyncio
import websockets
import json
import os
import re
from collections import deque, defaultdict
from dotenv import load_dotenv

# Gemini AI Imports
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()

# --- NEW AI PROMPT ---
AI_SYSTEM_PROMPT_TRADER = """
You are an autonomous prediction market trader. Your job is to trade
against an Automated Market Maker (AMM).

You will receive:
1. "NEW TWEETS": Real-time information and signals.
2. "ACTIVE MARKETS": The current AMM price for each asset.
3. "CURRENT_PORTFOLIO": Your USD and token holdings.

Your goals:
1. Infer a "true_probability" for each market from the news.
2. Compare your belief to the "current_price" from the AMM.
3. Decide whether to BUY (if true_prob > current_price) or SELL (if true_prob < current_price).
4. Manage your inventory — don’t over-accumulate tokens or cash.

### DECISION RULES

- **BUY Opportunity:** If `true_probability` > `current_price`.
- **SELL Opportunity:** If `true_probability` < `current_price`.
- **Trade Sizing:** Use `trade_size_pct` to determine how much of your
  available USD (for BUYS) or tokens (for SELLS) to use.
- **Risk Management:** Avoid going all-in. Your trade size will be
  a percentage of your available assets.

### RESPONSE FORMAT

Respond **only** with a compact JSON object.
Use "BUY" or "SELL" for the `side`, and specify the *full asset name*
(e.g., "MarketName_YES") in `market_asset`.

```json
{"actions": [
  {
    "action": "TRADE",
    "side": "BUY",
    "market_asset": "Election2028_BidenWins_YES",
    "trade_size_pct": 0.25,
    "true_probability": 0.65,
    "reason": "My true prob 0.65 > amm_price 0.60. Buying with 25% of USD."
  },
  {
    "action": "TRADE",
    "side": "SELL",
    "market_asset": "ProjectChimeraLaunch_YES",
    "trade_size_pct": 0.40,
    "true_probability": 0.45,
    "reason": "My true prob 0.45 < amm_price 0.50. Selling 40% of my holdings."
  }
]}
"""

class AgenticTrader:
    
    def __init__(self, 
                 agent_id: str, 
                 gemini_api_key: str, 
                 exchange_url: str = "ws://localhost:8767", 
                 tweet_server_uri: str = "ws://localhost:8765", 
                 buffer_size: int = 5, 
                 history_size: int = 50,
                 system_prompt: str = AI_SYSTEM_PROMPT_TRADER):

        self.agent_id = agent_id
        self.gemini_api_key = gemini_api_key
        self.exchange_url = exchange_url
        self.tweet_server_uri = tweet_server_uri
        self.buffer_size = buffer_size
        self.system_prompt = system_prompt
        
        self.ai_model = None
        self.ws_exchange = None 
        
        self.portfolio = defaultdict(float)
        self.active_markets = {} # Stores current prices { "Market_YES": {"price": 0.6}, ... }
        self.historical_context = deque(maxlen=history_size)
        
        print(f"--- Trader [{self.agent_id}] Initialized (AMM Mode) ---")

    def _configure_gemini(self):
        """Sets up the Gemini AI model for this instance."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.ai_model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-09-2025', # Using latest flash
                system_instruction=self.system_prompt
            )
            print(f"[{self.agent_id}] Gemini AI Model configured.")
        except Exception as e:
            print(f"[{self.agent_id}] Error: Failed to configure Gemini: {e}")
            self.ai_model = None

    @staticmethod
    def _static_format_tweet(tweet: dict) -> str:
        """Static helper to format a tweet dictionary into a string."""
        user = tweet.get('user', {}).get('username', 'unknown')
        followers = tweet.get('user', {}).get('followers', 0)
        text = tweet.get('text', '')
        return f"@{user} ({followers} followers): \"{text}\""

    @staticmethod
    def _static_clean_json_response(text: str) -> str:
        """Static helper to extract JSON from a raw AI response string."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return match.group(0)
        return ""

    def _format_prompt_for_ai(self, current_buffer: list) -> str:
        """Combines portfolio, market state, and tweets into a prompt."""
        
        # 1. Format portfolio
        portfolio_lines = ["\n--- CURRENT_PORTFOLIO ---"]
        if not self.portfolio:
            portfolio_lines.append("USD: 10000.0 (Initial)") # Show default
        else:
            for asset, quantity in self.portfolio.items():
                portfolio_lines.append(f"{asset}: {quantity:.2f}")

        # 2. Format active markets (now showing price)
        market_lines = ["\n--- ACTIVE MARKETS (Current AMM Prices) ---"]
        if not self.active_markets:
            market_lines.append("None")
        else:
            for asset, state in self.active_markets.items():
                market_lines.append(
                    f"- {asset}: "
                    f"current_price: {state.get('price', 'N/A')}"
                )
        
        # 3. Format tweets
        history_lines = [
            f"\n--- RECENT HISTORY (Last {len(self.historical_context)} Processed Tweets) ---"
        ] + [self._static_format_tweet(tweet) for tweet in self.historical_context]
        
        new_lines = [
            "\n--- NEW TWEETS (To Analyze) ---"
        ] + [self._static_format_tweet(tweet) for tweet in current_buffer]
        
        # Combine all parts
        return "\n".join(portfolio_lines + market_lines + history_lines + new_lines)

    async def _get_ai_decision(self, current_buffer: list) -> dict:
        """Sends portfolio, market state, and tweets to Gemini."""
        if not self.ai_model: return {"actions": []}

        prompt_text = self._format_prompt_for_ai(current_buffer)
        
        print(f"\n[{self.agent_id}] --- Sending prompt to AI for analysis ---")
        
        try:
            response = await self.ai_model.generate_content_async(prompt_text)
            json_text = self._static_clean_json_response(response.text)
            
            if not json_text:
                print(f"   [{self.agent_id}] AI Error: No JSON found in response: {response.text}")
                return {"actions": []}
            
            decision = json.loads(json_text)
            return decision
        
        except Exception as e:
            print(f"   [{self.agent_id}] AI Error: {e}")
            return {"actions": []}

    async def _listen_to_exchange(self):
        """Listens FOR messages FROM the exchange."""
        print(f"[{self.agent_id}] Exchange listener started.")
        try:
            async for message in self.ws_exchange:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == 'registered':
                    print(f"[{self.agent_id}] Successfully registered with exchange.")
                
                elif msg_type == 'account_update':
                    print(f"[{self.agent_id}] Portfolio Update: {data['balances']}")
                    self.portfolio.clear()
                    self.portfolio.update(data['balances'])

                elif msg_type == 'new_market':
                    print(f"[{self.agent_id}] New Market Seen: {data['market_name']}")
                    # We don't know the price yet, wait for price_update
                    self.active_markets[data['market_yes']] = {"status": "OPEN"}
                    self.active_markets[data['market_no']] = {"status": "OPEN"}

                elif msg_type == 'market_resolved':
                    print(f"[{self.agent_id}] Market Resolved: {data['market_name']}")
                    market_yes = f"{data['market_name']}_YES"
                    market_no = f"{data['market_name']}_NO"
                    if market_yes in self.active_markets: del self.active_markets[market_yes]
                    if market_no in self.active_markets: del self.active_markets[market_no]

                # --- UPDATED: Listen for AMM price updates ---
                elif msg_type == 'price_update':
                    market_name = data['market']
                    market_yes = f"{market_name}_YES"
                    market_no = f"{market_name}_NO"
                    
                    if market_yes in self.active_markets:
                        self.active_markets[market_yes]['price'] = data['price_yes']
                    if market_no in self.active_markets:
                        self.active_markets[market_no]['price'] = data['price_no']

                elif msg_type == 'trade_executed':
                    if data['agent'] == self.agent_id:
                        if data['quantity'] > 0:
                            print(f"*** [{self.agent_id}] => Our BUY was executed! ({data['quantity']} {data['market']} for ${data['cost']:.2f}) ***")
                        else:
                            print(f"*** [{self.agent_id}] => Our SELL was executed! ({abs(data['quantity'])} {data['market']} for ${-data['cost']:.2f}) ***")

                elif msg_type == 'error':
                    print(f"[{self.agent_id}] Exchange Error: {data['message']}")

        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.agent_id}] Connection to exchange lost.")
        except Exception as e:
            print(f"[{self.agent_id}] Error in exchange listener: {e}")

    async def _listen_to_tweets(self):
        """Listens FOR messages FROM the tweet stream."""
        print(f"[{self.agent_id}] Tweet listener connecting to {self.tweet_server_uri}...")
        tweet_buffer = []
        
        while True:
            try:
                async with websockets.connect(self.tweet_server_uri) as tweet_ws:
                    print(f"[{self.agent_id}] Tweet listener connected!")
                    
                    async for message in tweet_ws:
                        tweet = json.loads(message)
                        tweet_buffer.append(tweet)
                        
                        if len(tweet_buffer) >= self.buffer_size:
                            # 1. Get AI decision
                            ai_decision = await self._get_ai_decision(
                                tweet_buffer
                            )
                            
                            self.historical_context.extend(tweet_buffer)
                            tweet_buffer = []
                            
                            # 2. Act on decision
                            actions = ai_decision.get("actions", [])
                            if not actions:
                                print(f"   [{self.agent_id}] AI: No actions to take.")
                            
                            else:
                                available_usd_for_batch = self.portfolio.get("USD", 0)
                                available_tokens_for_batch = self.portfolio.copy()
                                
                                print(f"   [{self.agent_id}] Starting batch with ${available_usd_for_batch:.2f} available.")
                                
                                for action in actions:
                                    if action.get("action") == "TRADE":
                                        
                                        usd_spent, token_market, tokens_sold = await self._process_ai_trade(
                                            action,
                                            available_usd_for_batch,
                                            available_tokens_for_batch
                                        )
                                        
                                        # Earmark the funds for the *next* action
                                        available_usd_for_batch -= usd_spent
                                        if token_market:
                                            available_tokens_for_batch[token_market] -= tokens_sold

            except websockets.exceptions.ConnectionClosed:
                print(f"[{self.agent_id}] Tweet stream connection lost. Reconnecting in 5s...")
            except Exception as e:
                print(f"[{self.agent_id}] Error in tweet listener: {e}. Retrying in 5s...")
            
            await asyncio.sleep(5)

    # --- REBUILT: _process_ai_trade for AMM ---
    async def _process_ai_trade(self, action: dict, available_usd: float, available_tokens: dict) -> (float, str, float):
        """
        Calculates quantity from percentage and sends AMM trade order,
        but caps the trade to a maximum size to prevent price collapse.
        Returns (usd_spent, token_market, tokens_sold) to earmark funds.
        """
        
        # --- NEW: Define maximums for a single trade ---
        # This is the "sanity check" to prevent market collapse.
        # You can tune these values.
        MAX_USD_PER_TRADE = 50.0   # Cap any single BUY at $500
        MAX_TOKENS_PER_TRADE = 50.0 # Cap any single SELL at 500 tokens
        
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
            # This is the "earmarked" USD for this specific trade
            desired_usd_to_spend = available_usd * trade_size_pct
            
            # 2. Apply the hard cap
            # THIS IS THE CRITICAL FIX:
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
            # The log will now show the *capped* amount
            print(f"     > Sent BUY for {quantity_to_buy} {market_asset}. (Desired ${desired_usd_to_spend:.2f}, Capped at ${usd_to_spend:.2f})")
            return (usd_to_spend, None, 0) # Earmark the capped amount

        # --- 3. Process SELL Logic (with new cap) ---
        elif side == "SELL":
            available_tokens_for_market = available_tokens.get(market_asset, 0)
            
            # 1. Calculate desired sell amount based on AI confidence
            desired_tokens_to_sell = available_tokens_for_market * trade_size_pct
            
            # 2. Apply the hard cap
            # THIS IS THE CRITICAL FIX:
            quantity_to_sell = min(desired_tokens_to_sell, MAX_TOKENS_PER_TRADE)
            
            if quantity_to_sell < 0.001:
                print(f"     ACTION FAILED: No tokens to sell ({available_tokens_for_market}) or trade size too small.")
                return (0, None, 0)
            
            await self.ws_exchange.send(json.dumps({
                "action": "trade",
                "market": market_asset,
                "quantity": -quantity_to_sell # Negative for SELL
            }))
            print(f"     > Sent SELL for {quantity_to_sell} {market_asset}. (Desired {desired_tokens_to_sell:.2f}, Capped at {quantity_to_sell:.2f} tokens)")
            return (0, market_asset, quantity_to_sell) # Earmark the capped tokens

        return (0, None, 0)
    
    async def run(self):
        """
        The main entry point for the agent. Connects to the exchange
        and starts its listening loops.
        """
        self._configure_gemini()
        if not self.ai_model:
            print(f"[{self.agent_id}] Cannot run without a valid AI model.")
            return

        print(f"--- Intelligent Trader Agent [{self.agent_id}] v2.0 (AMM Mode) ---")
        
        while True:
            try:
                async with websockets.connect(self.exchange_url) as ws:
                    self.ws_exchange = ws # Store the connection
                    print(f"[{self.agent_id}] Connected to exchange {self.exchange_url}")
                    
                    # Register with the exchange
                    await self.ws_exchange.send(json.dumps({
                        "action": "register",
                        "agent_id": self.agent_id
                    }))
                    
                    # Start the two concurrent listener tasks
                    exchange_listener_task = asyncio.create_task(
                        self._listen_to_exchange()
                    )
                    
                    tweet_listener_task = asyncio.create_task(
                        self._listen_to_tweets()
                    )
                    
                    await asyncio.gather(exchange_listener_task, tweet_listener_task)

            except websockets.exceptions.ConnectionClosed:
                print(f"[{self.agent_id}] Main connection to exchange failed. Retrying in 5s...")
            except Exception as e:
                print(f"[{self.agent_id}] Main loop error: {e}. Retrying in 5s...")
            
            self.ws_exchange = None # Clear the connection
            await asyncio.sleep(5)

# --- MAIN SCRIPT TO RUN THE TRADERS ---
async def main():
    gemini_api_key = os.getenv("GEMINI_API_KEY_TRADE") # Use your .env var names
    gemini_api_key2 = os.getenv("GEMINI_API_KEY_TRADE_2")
    
    if not gemini_api_key or not gemini_api_key2:
        print("Error: Make sure GEMINI_API_KEY and GEMINI_API_KEY_2 are in .env file.")
        return

    # --- Create Your Trader Instances ---
    trader_alpha = AgenticTrader(
        agent_id="Trader_Alpha_v3_AMM",
        gemini_api_key=gemini_api_key,
        buffer_size=3
    )
    
    trader_beta = AgenticTrader(
        agent_id="Trader_Beta_v3_AMM",
        gemini_api_key=gemini_api_key2,
        buffer_size=8,
        history_size=100
    )
    
    # --- Start Them Up ---
    print("--- Starting all trader agents (AMM Mode)... ---")
    await asyncio.gather(
        trader_alpha.run(),
        trader_beta.run()
    )

if __name__ == "__main__":
    asyncio.run(main())