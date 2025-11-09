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
# We load .env here to get the API key for the main script
load_dotenv() 

# --- AI PROMPT (Defined here to be used as a default) ---
AI_SYSTEM_PROMPT_TRADER = """
You are an autonomous prediction market trader. Your job is to both 
find arbitrage opportunities and actively provide liquidity by buying 
and selling tokens based on your beliefs — even if there are no 
existing buyers or sellers at the moment.

You will receive:
1. "NEW TWEETS": Real-time information and signals.
2. "ACTIVE MARKETS": Current order book prices (best bids/asks).
3. "CURRENT_PORTFOLIO": Your USD and token holdings.

Your goals:
1. Infer a "true_probability" for each market from the news.
2. Compare your belief to current market prices.
3. Decide whether to BUY, SELL, or POST a resting order to earn spread.
4. Manage your inventory — don’t over-accumulate tokens or cash.

### DECISION RULES

- **BUY Opportunity:** If `true_probability` > `best_ask`, buy aggressively.
- **SELL Opportunity:** If `true_probability` < `best_bid`, sell aggressively.
- **No Counterparty?** You may still place *resting limit orders* slightly
  away from your fair value (e.g., ±0.05) to attract other traders.
  This increases liquidity and potential profit when others trade later.
- **Rebalancing:** 
  - If holding too many tokens and price is near or above your fair value, SELL.
  - If holding too much USD and price is below your fair value, BUY.
- **Profit Taking:** Sell a portion of holdings after favorable moves.
- **Risk Management:** Avoid going all-in on any single market.

### RESPONSE FORMAT

Respond **only** with a compact JSON object like:

```json
{"actions": [
  {
    "action": "PLACE_ORDER",
    "side": "BUY",
    "market_name": "Election2028_BidenWins_YES",
    "trade_size_pct": 0.25,
    "true_probability": 0.65, 
    "price_offset": +0.03,
    "reason": "My true prob 0.68 > best ask 0.60. Buying 25% of USD, posting at +0.03."
  },
  {
    "action": "PLACE_ORDER",
    "side": "SELL",
    "market_name": "ProjectChimeraLaunch_YES",
    "trade_size_pct": 0.40,
    "true_probability": 0.45,
    "price_offset": -0.02,
    "reason": "I think fair value 0.45 < market 0.50, selling 40% holdings slightly under best bid."
  }
]}
"""


class AgenticTrader:
    """
    An autonomous AI agent that trades in prediction markets based on
    real-time tweet data.
    """
    
    def __init__(self, 
                 agent_id: str, 
                 gemini_api_key: str, 
                 exchange_url: str = "ws://localhost:8767", 
                 tweet_server_uri: str = "ws://localhost:8765", 
                 buffer_size: int = 5, 
                 history_size: int = 50,
                 system_prompt: str = AI_SYSTEM_PROMPT_TRADER):
        """
        Initializes a new trader instance.

        Args:
            agent_id: The unique name for this agent (e.g., "Trader_Alpha").
            gemini_api_key: The API key for Google Gemini.
            exchange_url: WebSocket URL for the exchange.
            tweet_server_uri: WebSocket URL for the tweet server.
            buffer_size: Number of tweets to batch before AI analysis.
            history_size: Max number of recent tweets to keep as context.
            system_prompt: The system prompt to guide the AI's behavior.
        """
        self.agent_id = agent_id
        self.gemini_api_key = gemini_api_key
        self.exchange_url = exchange_url
        self.tweet_server_uri = tweet_server_uri
        self.buffer_size = buffer_size
        self.system_prompt = system_prompt
        
        self.ai_model = None
        self.ws_exchange = None # To store the active connection
        
        # --- Instance State (Replaces global variables) ---
        self.portfolio = defaultdict(float)
        self.active_markets = {}
        self.historical_context = deque(maxlen=history_size)
        
        print(f"--- Trader [{self.agent_id}] Initialized ---")

    def _configure_gemini(self):
        """Sets up the Gemini AI model for this instance."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.ai_model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-09-2025', # Updated to a common, powerful model
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

        # 2. Format active markets
        market_lines = ["\n--- ACTIVE MARKETS (Current Prices) ---"]
        if not self.active_markets:
            market_lines.append("None")
        else:
            for market, state in self.active_markets.items():
                market_lines.append(
                    f"- {market}: "
                    f"best_ask (buy at): {state.get('best_ask', 'N/A')}, "
                    f"best_bid (sell at): {state.get('best_bid', 'N/A')}"
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

        # Pass the agent's portfolio to the prompt formatter
        prompt_text = self._format_prompt_for_ai(current_buffer)
        
        print(f"\n[{self.agent_id}] --- Sending prompt to AI for analysis ---")
        # print(prompt_text) # Uncomment for debugging the prompt
        
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
        """
        Listens FOR messages FROM the exchange.
        This is our "Observe" loop for market state.
        """
        print(f"[{self.agent_id}] Exchange listener started.")
        try:
            # self.ws_exchange is set by the run() method
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
                    self.active_markets[data['market_yes']] = {"status": "OPEN"}
                    self.active_markets[data['market_no']] = {"status": "OPEN"}

                elif msg_type == 'market_resolved':
                    print(f"[{self.agent_id}] Market Resolved: {data['market_name']}")
                    market_yes = f"{data['market_name']}_YES"
                    market_no = f"{data['market_name']}_NO"
                    if market_yes in self.active_markets: del self.active_markets[market_yes]
                    if market_no in self.active_markets: del self.active_markets[market_no]

                elif msg_type == 'order_book_update':
                    market = data['market']
                    if market in self.active_markets:
                        bids = data.get('bids', [])
                        asks = data.get('asks', [])
                        
                        self.active_markets[market]['best_bid'] = bids[0][0] if bids else None
                        self.active_markets[market]['best_ask'] = asks[0][0] if asks else None

                elif msg_type == 'trade_executed':
                    if data['buyer'] == self.agent_id:
                        print(f"*** [{self.agent_id}] => Our BUY order was filled! Bought {data['quantity']} {data['market']} ***")
                    if data['seller'] == self.agent_id:
                        print(f"*** [{self.agent_id}] => Our SELL order was filled! Sold {data['quantity']} {data['market']} ***")

                elif msg_type == 'error':
                    print(f"[{self.agent_id}] Exchange Error: {data['message']}")

        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.agent_id}] Connection to exchange lost.")
        except Exception as e:
            print(f"[{self.agent_id}] Error in exchange listener: {e}")

    async def _listen_to_tweets(self):
        """
        Listens FOR messages FROM the tweet stream.
        This is our "Orient, Decide, Act" loop.
        """
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
                                # Earmarking Logic
                                available_usd_for_batch = self.portfolio.get("USD", 0)
                                available_tokens_for_batch = self.portfolio.copy()
                                
                                print(f"   [{self.agent_id}] Starting batch with ${available_usd_for_batch:.2f} available.")
                                
                                for action in actions:
                                    if action.get("action") == "PLACE_ORDER":
                                        
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

    async def _process_ai_trade(self, action: dict, available_usd: float, available_tokens: dict) -> (float, str, float):
        """
        Calculates quantity from percentage and sends order.
        Returns (usd_spent, token_market, tokens_sold) to earmark funds.
        """
        
        # --- 1. Get All AI Parameters ---
        side = action.get("side")
        market = action.get("market_name")
        trade_size_pct = action.get("trade_size_pct", 0.1) # Default to 10%
        true_prob = action.get("true_probability")
        price_offset = action.get("price_offset", 0)
        
        print(f"--- [{self.agent_id}] AI ACTION: {side} {market} ({trade_size_pct*100}%) ---")
        print(f"    Reason: {action.get('reason')}")

        if true_prob is None:
            print(f"    ACTION FAILED: AI response did not include 'true_probability'. Cannot calculate limit price.")
            return (0, None, 0)
        
        # --- 2. Calculate Agent's Limit Price ---
        # This is the price the agent is *willing* to trade at.
        limit_price = round(true_prob + price_offset, 3)
        limit_price = max(0.01, min(0.99, limit_price)) # Clamp price
        
        price_to_use = limit_price # Default to our own limit price (making a market)
        
        # --- 3. Process BUY Logic ---
        if side == "BUY":
            best_ask = self.active_markets.get(market, {}).get('best_ask')
            
            # Check if we should be a TAKER (aggressive)
            if best_ask is not None and limit_price >= best_ask:
                # Our limit price is higher than or equal to the cheapest seller.
                # We should TAKE their offer, as it's a better deal.
                print(f"    > Aggressively TAKING ask. Our limit {limit_price} >= best ask {best_ask}")
                price_to_use = best_ask
            else:
                print(f"    > Placing resting BID at our limit price {limit_price}")

            usd_to_spend = available_usd * trade_size_pct
            
            # This check is now safe from divide-by-zero
            quantity_to_buy = round(usd_to_spend / price_to_use, 3) 

            if quantity_to_buy < 0.001:
                print(f"    ACTION FAILED: Trade size too small (Qty: {quantity_to_buy}).")
                return (0, None, 0)

            actual_cost = quantity_to_buy * price_to_use
            if available_usd < actual_cost:
                print(f"    ACTION FAILED: Insufficient *batch* USD. Need ${actual_cost:.2f}, have ${available_usd:.2f}")
                return (0, None, 0)
                
            await self.ws_exchange.send(json.dumps({
                "action": "place_order",
                "market": market,
                "side": "buy",
                "price": price_to_use, 
                "quantity": quantity_to_buy
            }))
            print(f"    > Sent BUY order for {quantity_to_buy} {market} @ ${price_to_use}")
            return (actual_cost, None, 0) # (usd_spent, token_market, tokens_sold)

        # --- 4. Process SELL Logic ---
        elif side == "SELL":
            best_bid = self.active_markets.get(market, {}).get('best_bid')
            available_tokens_for_market = available_tokens.get(market, 0)
            
            # Check if we should be a TAKER (aggressive)
            if best_bid is not None and limit_price <= best_bid:
                # Our limit price is lower than or equal to the highest buyer.
                # We should TAKE their offer, as it's a better deal.
                print(f"    > Aggressively TAKING bid. Our limit {limit_price} <= best bid {best_bid}")
                price_to_use = best_bid
            else:
                print(f"    > Placing resting ASK at our limit price {limit_price}")

            quantity_to_sell = round(available_tokens_for_market * trade_size_pct, 3)

            if quantity_to_sell < 0.001:
                print(f"    ACTION FAILED: No tokens to sell ({available_tokens_for_market}) or trade size too small.")
                return (0, None, 0)
            
            await self.ws_exchange.send(json.dumps({
                "action": "place_order",
                "market": market,
                "side": "sell",
                "price": price_to_use, 
                "quantity": quantity_to_sell
            }))
            print(f"    > Sent SELL order for {quantity_to_sell} {market} @ ${price_to_use}")
            return (0, market, quantity_to_sell) # (usd_spent, token_market, tokens_sold)

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

        print(f"--- Intelligent Trader Agent [{self.agent_id}] v2.0 (Stateful) ---")
        
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
# (This file, 'run_traders.py', remains unchanged)

async def main():
    gemini_api_key = os.getenv("GEMINI_API_KEY_TRADE")
    gemini_api_key2 = os.getenv("GEMINI_API_KEY_TRADE_2")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    # --- Create Your Trader Instances ---
    trader_alpha = AgenticTrader(
        agent_id="Trader_Alpha_v3",
        gemini_api_key=gemini_api_key,
        buffer_size=3
    )
    
    trader_beta = AgenticTrader(
        agent_id="Trader_Beta_v3",
        gemini_api_key=gemini_api_key2,
        buffer_size=8,
        history_size=100
    )
    
    # --- Start Them Up ---
    print("--- Starting all trader agents... ---")
    await asyncio.gather(
        trader_alpha.run(),
        trader_beta.run()
    )

if __name__ == "__main__":
    # This assumes you save the code above as `agent_trader.py`
    # and you run it from a separate `run_traders.py`
    # If this *is* the main file, just uncomment the line below:
    asyncio.run(main())
