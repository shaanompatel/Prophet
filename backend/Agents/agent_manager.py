import asyncio
import websockets
import json
import os
import re
from collections import deque, defaultdict
from dotenv import load_dotenv
from openai import AsyncOpenAI  # <-- Using OpenAI client for OpenRouter

# --- CONFIGURATION ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EXCHANGE_URL = "ws://localhost:8767"

# --- 1. UPDATED, STRICTER BASE PROMPT ---
AI_BASE_PROMPT = """
You are an autonomous prediction market trader. Your job is to trade
against an Automated Market Maker (AMM).

You will receive:
1. "NEW TWEETS": Real-time information and signals.
2. "ACTIVE MARKETS": The current AMM price for each asset.
3. "CURRENT_PORTFOLIO": Your USD and token holdings.

### CRITICAL RULES
1.  **You can ONLY 'SELL' an asset that is listed in your `CURRENT_PORTFOLIO` with a quantity greater than 0.** Do not attempt to sell assets you do not own.
2.  **Do NOT trade if the `current_price` is very low or high** (e.g., < 0.001 or > 0.999). These markets are illiquid, and you MUST ignore them.
3.  Your response MUST be ONLY a compact JSON object. Do not include
    any other text, markdown, or explanations.

### RESPONSE FORMAT
```json
{"actions": [
  {
    "action": "TRADE",
    "side": "BUY",
    "market_asset": "MarketName_YES",
    "trade_size_pct": 0.25,
    "true_probability": 0.65,
    "reason": "My reason..."
  },
  {
    "action": "TRADE",
    "side": "SELL",
    "market_asset": "MarketName_NO",
    "trade_size_pct": 0.50,
    "true_probability": 0.30,
    "reason": "My reason..."
  }
]}
```

YOUR STRATEGY
You will follow this specific strategy: """


class AgenticTrader:
    def __init__(
        self,
        agent_id: str,
        openrouter_key: str,
        model_name: str,
        strategy_prompt: str,
        exchange_url: str = EXCHANGE_URL,
        tweet_server_uri: str = "ws://localhost:8765",
        buffer_size: int = 5,
        history_size: int = 50
    ):
        self.agent_id = agent_id
        self.openrouter_key = openrouter_key
        self.model_name = model_name
        self.exchange_url = exchange_url
        self.tweet_server_uri = tweet_server_uri
        self.buffer_size = buffer_size

        # --- Build system prompt ---
        # This is correct: it appends the specific strategy after the base prompt
        self.system_prompt = f"{AI_BASE_PROMPT}\n{strategy_prompt}"

        self.ai_model = None
        self.ws_exchange = None

        self.portfolio = defaultdict(float)
        self.active_markets = {}
        self.historical_context = deque(maxlen=history_size)

        print(f"--- Trader [{self.agent_id}] Initialized (Model: {self.model_name}) ---")

    def _configure_openrouter(self):
        """Sets up the OpenRouter AI model for this instance."""
        try:
            self.ai_model = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
            )
            print(f"[{self.agent_id}] OpenRouter AI Model configured for {self.model_name}.")
        except Exception as e:
            print(f"[{self.agent_id}] Error: Failed to configure OpenRouter: {e}")
            self.ai_model = None

    @staticmethod
    def _static_format_tweet(tweet: dict) -> str:
        user = tweet.get('user', {}).get('username', 'unknown')
        followers = tweet.get('user', {}).get('followers', 0)
        text = tweet.get('text', '')
        return f"@{user} ({followers} followers): \"{text}\""

    @staticmethod
    def _static_clean_json_response(text: str) -> str:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return ""

    def _format_prompt_for_ai(self, current_buffer: list) -> str:
        # 1. Format portfolio
        portfolio_lines = ["\n--- CURRENT_PORTFOLIO ---"]
        if not self.portfolio:
            portfolio_lines.append("USD: 10000.0 (Initial)")
        else:
            for asset, quantity in self.portfolio.items():
                portfolio_lines.append(f"{asset}: {quantity:.2f}")

        # 2. Format active markets
        market_lines = ["\n--- ACTIVE MARKETS (Current AMM Prices) ---"]
        if not self.active_markets:
            market_lines.append("None")
        else:
            for asset, state in self.active_markets.items():
                # Format price to avoid sending 'N/A'
                price_str = f"{state.get('price'):.4f}" if state.get('price') is not None else "N/A"
                market_lines.append(f"- {asset}: current_price: {price_str}")

        # 3. Format tweets
        history_lines = [
            f"\n--- RECENT HISTORY (Last {len(self.historical_context)} Processed Tweets) ---"
        ] + [self._static_format_tweet(tweet) for tweet in self.historical_context]

        new_lines = ["\n--- NEW TWEETS (To Analyze) ---"] + [
            self._static_format_tweet(tweet) for tweet in current_buffer
        ]

        return "\n".join(portfolio_lines + market_lines + history_lines + new_lines)

    async def _get_ai_decision(self, current_buffer: list) -> dict:
        """Sends portfolio, market state, and tweets to OpenRouter."""
        if not self.ai_model:
            return {"actions": []}

        user_prompt = self._format_prompt_for_ai(current_buffer)
        print(f"\n[{self.agent_id}] --- Sending prompt to AI ({self.model_name}) ---")

        try:
            chat_completion = await self.ai_model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            response_text = chat_completion.choices[0].message.content
            json_text = self._static_clean_json_response(response_text)

            if not json_text:
                print(f"   [{self.agent_id}] AI Error: No JSON found in response: {response_text}")
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
                    self.portfolio.clear()
                    self.portfolio.update(data['balances'])

                elif msg_type == 'new_market':
                    self.active_markets[data['market_yes']] = {"status": "OPEN"}
                    self.active_markets[data['market_no']] = {"status": "OPEN"}

                elif msg_type == 'market_resolved':
                    market_yes = f"{data['market_name']}_YES"
                    market_no = f"{data['market_name']}_NO"
                    self.active_markets.pop(market_yes, None)
                    self.active_markets.pop(market_no, None)

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
                            ai_decision = await self._get_ai_decision(tweet_buffer)
                            self.historical_context.extend(tweet_buffer)
                            tweet_buffer = []

                            actions = ai_decision.get("actions", [])
                            if not actions:
                                print(f"   [{self.agent_id}] AI: No actions to take.")
                            else:
                                available_usd_for_batch = self.portfolio.get("USD", 0)
                                available_tokens_for_batch = self.portfolio.copy()

                                for action in actions:
                                    if action.get("action") == "TRADE":
                                        usd_spent, token_market, tokens_sold = await self._process_ai_trade(
                                            action,
                                            available_usd_for_batch,
                                            available_tokens_for_batch
                                        )
                                        available_usd_for_batch -= usd_spent
                                        if token_market:
                                            available_tokens_for_batch[token_market] -= tokens_sold
            except Exception:
                print(f"[{self.agent_id}] Tweet stream connection lost. Reconnecting in 5s...")
            await asyncio.sleep(5)

    # --- 2. UPDATED, SAFER TRADE LOGIC ---
    async def _process_ai_trade(self, action: dict, available_usd: float, available_tokens: dict) -> tuple:
        """
        Calculates quantity based on a STANDARD_TRADE_QUANTITY to avoid
        divide-by-zero errors with market price.
        """
        
        # --- NEW LOGIC ---
        # A "full" 100% trade (trade_size_pct: 1.0) will be 100 shares.
        # A 30% trade (trade_size_pct: 0.3) will be 30 shares.
        # This avoids all division by 'current_price'.
        STANDARD_TRADE_QUANTITY = 100.0 

        side = action.get("side")
        market_asset = action.get("market_asset")
        trade_size_pct = action.get("trade_size_pct", 0.1)  # Default to 10% (10 shares)
        
        # Handle cases where AI might pass a string or None
        try:
            trade_size_pct = float(trade_size_pct)
        except (ValueError, TypeError):
            trade_size_pct = 0.1  # Default to 10%

        print(f"--- [{self.agent_id}] AI ACTION: {side} {market_asset} ({trade_size_pct*100}%) ---")
        print(f"     Reason: {action.get('reason')}")

        current_price = self.active_markets.get(market_asset, {}).get('price')
        if current_price is None:
            print(f"     ACTION FAILED: No price data for {market_asset}.")
            return (0, None, 0)

        # --- NEW GUARDRAIL (matches prompt rule 2) ---
        if current_price < 0.001 or current_price > 0.999:
            print(f"     ACTION FAILED: Market is illiquid (Price: {current_price:.6f}). No trade.")
            return (0, None, 0)
        
        # --- 2. Process BUY Logic ---
        if side == "BUY":
            # Calculate quantity directly
            quantity_to_buy = round(STANDARD_TRADE_QUANTITY * trade_size_pct, 3)

            if quantity_to_buy < 0.001:
                print(f"     ACTION FAILED: Trade quantity too small.")
                return (0, None, 0)

            # Estimate cost for earmarking.
            # We add a 50% buffer for slippage estimation, as this is just a
            # client-side check. The server will do the *real* check.
            estimated_cost = quantity_to_buy * current_price
            if (estimated_cost * 1.5) > available_usd:
                print(f"     ACTION FAILED: Not enough *available* USD for trade. Need ~${estimated_cost * 1.5:.2f}, have ${available_usd:.2f}")
                return (0, None, 0)
                
            await self.ws_exchange.send(json.dumps({
                "action": "trade",
                "market": market_asset,
                "quantity": quantity_to_buy  # Positive for BUY
            }))
            print(f"     > Sent BUY for {quantity_to_buy} {market_asset}. (Estimated Cost: ${estimated_cost:.2f})")
            
            # Earmark the *estimated* cost. The server will charge the true cost.
            return (estimated_cost, None, 0) 

        # --- 3. Process SELL Logic ---
        elif side == "SELL":
            available_tokens_for_market = available_tokens.get(market_asset, 0)
            
            # Calculate quantity directly
            quantity_to_sell = round(STANDARD_TRADE_QUANTITY * trade_size_pct, 3)

            # --- NEW GUARDRAIL (matches prompt rule 1) ---
            if quantity_to_sell > available_tokens_for_market:
                print(f"     ACTION FAILED: Not enough tokens for trade. Need {quantity_to_sell}, have {available_tokens_for_market}")
                return (0, None, 0)

            if quantity_to_sell < 0.001:
                print(f"     ACTION FAILED: Trade quantity too small.")
                return (0, None, 0)
            
            await self.ws_exchange.send(json.dumps({
                "action": "trade",
                "market": market_asset,
                "quantity": -quantity_to_sell  # Negative for SELL
            }))
            print(f"     > Sent SELL for {quantity_to_sell} {market_asset}.")
            return (0, market_asset, quantity_to_sell)  # Earmark the tokens

        return (0, None, 0)

    async def run(self):
        """Main entry point for the agent."""
        self._configure_openrouter()
        if not self.ai_model:
            print(f"[{self.agent_id}] Cannot run without a valid AI model.")
            return

        while True:
            try:
                async with websockets.connect(self.exchange_url) as ws:
                    self.ws_exchange = ws
                    print(f"[{self.agent_id}] Connected to exchange {self.exchange_url}")

                    await self.ws_exchange.send(json.dumps({
                        "action": "register", "agent_id": self.agent_id
                    }))

                    exchange_listener_task = asyncio.create_task(self._listen_to_exchange())
                    tweet_listener_task = asyncio.create_task(self._listen_to_tweets())

                    await asyncio.gather(exchange_listener_task, tweet_listener_task)

            except websockets.exceptions.ConnectionClosed:
                print(f"[{self.agent_id}] Main connection to exchange failed. Retrying in 5s...")
            except Exception as e:
                print(f"[{self.agent_id}] Main loop error: {e}. Retrying in 5s...")

            self.ws_exchange = None
            await asyncio.sleep(5)


# --- MANAGER FUNCTIONS ---
def spawn_agent_from_command(data: dict):
    agent_id = data.get("name")
    model = data.get("model")
    strategy = data.get("strategy")

    if not all([agent_id, model, strategy]):
        print("[MANAGER] Error: Invalid spawn command, missing fields.")
        return

    if not OPENROUTER_API_KEY:
        print(f"[MANAGER] Error: Cannot spawn {agent_id}, OPENROUTER_API_KEY not set.")
        return

    print(f"[MANAGER] Spawning new agent: {agent_id} (Model: {model})")

    trader = AgenticTrader(
        agent_id=agent_id,
        openrouter_key=OPENROUTER_API_KEY,
        model_name=model,
        strategy_prompt=strategy,
        buffer_size=5
    )
    asyncio.create_task(trader.run())


def spawn_default_agents():
    default_strategy_alpha = "You are a cautious trader. You only trade with 10% of your assets (trade_size_pct: 0.1) and only when you are very confident (true_probability is >0.15 from market price)."
    default_strategy_beta = "You are an aggressive trader. You use 30% of your assets (trade_size_pct: 0.3) and trade on any small perceived edge (true_probability is >0.05 from market price)."

    spawn_agent_from_command({
        "name": "Trader_GPT",
        "model": "anthropic/claude-3-haiku",  # Using a more standard, reliable model
        "strategy": default_strategy_alpha
    })

    spawn_agent_from_command({
        "name": "Trader_Llama",
        "model": "meta-llama/llama-3-8b-instruct",  # A good, fast model
        "strategy": default_strategy_beta
    })


async def main():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file. Cannot start manager.")
        return

    spawn_default_agents()
    print(f"--- Agent Manager connecting to Exchange at {EXCHANGE_URL} ---")

    while True:
        print("here")
        try:
            async with websockets.connect(EXCHANGE_URL) as ws:
                await ws.send(json.dumps({"action": "register_manager"}))
                print("[MANAGER] Successfully registered with exchange.")

                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") == "command_spawn_agent":
                        print("[MANAGER] Received spawn command from exchange.")
                        spawn_agent_from_command(data.get("data"))

        except websockets.exceptions.ConnectionClosed:
            print("[MANAGER] Connection to exchange lost. Reconnecting in 5s...")
        except Exception as e:
            print(f"[MANAGER] Error: {e}. Retrying in 5s...")

        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())