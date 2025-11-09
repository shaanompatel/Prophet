import asyncio
import json
import websockets
from collections import defaultdict
import math  # --- NEW ---
import time

# --- Global State ---
CONNECTED_CLIENTS = {} # Trader clients connected to this exchange
ACCOUNTS = defaultdict(lambda: defaultdict(float))
LMSR_MARKETS = {} # --- REPLACED ORDER_BOOKS ---
RESOLVED_MARKETS = set() 

# --- Configuration ---
AI_ACTION_URL = "ws://localhost:8766"  # URL of *your* AI server
EXCHANGE_HOST = "localhost"            # Host *this* exchange server on...
EXCHANGE_PORT = 8767                   # ...this port (for traders)

MAKER_AGENT_ID = "market_maker" # The "account ID" for the AMM "house"
TRADER_INITIAL_USD = 10_000.0
MAKER_INITIAL_USD = 1_000_000.0
LMSR_B_VALUE = 100.0 # --- NEW: Default liquidity parameter ---
                     # Higher 'b' = more liquid market, less slippage

# --- Exchange Server Logic (for Traders) ---

async def broadcast_to_traders(message):
    """Sends a JSON message to all connected traders."""
    if CONNECTED_CLIENTS:
        tasks = [
            asyncio.create_task(client.send(json.dumps(message)))
            for client in CONNECTED_CLIENTS.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

async def send_to_trader(agent_id, message):
    """Sends a JSON message to a specific trader."""
    if agent_id in CONNECTED_CLIENTS:
        try:
            await CONNECTED_CLIENTS[agent_id].send(json.dumps(message))
        except websockets.ConnectionClosed:
            pass

async def get_account_update(agent_id):
    return {"type": "account_update", "balances": ACCOUNTS.get(agent_id, {})}

# --- NEW: LMSR Core Functions ---

def get_lmsr_cost(q_yes: float, q_no: float, b: float) -> float:
    """Calculates the total cost C(q_yes, q_no) for an LMSR market."""
    try:
        return b * math.log(math.exp(q_yes / b) + math.exp(q_no / b))
    except OverflowError:
        # Handle potential overflow if q is very large
        max_q = max(q_yes, q_no)
        return max_q + b * math.log(math.exp((q_yes - max_q) / b) + math.exp((q_no - max_q) / b))

def get_lmsr_prices(market_name: str) -> dict:
    """Calculates the current instantaneous prices for a market."""
    market = LMSR_MARKETS.get(market_name)
    if not market:
        return {"error": "Market not found"}
    
    q_yes, q_no, b = market["q_yes"], market["q_no"], market["b"]
    
    try:
        exp_yes = math.exp(q_yes / b)
        exp_no = math.exp(q_no / b)
        sum_exp = exp_yes + exp_no
    except OverflowError:
        # Handle potential overflow
        max_q = max(q_yes, q_no)
        exp_yes = math.exp((q_yes - max_q) / b)
        exp_no = math.exp((q_no - max_q) / b)
        sum_exp = exp_yes + exp_no

    if sum_exp == 0:
        return {"YES": 0.5, "NO": 0.5}
        
    price_yes = exp_yes / sum_exp
    price_no = exp_no / sum_exp
    
    return {"YES": price_yes, "NO": price_no}

async def broadcast_price_update(market_name: str, individual_client=None):
    """Broadcasts the new AMM prices to all traders."""
    prices = get_lmsr_prices(market_name)
    if "error" in prices:
        return

    market = LMSR_MARKETS[market_name] # Get market to access q values

    update_msg = {
        "type": "price_update",
        "market": market_name,
        "price_yes": prices["YES"],
        "price_no": prices["NO"],
        "q_yes": market["q_yes"], # Add q values for the UI
        "q_no": market["q_no"]
    }
    
    if individual_client:
        # Send to just one client (for syncing on connect)
        await individual_client.send(json.dumps(update_msg))
    else:
        # Broadcast to all
        await broadcast_to_traders(update_msg)
    
    return update_msg # Return the message for the sync handler

# --- REMOVED: get_order_book_update ---
# --- REMOVED: match_orders ---

async def trader_client_handler(websocket):
    """Handles new traders AND spectators connecting to *this* exchange server."""
    agent_id = None
    client_type = "trader" # Default
    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get("action")
            
            if action == "register":
                agent_id = data.get("agent_id")
                if agent_id in CONNECTED_CLIENTS:
                    await websocket.send(json.dumps({"type": "error", "message": "Agent ID already connected"}))
                    continue
                
                CONNECTED_CLIENTS[agent_id] = websocket
                print(f"[TRADER REGISTER] Agent registered: {agent_id}")
                
                if agent_id == MAKER_AGENT_ID:
                    if "USD" not in ACCOUNTS[agent_id]: ACCOUNTS[agent_id]["USD"] = MAKER_INITIAL_USD
                else:
                    if "USD" not in ACCOUNTS[agent_id]: ACCOUNTS[agent_id]["USD"] = TRADER_INITIAL_USD
                
                await send_to_trader(agent_id, {"type": "registered", "agent_id": agent_id})
                await send_to_trader(agent_id, await get_account_update(agent_id))
            
            # --- NEW BLOCK FOR THE FRONTEND ---
            elif action == "register_spectator":
                client_type = "spectator"
                # Generate a unique ID for this spectator
                agent_id = f"spectator_{int(time.time() * 1000)}"
                CONNECTED_CLIENTS[agent_id] = websocket
                print(f"[SPECTATOR REGISTER] Spectator connected: {agent_id}")
                
                # --- SYNC FULL STATE ---
                # 1. Send all existing account balances
                for existing_agent, balances in ACCOUNTS.items():
                    await send_to_trader(agent_id, {
                        "type": "account_update", 
                        "balances": balances,
                        "agent_id": existing_agent # Manually add agent_id for the frontend
                    })
                
                # 2. Send all existing market prices
                for market_name in LMSR_MARKETS:
                    if market_name not in RESOLVED_MARKETS:
                        await send_to_trader(agent_id, await broadcast_price_update(market_name, individual_client=websocket))
                
                await send_to_trader(agent_id, {"type": "spectator_registered"})
                # This client is now in CONNECTED_CLIENTS and will get all future broadcasts
            
            elif not agent_id:
                await websocket.send(json.dumps({"type": "error", "message": "You must register first"}))
                continue

            elif action == "trade":
                if client_type == "spectator": # Spectators can't trade
                     await websocket.send(json.dumps({"type": "error", "message": "Spectators cannot trade"}))
                     continue
                await handle_amm_trade(agent_id, data)

    except websockets.exceptions.ConnectionClosed:
        if client_type == "spectator":
             print(f"[SPECTATOR DISCONNECT] Spectator {agent_id} disconnected.")
        else:
            print(f"[TRADER DISCONNECT] Client {agent_id} disconnected.")
    finally:
        if agent_id in CONNECTED_CLIENTS:
            del CONNECTED_CLIENTS[agent_id]

# --- REPLACED: handle_place_order -> handle_amm_trade ---
async def handle_amm_trade(agent_id, data):
    """Processes a 'trade' request against the AMM."""
    asset_name = data.get("market") # e.g., "Election_YES"
    
    try:
        quantity = float(data.get("quantity"))
    except Exception:
        return await send_to_trader(agent_id, {"type": "error", "message": "Invalid quantity"})

    try:
        market_name, outcome = asset_name.rsplit('_', 1)
        if outcome not in ["YES", "NO"]: raise ValueError()
    except Exception:
        return await send_to_trader(agent_id, {"type": "error", "message": f"Invalid market asset name: {asset_name}"})
    
    if market_name in RESOLVED_MARKETS:
        return await send_to_trader(agent_id, {"type": "error", "message": f"Market {market_name} is resolved."})
    if market_name not in LMSR_MARKETS:
        return await send_to_trader(agent_id, {"type": "error", "message": f"Market {market_name} does not exist"})

    market = LMSR_MARKETS[market_name]
    q_yes, q_no, b = market["q_yes"], market["q_no"], market["b"]
    
    current_cost = get_lmsr_cost(q_yes, q_no, b)
    
    trade_cost = 0.0
    
    # --- BUYING (quantity is positive) ---
    if quantity > 0:
        new_q_yes = q_yes + quantity if outcome == "YES" else q_yes
        new_q_no = q_no + quantity if outcome == "NO" else q_no
        
        new_cost = get_lmsr_cost(new_q_yes, new_q_no, b)
        trade_cost = new_cost - current_cost
        
        if ACCOUNTS[agent_id]["USD"] < trade_cost:
            return await send_to_trader(agent_id, {"type": "error", "message": "Insufficient USD"})
        
        ACCOUNTS[agent_id]["USD"] -= trade_cost
        ACCOUNTS[agent_id][asset_name] += quantity
        ACCOUNTS[MAKER_AGENT_ID]["USD"] += trade_cost # AMM "house" gets the USD
        
        market["q_yes"], market["q_no"] = new_q_yes, new_q_no
        print(f"[TRADE] {agent_id} BUYS {quantity} {asset_name} for ${trade_cost:.3f}")

    # --- SELLING (quantity is negative) ---
    elif quantity < 0:
        dq = abs(quantity)
        if ACCOUNTS[agent_id].get(asset_name, 0) < dq:
            return await send_to_trader(agent_id, {"type": "error", "message": f"Insufficient {asset_name} tokens"})
        
        new_q_yes = q_yes - dq if outcome == "YES" else q_yes
        new_q_no = q_no - dq if outcome == "NO" else q_no

        new_cost = get_lmsr_cost(new_q_yes, new_q_no, b)
        payout = current_cost - new_cost
        trade_cost = -payout # Store negative cost for logging
        
        ACCOUNTS[agent_id]["USD"] += payout
        ACCOUNTS[agent_id][asset_name] -= dq
        ACCOUNTS[MAKER_AGENT_ID]["USD"] -= payout # AMM "house" pays out USD
        
        market["q_yes"], market["q_no"] = new_q_yes, new_q_no
        print(f"[TRADE] {agent_id} SELLS {dq} {asset_name} for ${payout:.3f}")
    
    else:
        return await send_to_trader(agent_id, {"type": "error", "message": "Quantity cannot be zero"})

    # --- Notify participants ---
    print(get_exchange_state_string())
    
    # Send private update to the trader
    await send_to_trader(agent_id, await get_account_update(agent_id))
    
    # --- BROADCAST ACCOUNT UPDATES FOR SPECTATORS ---
    # We create a custom message here to include the agent_id in the broadcast
    trader_update = await get_account_update(agent_id)
    trader_update["agent_id"] = agent_id
    await broadcast_to_traders(trader_update)
    
    maker_update = await get_account_update(MAKER_AGENT_ID)
    maker_update["agent_id"] = MAKER_AGENT_ID
    await broadcast_to_traders(maker_update)
    # --- END OF NEW BLOCK ---
    
    trade_message = {
        "type": "trade_executed", "market": asset_name, "quantity": quantity,
        "agent": agent_id, "cost": trade_cost
    }
    await broadcast_to_traders(trade_message)
    await broadcast_price_update(market_name) # This will broadcast to everyone

# --- AI Client Logic (Listens to your script) ---

async def internal_handle_create(data: dict):
    """Processes a CREATE action from the AI."""
    market_name = data.get("market_name")
    
    # Clamp probability to avoid math.log(0)
    prob_yes = max(0.001, min(0.999, data.get("probability", 0.5)))
    prob_no = 1.0 - prob_yes
    b_value = data.get("b_value", LMSR_B_VALUE)
    
    market_yes = f"{market_name}_YES"
    market_no = f"{market_name}_NO"

    if market_name in LMSR_MARKETS:
        print(f"[AI] Received CREATE for existing market {market_name}. Ignoring.")
        return

    print(f"\n[AI ACTION] Received CREATE: {market_name} (Prob={prob_yes}, b={b_value})")
    
    # 1. Calculate initial q values to set the desired starting price
    # We set initial q_yes = b * log(P_yes) and q_no = b * log(P_no)
    initial_q_yes = b_value * math.log(prob_yes)
    initial_q_no = b_value * math.log(prob_no)
    
    # 2. Create the market
    LMSR_MARKETS[market_name] = {
        "q_yes": initial_q_yes,
        "q_no": initial_q_no,
        "b": b_value
    }
    print(f"  > Initialized LMSR for {market_name} with q_yes={initial_q_yes:.2f}, q_no={initial_q_no:.2f}")

    # 3. Notify all connected *traders*
    await broadcast_to_traders({
        "type": "new_market",
        "market_name": market_name,
        "market_yes": market_yes,
        "market_no": market_no,
        "created_by": MAKER_AGENT_ID
    })
    
    # 4. Broadcast initial prices
    await broadcast_price_update(market_name)

async def internal_handle_resolve(data: dict):
    """Processes a RESOLVE action from the AI."""
    market_name = data.get("market_name")
    outcome = data.get("outcome", "YES").upper()
    
    market_yes = f"{market_name}_YES"
    market_no = f"{market_name}_NO"

    if market_name in RESOLVED_MARKETS:
        print(f"[AI] Received RESOLVE for already resolved market {market_name}. Ignoring.")
        return

    print(f"\n[AI ACTION] Received RESOLVE: {market_name} -> {outcome}")

    # 1. Stop trading
    RESOLVED_MARKETS.add(market_name)
    if market_name in LMSR_MARKETS:
        del LMSR_MARKETS[market_name] # Remove from active AMMs
    
    winning_token = market_yes if outcome == "YES" else market_no
    losing_token = market_no if outcome == "YES" else market_yes

    # 2. Settle accounts (This logic remains the same)
    print(f"[SETTLE] Cashing out {winning_token} at $1.00")
    for agent, balances in ACCOUNTS.items():
        if winning_token in balances:
            winnings = balances[winning_token]
            if winnings > 0:
                print(f"  > Paying {agent} ${winnings} for {winnings} {winning_token} tokens")
                balances["USD"] += winnings
                balances[winning_token] = 0.0
        
        if losing_token in balances:
            if balances[losing_token] > 0:
                print(f"  > Clearing {balances[losing_token]} worthless {losing_token} tokens from {agent}")
                balances[losing_token] = 0.0
                
    # 3. Notify all connected *traders*
    await broadcast_to_traders({
        "type": "market_resolved",
        "market_name": market_name,
        "outcome": outcome,
        "winning_token": winning_token
    })

    # 4. Send final account updates
    for agent_id in CONNECTED_CLIENTS.keys():
        await send_to_trader(agent_id, await get_account_update(agent_id))
    print(f"[SETTLE] Settlement complete for {market_name}.\n")

async def listen_to_ai_actions():
    """Connects to the AI's server and listens for actions."""
    print(f"Connecting to AI Action Server at {AI_ACTION_URL}...")
    while True:
        try:
            async with websockets.connect(AI_ACTION_URL) as ws:
                print("Successfully connected to AI Action Server!")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        action = data.get("action")
                        
                        if action == "CREATE":
                            await internal_handle_create(data)
                        elif action == "RESOLVE":
                            await internal_handle_resolve(data)
                            
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON from AI: {message}")
                    except Exception as e:
                        print(f"Error processing AI message: {e}")

        except websockets.exceptions.ConnectionClosed:
            print("AI Server connection lost. Reconnecting in 5s...")
        except Exception as e:
            print(f"Error connecting to AI Server: {e}. Retrying in 5s...")
        
        await asyncio.sleep(5)

def get_exchange_state_string():
    """Return a nicely formatted string of the current exchange state."""
    output = []
    output.append("\n=== 📊 EXCHANGE STATE ===")

    # --- Balances ---
    output.append("\n--- ACCOUNTS ---")
    for agent, balances in ACCOUNTS.items():
        nonzero_assets = {a: amt for a, amt in balances.items() if amt != 0}
        if not nonzero_assets:
            continue
        output.append(f" {agent}:")
        for asset, amount in sorted(nonzero_assets.items()):
            output.append(f"     {asset:35} {amount:10.3f}")

    # --- LMSR Markets ---
    output.append("\n--- LMSR MARKETS (Prices) ---")
    if not LMSR_MARKETS:
        output.append("  (no active markets)")
        
    for market_name, market in LMSR_MARKETS.items():
        if market_name in RESOLVED_MARKETS:
            continue

        prices = get_lmsr_prices(market_name)
        output.append(f"\n📈 {market_name} (b={market['b']})")
        output.append(f"   YES Price: ${prices['YES']:.4f}")
        output.append(f"   NO Price:  ${prices['NO']:.4f}")
        output.append(f"   (q_yes={market['q_yes']:.2f}, q_no={market['q_no']:.2f})")

    output.append("==========================\n")
    return "\n".join(output)

# --- Main Startup ---

async def main():
    print(f"--- Exchange Server (LMSR AMM) starting... ---") # Updated
    print(f"   > Will listen for traders on ws://{EXCHANGE_HOST}:{EXCHANGE_PORT}")
    print(f"   > Will listen for AI on ws://{AI_ACTION_URL}")
    
    # 1. Initialize the market_maker's USD account
    ACCOUNTS[MAKER_AGENT_ID]["USD"] = MAKER_INITIAL_USD

    # 2. Start the AI listener as a background task
    asyncio.create_task(listen_to_ai_actions())
    
    # 3. Start the trader server
    async with websockets.serve(trader_client_handler, EXCHANGE_HOST, EXCHANGE_PORT):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())