import asyncio
import json
import websockets
from collections import defaultdict
import heapq
import time

# --- Global State ---
CONNECTED_CLIENTS = {} # Trader clients connected to this exchange
ACCOUNTS = defaultdict(lambda: defaultdict(float))
ORDER_BOOKS = defaultdict(lambda: {"bids": [], "asks": []})
RESOLVED_MARKETS = set() 

# --- Configuration ---
AI_ACTION_URL = "ws://localhost:8766"  # URL of *your* AI server
EXCHANGE_HOST = "localhost"            # Host *this* exchange server on...
EXCHANGE_PORT = 8767                   # ...this port (for traders)

MAKER_AGENT_ID = "market_maker" # The "account ID" for the AI
MAKER_INITIAL_TOKENS = 1000
TRADER_INITIAL_USD = 10_000.0
MAKER_INITIAL_USD = 1_000_000.0

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

async def get_order_book_update(market):
    ob = ORDER_BOOKS.get(market)
    if not ob: return None
    return {
        "type": "order_book_update",
        "market": market,
        "bids": sorted([[-p, q, a] for p, q, a in ob["bids"]], reverse=True)[:10],
        "asks": sorted([[p, q, a] for p, q, a in ob["asks"]])[:10]
    }

async def match_orders(market):
    if market in RESOLVED_MARKETS: return
    ob = ORDER_BOOKS[market]
    
    while ob["bids"] and ob["asks"] and -ob["bids"][0][0] >= ob["asks"][0][0]:
        bid_price_neg, bid_quantity, buyer_id = heapq.heappop(ob["bids"])
        ask_price, ask_quantity, seller_id = heapq.heappop(ob["asks"])
        
        bid_price = -bid_price_neg
        trade_price = ask_price
        trade_quantity = min(bid_quantity, ask_quantity)
        
        if trade_quantity <= 0: continue

        print(f"[TRADE] {market}: {trade_quantity} units @ ${trade_price}")
        cost = trade_quantity * trade_price
        
        ACCOUNTS[buyer_id]["USD"] -= cost
        ACCOUNTS[buyer_id][market] += trade_quantity
        ACCOUNTS[seller_id]["USD"] += cost
        ACCOUNTS[seller_id][market] -= trade_quantity
        
        trade_message = {
            "type": "trade_executed", "market": market, "quantity": trade_quantity,
            "price": trade_price, "buyer": buyer_id, "seller": seller_id
        }

        print(get_exchange_state_string())
        await broadcast_to_traders(trade_message)
        
        await send_to_trader(buyer_id, await get_account_update(buyer_id))
        await send_to_trader(seller_id, await get_account_update(seller_id))
        
        if bid_quantity > trade_quantity:
            heapq.heappush(ob["bids"], (-bid_price, bid_quantity - trade_quantity, buyer_id))
        if ask_quantity > trade_quantity:
            heapq.heappush(ob["asks"], (ask_price, ask_quantity - trade_quantity, seller_id))

    await broadcast_to_traders(await get_order_book_update(market))

async def trader_client_handler(websocket):
    """Handles new traders connecting to *this* exchange server."""
    agent_id = None
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
            
            elif not agent_id:
                await websocket.send(json.dumps({"type": "error", "message": "You must register first"}))
                continue

            elif action == "place_order":
                await handle_place_order(agent_id, data)

    except websockets.exceptions.ConnectionClosed:
        print(f"[TRADER DISCONNECT] Client {agent_id} disconnected.")
    finally:
        if agent_id in CONNECTED_CLIENTS:
            del CONNECTED_CLIENTS[agent_id]

async def handle_place_order(agent_id, data):
    """Processes a 'place_order' request from a trader."""
    market = data.get("market")
    if market in RESOLVED_MARKETS:
        return await send_to_trader(agent_id, {"type": "error", "message": f"Market {market} is resolved."})
        
    side = data.get("side")
    try:
        price = float(data.get("price"))
        quantity = float(data.get("quantity"))
    except Exception:
        return await send_to_trader(agent_id, {"type": "error", "message": "Invalid price or quantity"})

    if market not in ORDER_BOOKS:
        return await send_to_trader(agent_id, {"type": "error", "message": "Market does not exist"})
    if side not in ["buy", "sell"] or price <= 0 or quantity <= 0:
        return await send_to_trader(agent_id, {"type": "error", "message": "Invalid order parameters"})

    if side == "buy":
        cost = price * quantity
        if ACCOUNTS[agent_id]["USD"] < cost:
            return await send_to_trader(agent_id, {"type": "error", "message": "Insufficient USD"})
    else: # "sell"
        if ACCOUNTS[agent_id].get(market, 0) < quantity:
            return await send_to_trader(agent_id, {"type": "error", "message": f"Insufficient {market} tokens"})

    ob = ORDER_BOOKS[market]
    if side == "buy":
        heapq.heappush(ob["bids"], (-price, quantity, agent_id))
        print(f"[TRADER ORDER] {agent_id} places BUY {quantity} {market} @ ${price}")
    else: # "sell"
        heapq.heappush(ob["asks"], (price, quantity, agent_id))
        print(f"[TRADER ORDER] {agent_id} places SELL {quantity} {market} @ ${price}")
        
    await broadcast_to_traders(await get_order_book_update(market))
    await match_orders(market)

# --- AI Client Logic (Listens to your script) ---

async def internal_handle_create(data: dict):
    """Processes a CREATE action from the AI."""
    market_name = data.get("market_name")
    prob_yes = data.get("probability", 0.5)
    prob_no = round(1.0 - prob_yes, 2)
    
    market_yes = f"{market_name}_YES"
    market_no = f"{market_name}_NO"

    if market_yes in ORDER_BOOKS:
        print(f"[AI] Received CREATE for existing market {market_name}. Ignoring.")
        return

    print(f"\n[AI ACTION] Received CREATE: {market_name} (Prob={prob_yes})")
    
    # 1. "Mint" tokens into the AI's account
    ACCOUNTS[MAKER_AGENT_ID][market_yes] = MAKER_INITIAL_TOKENS
    ACCOUNTS[MAKER_AGENT_ID][market_no] = MAKER_INITIAL_TOKENS
    print(f"  > Minted {MAKER_INITIAL_TOKENS} of {market_yes} and {market_no} for {MAKER_AGENT_ID}")

    # 2. Create the order books
    ORDER_BOOKS[market_yes] = {"bids": [], "asks": []}
    ORDER_BOOKS[market_no] = {"bids": [], "asks": []}
    
    # 3. Place the AI's initial SELL orders
    heapq.heappush(ORDER_BOOKS[market_yes]["asks"], (prob_yes, MAKER_INITIAL_TOKENS, MAKER_AGENT_ID))
    print(f"  > Placed initial SELL for {market_yes} @ ${prob_yes}")
    heapq.heappush(ORDER_BOOKS[market_no]["asks"], (prob_no, MAKER_INITIAL_TOKENS, MAKER_AGENT_ID))
    print(f"  > Placed initial SELL for {market_no} @ ${prob_no}")
    
    # 4. Notify all connected *traders*
    await broadcast_to_traders({
        "type": "new_market",
        "market_name": market_name,
        "market_yes": market_yes,
        "market_no": market_no,
        "created_by": MAKER_AGENT_ID
    })
    await broadcast_to_traders(await get_order_book_update(market_yes))
    await broadcast_to_traders(await get_order_book_update(market_no))

async def internal_handle_resolve(data: dict):
    """Processes a RESOLVE action from the AI."""
    market_name = data.get("market_name")
    outcome = data.get("outcome", "YES").upper()
    
    market_yes = f"{market_name}_YES"
    market_no = f"{market_name}_NO"

    if market_yes in RESOLVED_MARKETS:
        print(f"[AI] Received RESOLVE for already resolved market {market_name}. Ignoring.")
        return

    print(f"\n[AI ACTION] Received RESOLVE: {market_name} -> {outcome}")

    # 1. Stop trading
    RESOLVED_MARKETS.add(market_yes)
    RESOLVED_MARKETS.add(market_no)
    
    # 2. Clear order books
    ORDER_BOOKS[market_yes] = {"bids": [], "asks": []}
    ORDER_BOOKS[market_no] = {"bids": [], "asks": []}
    
    winning_token = market_yes if outcome == "YES" else market_no
    losing_token = market_no if outcome == "YES" else market_yes

    # 3. Settle accounts
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
                
    # 4. Notify all connected *traders*
    await broadcast_to_traders({
        "type": "market_resolved",
        "market_name": market_name,
        "outcome": outcome,
        "winning_token": winning_token
    })

    # 5. Send final account updates
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
    """Return a nicely formatted string of the current exchange state, excluding resolved markets."""
    output = []
    output.append("\n=== 📊 EXCHANGE STATE ===")

    # --- Balances ---
    output.append("\n--- ACCOUNTS ---")
    for agent, balances in ACCOUNTS.items():
        nonzero_assets = {a: amt for a, amt in balances.items() if amt != 0}
        if not nonzero_assets:
            continue
        output.append(f" {agent}:")
        for asset, amount in nonzero_assets.items():
            output.append(f"    {asset:35} {amount:10.3f}")

    # --- Order Books ---
    output.append("\n--- ORDER BOOKS ---")
    for market, book in ORDER_BOOKS.items():
        if market in RESOLVED_MARKETS:
            continue  # Skip resolved markets

        bids = sorted([[-p, q, a] for p, q, a in book["bids"]], reverse=True)
        asks = sorted([[p, q, a] for p, q, a in book["asks"]])

        output.append(f"\n📈 {market}")
        if not bids and not asks:
            output.append("   (empty)")
            continue

        if bids:
            output.append("   BIDS:")
            for price, qty, aid in bids[:5]:
                output.append(f"     {aid:15} {qty:8.3f} @ ${price:6.3f}")
        else:
            output.append("   (no bids)")

        if asks:
            output.append("   ASKS:")
            for price, qty, aid in asks[:5]:
                output.append(f"     {aid:15} {qty:8.3f} @ ${price:6.3f}")
        else:
            output.append("   (no asks)")

    output.append("==========================\n")
    return "\n".join(output)

# --- Main Startup ---

async def main():
    print(f"--- Exchange Server starting... ---")
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





