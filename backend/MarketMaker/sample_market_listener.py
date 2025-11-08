import asyncio
import websockets
import json

ACTION_SERVER_URI = "ws://localhost:8766"

async def listen_to_action_stream():
    """
    Connects to the ACTION server and listens for
    instructions from the Prophet AI.
    """
    print(f"Action Listener Client connecting to {ACTION_SERVER_URI}...")
    
    while True:
        try:
            async with websockets.connect(ACTION_SERVER_URI) as websocket:
                print("Successfully connected! Listening for actions...")
                
                async for message in websocket:
                    action = json.loads(message)
                    
                    print("\n--- [CLIENT] Received New Action ---")
                    
                    if action.get("action") == "CREATE":
                        print(f"  Action:     CREATE")
                        print(f"  Market:     {action.get('market_name')}")
                        print(f"  Probability: {action.get('probability')}")
                        print("  (This is where the Market Maker would create a pool)")
                        
                    elif action.get("action") == "RESOLVE":
                        print(f"  Action:     RESOLVE")
                        print(f"  Market:     {action.get('market_name')}")
                        print(f"  Outcome:    {action.get('outcome')}")
                        print("  (This is where the on-chain resolution would be triggered)")
                    
                    print(f"\n  Raw JSON: {json.dumps(action)}")
                    print("---------------------------------------------")

        except websockets.exceptions.ConnectionClosed:
            print("Connection lost. Reconnecting in 5 seconds...")
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 5 seconds...")
            
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(listen_to_action_stream())