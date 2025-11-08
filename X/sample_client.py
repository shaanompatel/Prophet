import asyncio
import websockets
import json

SERVER_URI = "ws://localhost:8765"

async def listen_to_tweet_stream():
    """
    Connects to the tweet stream server and processes
    the incoming data (tweets).
    """
    print(f"Agent connecting to {SERVER_URI}...")
    while True:
        try:
            async with websockets.connect(SERVER_URI) as websocket:
                print("Successfully connected! Listening for tweets...")
                
                # This loop runs forever, processing messages as they arrive
                async for message in websocket:
                    
                    # 'message' is the raw JSON string from the server
                    tweet_data = json.loads(message)
                    
                    # This client just prints the received data.
                    user = tweet_data.get('user', {}).get('username', 'unknown')
                    text = tweet_data.get('text', 'No text')
                    
                    print(f"[AGENT LISTENER] Received from @{user}: {text[:70]}...")

        except websockets.exceptions.ConnectionClosed:
            print("Connection lost. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(listen_to_tweet_stream())