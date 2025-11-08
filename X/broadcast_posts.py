import asyncio
import websockets
import json
import time
import random

# Path to your fake tweet data
DATA_SOURCE = "data/posts.jsonl"

# We will store all connected clients in this set
CONNECTED_CLIENTS = set()

async def stream_tweets():
    """
    Reads the data source line by line and broadcasts
    each tweet to all connected clients.
    """
    print(f"Tweet Streamer is starting... Reading from {DATA_SOURCE}")
    while True: # Loop forever, so you can restart the demo
        try:
            with open(DATA_SOURCE, "r") as f:
                for line in f:
                    if not CONNECTED_CLIENTS:
                        # If no clients are connected, wait for one
                        await asyncio.sleep(1)
                        continue
                    
                    tweet_json = line.strip()
                    if not tweet_json:
                        continue
                    
                    # --- FIXED SECTION ---
                    # Create a list of tasks to send to all clients
                    # This is more robust than the old asyncio.wait
                    tasks = [
                        asyncio.create_task(client.send(tweet_json)) 
                        for client in CONNECTED_CLIENTS
                    ]
                    
                    if tasks:
                        # Use asyncio.gather to run all send tasks concurrently
                        # return_exceptions=True ensures that if one client
                        # disconnects, it doesn't crash the whole server.
                        await asyncio.gather(*tasks, return_exceptions=True)
                    # --- END FIXED SECTION ---
                    
                    # Simulate a real-time, variable delay between tweets
                    await asyncio.sleep(random.uniform(2, 7))
            
            print("Finished streaming all tweets. Looping back to the beginning.")
            await asyncio.sleep(5) # Wait 5s before re-streaming

        except FileNotFoundError:
            print(f"Error: {DATA_SOURCE} not found. Waiting 60s and retrying.")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"An error occurred in stream_tweets: {e}")
            await asyncio.sleep(5)


async def client_handler(websocket):
    """
    Handles a new client connection.
    """
    print(f"Client connected from {websocket.remote_address}")
    CONNECTED_CLIENTS.add(websocket)
    try:
        # Keep the connection open as long as the client is connected
        await websocket.wait_closed()
    finally:
        print(f"Client disconnected from {websocket.remote_address}")
        CONNECTED_CLIENTS.remove(websocket)

async def main():
    # Start the tweet broadcaster as a background task
    asyncio.create_task(stream_tweets())

    # Start the WebSocket server to accept client connections
    server_address = "localhost"
    server_port = 8765
    print(f"Tweet server listening on ws://{server_address}:{server_port}")
    
    async with websockets.serve(client_handler, server_address, server_port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())