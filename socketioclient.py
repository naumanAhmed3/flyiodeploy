import asyncio
import websockets
import json
import time

# WebSocket URL (Update if running on a different host)
WEBSOCKET_URL = "ws://0.0.0.0:8000/ws/client129?version=v1.1"

async def websocket_client():
    last_query = None
    last_message_id = None

    while True:  # Infinite loop for reconnection
        try:
            print(f"🌐 Connecting to {WEBSOCKET_URL}")

            async with websockets.connect(
                WEBSOCKET_URL, ping_interval=30, ping_timeout=600
            ) as websocket:
                print("✅ Connected to the server!")

                # If there was an unsent query, resend it after reconnection
                if last_query and last_message_id:
                    print(f"🔄 Resending query (ID: {last_message_id}) after reconnection...")
                    await websocket.send(json.dumps(last_query))

                    # Wait for the response to the re-sent query
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)

                        # Ignore keep-alive pings
                        if 'ping' in data:
                            continue

                        print(f"📩 Server response: {data}")

                        # ✅ Check if the server response contains a JSON string inside "data"
                        if "data" in data:
                            try:
                                nested_data = json.loads(data["data"]) if isinstance(data["data"], str) else data["data"]
                                if "responses" in nested_data:
                                    last_query = None
                                    last_message_id = None
                                    break  # Exit loop when response is complete
                            except json.JSONDecodeError:
                                print("⚠️ Warning: Received data is not valid JSON, continuing...")

                # Start normal query loop
                while True:
                    ask_user = input("\n💬 ENTER QUERY (type 'exit' to quit): ")
                    if ask_user.lower() == 'exit':
                        print("👋 Exiting chat. Goodbye!")
                        return

                    last_message_id = f"msg_{int(time.time())}"
                    last_query = {
                        "message_id": last_message_id,
                        "input": ask_user
                    }

                    await websocket.send(json.dumps(last_query))
                    print(f"📤 Sent query (ID: {last_message_id}):", last_query)

                    # ✅ Wait for the server response to current query
                    while True:
                        response = await websocket.recv()
                        data = json.loads(response)

                        if 'ping' in data:
                            continue  # Skip keep-alive pings

                        print(f"📩 Server response: {json.dumps(data)}")

                        # ✅ Check nested JSON inside "data"
                        if "data" in data:
                            try:
                                nested_data = json.loads(data["data"]) if isinstance(data["data"], str) else data["data"]
                                if "responses" in nested_data:
                                    last_query = None
                                    last_message_id = None
                                    break  # Exit loop when final response received
                            except json.JSONDecodeError:
                                print("⚠️ Warning: Received malformed JSON, continuing...")

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"❌ Connection closed by server: {e}")
        except websockets.exceptions.InvalidURI as e:
            print(f"❌ Invalid WebSocket URL: {str(e)}")
        except websockets.exceptions.InvalidHandshake as e:
            print(f"❌ Handshake failed: {str(e)}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ Connection closed unexpectedly: {str(e)}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

        print("🔄 Attempting to reconnect in 5 seconds...")
        await asyncio.sleep(5)  # Wait before reconnecting

# Start the WebSocket client
asyncio.run(websocket_client())
