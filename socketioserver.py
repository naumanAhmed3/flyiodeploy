from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import json
from main import executor  # Import LangChain executor
import asyncio
import uuid

ALLOWED_VERSIONS = ["v1.1", "v1.0"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

def is_valid_json(json_string):
    try:
        json.loads(json_string)  # Attempt to parse JSON
        return True  # JSON is valid
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")  # Print the error message
        return False  # JSON is invalid

def handle_executor(input):
    executor_response = executor.invoke({"input": input})['output']
    if is_valid_json(executor_response):
        return executor_response
    else:
        return json.dumps({"response_type": "message", "responses": executor_response}, ensure_ascii=False)

async def keep_alive(websocket):
    """Keep connection alive with periodic pings."""
    while True:
        try:
            await websocket.send_text(json.dumps({"ping": "keep-alive"}))
            await asyncio.sleep(30)  # Ping every 30 seconds
        except Exception as e:
            print("Ping failed, connection lost:", e)
            break

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, version: str = Query(...)):
    """WebSocket endpoint to handle client communication."""
    if version not in ALLOWED_VERSIONS:
        await websocket.close(code=1008)
        return

    if session_id not in sessions:
        sessions[session_id] = {"pending": [], "memory": []}

    await websocket.accept()
    print(f"✅ Client connected: session_id={session_id}, version={version}")

    asyncio.create_task(keep_alive(websocket))

    try:
        for response in sessions[session_id]["pending"]:
            await websocket.send_text(json.dumps(response))
        sessions[session_id]["pending"] = []

        while True:
            try:
                data = await websocket.receive_text()
                data = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
                continue

            if 'input' not in data:
                await websocket.send_text(json.dumps({"error": "Invalid input format."}))
                continue

            user_input = data['input']
            message_id = str(uuid.uuid4())
            await websocket.send_text(json.dumps({"response_type": "processing", "message_id": message_id, "response": "Processing..."}))

            try:
                result = handle_executor(user_input)  # Use handle_executor function
                print("Raw response from handle_executor:", result)
                
                response_payload = {
                    "message_id": message_id,
                    "data": result
                }
                await websocket.send_text(json.dumps(response_payload, ensure_ascii=False))

            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        print(f"❌ Client disconnected: session_id={session_id}")
    except Exception as e:
        print(f"❌ Backend Error: {str(e)}")

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "OK"}
