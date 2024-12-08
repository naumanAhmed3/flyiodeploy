import socketio

# Create a Socket.IO client
sio = socketio.Client()

# Server URL (use the appropriate IP or hostname if needed)
server_url = 'http://localhost:5000'  # Use Flask server URL here, not Redis

# Version to send (can be v1.0 or v1.1)
CLIENT_VERSION = "v1.1"  # Change this to v1.1 to test different versions

# Connect to the Flask-SocketIO server with version as a query parameter in the URL
def connect_to_server():
    # Manually append version to the server URL
    sio.connect(f"{server_url}?version={CLIENT_VERSION}")

# Event handler when connected to the server
@sio.event
def connect():
    print("Connected to the server!")

# Event handler for receiving responses from the server
@sio.event
def response(data):
    print("Received response from server:", data)

# Event handler when disconnected from the server
@sio.event
def disconnect():
    print("Disconnected from the server.")

# Function to send messages to the server
def send_message(message):
    print("Sending message:", message)
    sio.emit('message', message)

# Wait for user input to send messages
if __name__ == "__main__":
    try:
        connect_to_server()  # Establish connection with version query parameter
        while True:
            message = input("Enter message to send to the server: ")
            if message.lower() == 'exit':
                break
            send_message(message)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sio.disconnect()
