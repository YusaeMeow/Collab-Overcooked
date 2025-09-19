import asyncio
import websockets
import websocket
import json
import uuid
import time
import socket

port = ""


def change_port(new_port):
    global port
    port = new_port
    return


# The client side of the storage connection
connected_clients = set()
username_record = {"agent0": None, "agent1": None}


async def handler(websocket, path):
    # Adding a new connected client side
    connected_clients.add(websocket)
    print(f"New client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            data = json.loads(message)
            # print(f"Received message from client: {data['agent']}")
            await broadcast(data)
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    finally:
        connected_clients.remove(websocket)


async def broadcast(data):
    if connected_clients: 
        message = json.dumps(data)
        await asyncio.wait([client.send(message) for client in connected_clients])

async def send_message(data):
    if connected_clients:
        message = json.dumps(data)
        await asyncio.wait([client.send(message) for client in connected_clients])


def output_to_port(
    agent, observation, recipe=None, map=None, error=None, mission="doing"
):
    """
    Connect to the WebSocket server and send a message, wait for a response from the server, and then close the connection.
    Add a unique message_id for each message sent to distinguish between the echo of the message sent by yourself and the actual processing result of the server.
    """
    try:
        uri = f"ws://localhost:{port}"
        ws = websocket.create_connection(uri)
        print(f"Connected to the WebSocket server{uri}")

        message_id = str(uuid.uuid4())
        data = {
            "agent": agent,
            "observation": observation,
            "message_id": message_id,
            "map": map,
            "recipe": recipe,
            "error": error,
            "mission": mission,
        }

        response = None

        while response == None and mission == "doing":
            ws.send(json.dumps(data))

            time.sleep(0.5)
            response = ws.recv()
            response_data = json.loads(response)
            if response_data.get("message_id") == message_id:
                response = None
                continue
            print("Received server response")
            ws.close()
        return response_data

    except websocket.WebSocketException as e:
        print("WebSocket error:", e)
        return None
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        return None


def check_port_in_use(port, host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, int(port)))
        s.settimeout(1)
        s.shutdown(2)
        return True
    except:
        return False


async def listen_to_server():
    uri = "ws://localhost:8999"
    async with websockets.connect(uri) as websocket:
        print("The second process has connected to the server.")
        while True:
            message = await websocket.recv()
            print("The second process has recieved message: ", message)
            yield message


def listen_to_server(uri="ws://localhost:8999"):
    """
    Connect to the WebSocket server and wait for a message.
    Once the message is received, it is returned.

    :param uri: URI of the WebSocket server
    :return: Decoded JSON message
    """
    try:
        ws = websocket.create_connection(uri)
        print("Connected to the WebSocket server")

        message = ws.recv()
        print("Recieved message:", message)

        ws.close()

        data = json.loads(message)
        return data

    except websocket.WebSocketException as e:
        print("WebSocket error:", e)
        return None
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        return None


def start_server(port):
    server = websockets.serve(handler, "0.0.0.0", port)

    asyncio.get_event_loop().run_until_complete(server)
    print(f"WebSocket server started on port {port}")
    asyncio.get_event_loop().run_forever()
