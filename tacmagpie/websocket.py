"""WebSocket server for broadcasting simulation magnetic field data.

Provides real-time streaming of magnetic sensor readings to connected clients
via WebSocket protocol.

Classes
-------
MagneticDataServer
    Manages WebSocket connections and broadcasts magnetic field data.
"""

import asyncio
import json
from datetime import datetime
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MagneticDataServer:
    """WebSocket server that broadcasts magnetic field simulation data.

    Args:
        host: Server bind address
        port: Server bind port

    Attributes:
        clients: Set of currently connected WebSocket clients
        latest_data: Most recent magnetic field data dict for broadcasting
    """

    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.latest_data = None

    async def handler(self, websocket):
        """Handle a single client WebSocket connection.

        Sends connection confirmation, responds to ping messages,
        and cleans up on disconnect.

        Args:
            websocket: WebSocket connection object
        """
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            await websocket.send(json.dumps({"status": "connected"}))
            async for message in websocket:
                msg = json.loads(message)
                if msg.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")

    def update_data(self, B_deltas):
        """Update magnetic field data from simulator.

        Converts magnetic field delta vectors to Gauss (1e4 scaling)
        and stores as JSON-serializable dict with timestamp.

        Note:
            Axis mapping: simulator (x,y,z) -> output (x=x, y=z, z=y)

        Args:
            B_deltas: List of magnetic field change vectors, each shape=(3,)
        """
        data = {}
        for i, B in enumerate(B_deltas, 1):
            data[f"B{i}x"] = round(float(B[0] * 1e4), 3)
            data[f"B{i}y"] = round(float(B[2] * 1e4), 3)
            data[f"B{i}z"] = round(float(B[1] * 1e4), 3)
        data["timestamp"] = datetime.now().isoformat()
        data["sensor_count"] = len(B_deltas)
        self.latest_data = data

    async def broadcast(self):
        """Broadcast latest data to all connected clients.

        Automatically removes clients that fail to receive.
        """
        if not self.clients or not self.latest_data:
            return

        message = json.dumps(self.latest_data)
        disconnected = []
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.ConnectionClosed:
                disconnected.append(client)
        for client in disconnected:
            self.clients.discard(client)

    async def start(self):
        """Start the WebSocket server (runs indefinitely)."""
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"WebSocket server started: ws://{self.host}:{self.port}")
            await asyncio.Future()
