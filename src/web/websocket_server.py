import asyncio
import json
import logging
import websockets
from typing import Set, Dict, Any
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.message_queue = queue.Queue()
        self.server = None
        self.loop = None
        self.running = False
        
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial status
        await websocket.send(json.dumps({
            "type": "status",
            "data": {
                "message": "Connected to Live Audio Pipeline",
                "timestamp": datetime.now().isoformat(),
                "clients": len(self.clients)
            }
        }))
        
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str = None):
        """Handle individual client connections"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                # Handle incoming messages from clients if needed
                try:
                    data = json.loads(message)
                    logger.debug(f"Received message from client: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
        finally:
            await self.unregister_client(websocket)
            
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            logger.info(f"No clients connected, dropping message: {message['type']}")
            return
            
        # Create JSON message
        json_message = json.dumps(message)
        logger.info(f"Broadcasting message to {len(self.clients)} clients: {message['type']}")
        
        # Send to all clients
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send(json_message)
                logger.info(f"Successfully sent {message['type']} message to client")
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client connection closed while sending {message['type']}")
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
            logger.info(f"Removed disconnected client, {len(self.clients)} clients remaining")
            
    async def process_message_queue(self):
        """Process messages from the queue and broadcast them"""
        while self.running:
            try:
                # Check for new messages with timeout
                try:
                    message = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.message_queue.get(timeout=0.1)
                    )
                    logger.info(f"Processing queued message: {message['type']}")
                    try:
                        await self.broadcast_message(message)
                    except Exception as broadcast_error:
                        logger.error(f"Error broadcasting {message['type']} message: {broadcast_error}")
                except Exception as e:
                    # No message or timeout - continue
                    if "Empty" not in str(e):
                        logger.debug(f"Queue get timeout or error: {e}")
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(0.1)
                
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Start message processing task
        asyncio.create_task(self.process_message_queue())
        
        # Wait for server to close
        await self.server.wait_closed()
        
    def start_in_thread(self):
        """Start server in a separate thread"""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info(f"Starting WebSocket server thread on {self.host}:{self.port}")
            self.loop.run_until_complete(self.start_server())
            
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info("WebSocket server thread started")
        return thread
        
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self.server:
            self.server.close()
            
    def send_speech_detection(self, transcript: str, confidence: float = 0.0, trigger: str = ""):
        """Send speech detection to frontend"""
        message = {
            "type": "speech",
            "data": {
                "transcript": transcript,
                "confidence": confidence,
                "trigger": trigger,
                "timestamp": datetime.now().isoformat()
            }
        }
        logger.info(f"Queuing speech detection: {transcript[:50]}...")
        self.message_queue.put(message)
        
    def send_anomaly_detection(self, anomaly_type: str, classification: str, confidence: float):
        """Send anomaly detection to frontend"""
        # Convert numpy types to native Python types for JSON serialization
        confidence = float(confidence) if hasattr(confidence, 'dtype') else confidence
        
        message = {
            "type": "anomaly",
            "data": {
                "anomaly_type": anomaly_type,
                "classification": classification,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        }
        logger.info(f"Queuing anomaly detection: {anomaly_type} - {classification} ({confidence:.2f})")
        logger.info(f"Queue size before put: {self.message_queue.qsize()}")
        self.message_queue.put(message)
        logger.info(f"Queue size after put: {self.message_queue.qsize()}")
        
    def send_audio_classification(self, classification: str, confidence: float, window: int):
        """Send audio classification to frontend"""
        # Convert numpy types to native Python types for JSON serialization
        confidence = float(confidence) if hasattr(confidence, 'dtype') else confidence
        
        message = {
            "type": "classification",
            "data": {
                "classification": classification,
                "confidence": confidence,
                "window": window,
                "timestamp": datetime.now().isoformat()
            }
        }
        self.message_queue.put(message)
        
    def send_system_status(self, status: str, details: Dict[str, Any] = None):
        """Send system status to frontend"""
        message = {
            "type": "status",
            "data": {
                "status": status,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        }
        self.message_queue.put(message)