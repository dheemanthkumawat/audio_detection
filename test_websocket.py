#!/usr/bin/env python3
"""
Test WebSocket connection and message sending
"""
import asyncio
import websockets
import json
import time

async def test_websocket():
    """Test WebSocket connection"""
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server")
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"📨 Received: {data['type']} - {data}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔌 Testing WebSocket connection...")
    asyncio.run(test_websocket())