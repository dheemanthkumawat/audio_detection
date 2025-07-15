#!/usr/bin/env python3
"""
Live Audio Pipeline with Web Interface
Launch script for real-time audio detection with web frontend
"""
import os
import sys
import time
import webbrowser
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import LiveAudioPipeline

def main():
    """Main entry point for live audio pipeline with web interface"""
    
    print("🎵 Live Audio Pipeline with Web Interface")
    print("=" * 50)
    print()
    print("🔗 Web Interface: http://localhost:8765")
    print("📁 Frontend: web/index.html")
    print("🎧 Audio Input: Auto-detected microphone")
    print()
    print("Components:")
    print("  🧠 PANNs CNN14 for audio classification")
    print("  🗣️  Vosk for speech recognition")
    print("  🚨 Real-time anomaly detection")
    print("  🌐 WebSocket server for live updates")
    print("  💾 Local storage and MQTT logging")
    print()
    
    # Check if websockets is installed
    try:
        import websockets
        print("✅ WebSocket dependency found")
    except ImportError:
        print("❌ WebSocket dependency missing!")
        print("   Install with: pip install websockets>=11.0.0")
        return 1
    
    # Create pipeline
    try:
        pipeline = LiveAudioPipeline()
        
        # Start pipeline
        print("🚀 Starting pipeline...")
        print("   📡 WebSocket server will start on ws://localhost:8765")
        print("   🌐 Open web/index.html in your browser")
        print()
        
        # Optional: Auto-open browser
        web_path = Path(__file__).parent / "web" / "index.html"
        if web_path.exists():
            print(f"🌍 Opening browser: {web_path}")
            webbrowser.open(f"file://{web_path.absolute()}")
        
        # Start the pipeline
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())