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
    print("🌐 Web interface will be available at: http://localhost:8765")
    print("📊 Real-time audio detection dashboard")
    print("=" * 50)
    
    # Initialize and run pipeline
    pipeline = LiveAudioPipeline()
    
    try:
        print("🚀 Starting pipeline...")
        
        # Start the pipeline
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🔄 Cleaning up...")
        pipeline.stop()
        print("✅ Pipeline stopped")

if __name__ == "__main__":
    main()