#!/usr/bin/env python3
"""
Live Audio Analysis Pipeline

Real-time audio processing pipeline using PANNs CNN14 for audio classification,
Vosk for speech recognition, and advanced speech session management.

Usage:
    python main.py                          # Default configuration
    python main.py --profile quiet-office  # Office-optimized settings
    python main.py --list-devices          # List available audio devices
"""

import argparse
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import LiveAudioPipeline

def list_audio_devices():
    """List available audio input devices"""
    try:
        import sounddevice as sd
        print("Available audio devices:")
        print("-" * 50)
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(f"[{i:2d}] {device['name']}")
                print(f"     Channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
                print(f"     Sample rate: {device['default_samplerate']} Hz")
                print()
                
    except ImportError:
        print("Error: sounddevice not installed. Run: pip install sounddevice")
    except Exception as e:
        print(f"Error listing devices: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Live Audio Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default settings
  python main.py --profile quiet-office  # Run with office-optimized settings
  python main.py --list-devices          # List available microphones
  
Features:
  • Real-time audio classification with PANNs CNN14
  • Speech recognition with Vosk (offline)
  • Intelligent speech session management
  • Local data storage (CSV + JSON)
  • MQTT publishing support
  • Configurable anomaly detection
        """
    )
    
    parser.add_argument(
        "--profile", 
        type=str, 
        help="Configuration profile to use (e.g., 'quiet-office')"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Create and run pipeline
    try:
        pipeline = LiveAudioPipeline(args.profile)
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())