#!/usr/bin/env python3

import argparse
import logging
import signal
import sys
from typing import Generator
import numpy as np

from config import Config
from audio_processor import AudioProcessor
from classifier import AudioClassifier
from event_logger import EventLogger

class LiveAudioAnalyzer:
    def __init__(self, config_profile: str = None):
        self.config = Config(config_profile)
        self.setup_logging()
        
        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.classifier = AudioClassifier(self.config)
        self.event_logger = EventLogger(self.config)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.get("logging.level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get("logging.file", "logs/pipeline.log")),
                logging.StreamHandler()
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def audio_stream(self) -> Generator[np.ndarray, None, None]:
        """Generate audio windows from the audio processor"""
        for audio_window in self.audio_processor.stream():
            if not self.running:
                break
            yield audio_window

    def process_audio_window(self, audio_window: np.ndarray):
        """Process a single audio window"""
        try:
            # Get PANNs classification
            scores = self.classifier.classify(audio_window)
            
            if scores is not None:
                # Check for speech
                if self.classifier.is_speech(scores):
                    print(f"🗣 Speech detected (confidence: {scores[self.classifier.speech_idx]:.2f})")
                    
                    # Get transcript
                    buffered_audio = self.audio_processor.get_buffered_audio()
                    if buffered_audio is not None:
                        transcript = self.classifier.transcribe(buffered_audio)
                        
                        if transcript:
                            print(f"📝 Transcript: {transcript}")
                            
                            # Analyze sentiment
                            sentiment = self.classifier.analyze_sentiment(transcript)
                            print(f"💭 Sentiment: {sentiment}")
                            
                            # Log the event
                            self.event_logger.log_speech(transcript, sentiment)
                    
                # Check for anomalies
                anomaly_type = self.classifier.detect_anomaly(scores)
                if anomaly_type:
                    confidence = scores[self.classifier.abnormal_indices[anomaly_type][0]]
                    print(f"🚨 Anomaly detected: {anomaly_type} (confidence: {confidence:.2f})")
                    
                    # Log the anomaly
                    self.event_logger.log_anomaly(anomaly_type, confidence)
                    
                    # Get current classification
                    top_class = self.classifier.get_top_classification(scores)
                    if top_class:
                        class_name, class_confidence = top_class
                        print(f"🎵 Classification: {class_name} (confidence: {class_confidence:.2f})")
                        
                        # Check for negative sentiment as anomaly
                        if sentiment["sentiment"] in ["negative", "mixed"]:
                            self.event_logger.log_anomaly(
                                "negative_speech", 
                                scores[self.classifier.speech_idx],
                                transcript,
                                sentiment
                            )
            
        except Exception as e:
            self.logger.error(f"Error processing audio window: {e}")
    
    def run(self):
        """Main processing loop"""
        try:
            print("🎵 Live Audio Analysis Pipeline")
            print("=" * 40)
            print(f"📊 Using profile: {self.config.profile}")
            print(f"🎤 Audio device: {self.audio_processor.device_index}")
            print(f"🔊 Sample rate: {self.audio_processor.device_sr}")
            print(f"⏱️  Window size: {self.audio_processor.window_size}s")
            print(f"📈 Stride: {self.audio_processor.stride}s")
            print("=" * 40)
            print("🔴 Recording... Press Ctrl+C to stop")
            print()
            
            self.logger.info("Starting audio analysis pipeline")
            
            for audio_window in self.audio_stream():
                self.process_audio_window(audio_window)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping pipeline...")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
        finally:
            self.logger.info("Pipeline stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Live Audio Analysis Pipeline")
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
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        import sounddevice as sd
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Run analyzer
    analyzer = LiveAudioAnalyzer(args.profile)
    analyzer.run()

if __name__ == "__main__":
    main()