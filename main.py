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
        """Configure logging"""
        level = getattr(logging, self.config.get("logging.level", "INFO"))
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/app.log")
            ]
        )
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        self.logger.info("Received shutdown signal, stopping...")
        self.running = False
    
    def process_audio_window(self, wav32: np.ndarray, wav16: np.ndarray):
        """Process a single audio window through the complete pipeline"""
        try:
            # Run PANNs inference once and get scores
            clipwise, _ = self.classifier.model.inference(wav32[None, :])
            # Handle both torch tensor and numpy array returns
            if hasattr(clipwise, 'cpu'):
                scores = clipwise.squeeze(0).cpu().numpy()
            else:
                scores = clipwise.squeeze(0) if hasattr(clipwise, 'squeeze') else clipwise[0]
            
            # Get top classification
            top_class_idx = int(scores.argmax())
            classification = self.classifier.class_names[top_class_idx]
            confidence = float(scores[top_class_idx])
            
            # Print top classification (like in original script)
            print(f"[PANNs] {classification} ({confidence:.3f})")
            
            # Check for abnormal sounds
            abnormal_tag, abnormal_prob = None, 0.0
            for tag, indices in self.classifier.abnormal_indices.items():
                if indices:
                    prob = scores[indices].max()
                    if prob > abnormal_prob:
                        abnormal_prob, abnormal_tag = prob, tag
            
            is_abnormal = abnormal_prob > self.classifier.abnormal_threshold
            
            if is_abnormal and abnormal_tag:
                import time
                print(f"🚨 {time.strftime('%H:%M:%S')}  ABNORMAL "
                      f"{abnormal_tag.upper()} ({abnormal_prob:.2f})")
                self.event_logger.log_anomaly(abnormal_tag, abnormal_prob)
            
            # Check if this is speech
            if self.classifier.is_speech(scores):
                # Get buffered audio (includes context before speech detection)
                buffered_audio = self.audio_processor.get_buffered_audio()
                
                if len(buffered_audio) > 0:
                    # Use buffered audio for Vosk STT (includes pre-speech context)
                    transcript = self.classifier.transcribe_speech(buffered_audio)
                    if transcript:
                        print(f"🗣 Speech: {transcript}")
                        sentiment = self.classifier.analyze_sentiment(transcript)
                        self.event_logger.log_speech(transcript, sentiment)
                        
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
        """Run the live audio analysis pipeline"""
        print("🔴 Live demo (Ctrl-C to quit)…")
        self.logger.info("Starting live audio analysis...")
        
        try:
            # Start audio stream
            self.audio_processor.start_stream()
            
            # Process audio windows
            for wav32, wav16 in self.audio_processor.get_audio_windows():
                if not self.running:
                    break
                
                self.process_audio_window(wav32, wav16)
                
        except KeyboardInterrupt:
            print("\nStopping…")
            self.running = False
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up resources...")
        self.audio_processor.stop_stream()
        self.event_logger.close()

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