import logging
import signal
import sys
import time
import numpy as np
from typing import Optional

from .audio.processor import AudioProcessor
from .audio.speech_session import SpeechSessionManager
from .classification.audio_classifier import AudioClassifier
from .events.logger import EventLogger
from .events.storage import LocalEventStorage
from .utils.config import Config
from .web.websocket_server import WebSocketServer

logger = logging.getLogger(__name__)

class LiveAudioPipeline:
    """Main pipeline orchestrating audio processing, classification, and event handling"""
    
    def __init__(self, config_profile: str = None):
        self.config = Config(config_profile)
        self.setup_logging()
        
        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.classifier = AudioClassifier(self.config)
        self.speech_session_manager = SpeechSessionManager(self.config)
        self.event_logger = EventLogger(self.config)
        self.local_storage = LocalEventStorage(self.config)
        
        # Initialize WebSocket server for real-time frontend
        self.websocket_server = WebSocketServer(
            host=self.config.get("websocket.host", "localhost"),
            port=self.config.get("websocket.port", 8765)
        )
        
        # Pipeline state
        self.running = True
        self.total_windows_processed = 0
        self.speech_sessions_detected = 0
        self.anomalies_detected = 0
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Live audio pipeline initialized")
    
    def setup_logging(self):
        """Configure logging with file and console output"""
        level = getattr(logging, self.config.get("logging.level", "INFO"))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # File handler
        file_handler = logging.FileHandler("logs/pipeline.log")
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=[console_handler, file_handler]
        )
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received shutdown signal, stopping pipeline...")
        self.running = False
        
        # Force end any active speech session
        session_action = self.speech_session_manager.force_end_session()
        if session_action == 'end_session':
            self._handle_speech_session_end()
    
    def process_audio_window(self, wav32: np.ndarray, wav16: np.ndarray):
        """Process a single audio window through the complete pipeline"""
        try:
            self.total_windows_processed += 1
            
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
            
            # Print classification (every 10th window to reduce spam)
            if self.total_windows_processed % 10 == 0:
                print(f"[PANNs] {classification} ({confidence:.3f}) [Window {self.total_windows_processed}]")
                # Send to WebSocket frontend
                self.websocket_server.send_audio_classification(
                    classification, confidence, self.total_windows_processed
                )
            
            # Check for abnormal sounds
            self._handle_anomaly_detection(scores, classification, confidence)
            
            # Handle speech session management
            is_speech = self.classifier.is_speech(scores)
            session_action = self.speech_session_manager.process_window(is_speech)
            
            if session_action == 'start_session':
                self._handle_speech_session_start()
                # Try transcription when session starts (hybrid approach)
                # self._try_transcription("session_start")
                
            elif session_action == 'end_session':
                self._handle_speech_session_end()
                
            # Also try transcription during active speech (but limit frequency to prevent bouncing)
            elif is_speech and self.speech_session_manager.state.value == "speech_active":
                # Only try transcription every 10 windows during active speech to prevent spam
                if self.total_windows_processed % 10 == 0:
                    self._try_transcription("session_active")
            
            # Log debug info every 50 windows
            if self.total_windows_processed % 50 == 0:
                session_info = self.speech_session_manager.get_session_info()
                self.logger.debug(f"Session state: {session_info['state']}, "
                                f"Duration: {session_info['session_duration']:.1f}s")
                
        except Exception as e:
            self.logger.error(f"Error processing audio window: {e}")
    
    def _handle_anomaly_detection(self, scores: np.ndarray, classification: str, confidence: float):
        """Handle anomaly detection and logging"""
        abnormal_tag, abnormal_prob = None, 0.0
        
        for tag, indices in self.classifier.abnormal_indices.items():
            if indices:
                prob = scores[indices].max()
                if prob > abnormal_prob:
                    abnormal_prob, abnormal_tag = prob, tag
        
        is_abnormal = abnormal_prob > self.classifier.abnormal_threshold
        
        if is_abnormal and abnormal_tag:
            self.anomalies_detected += 1
            timestamp = time.strftime('%H:%M:%S')
            
            print(f"🚨 {timestamp} ABNORMAL {abnormal_tag.upper()} ({abnormal_prob:.2f})")
            
            # Store anomaly
            self.local_storage.store_anomaly(
                anomaly_type=abnormal_tag,
                classification=classification,
                confidence=abnormal_prob,
                description=f"Detected via PANNs classification"
            )
            
            # Also log to MQTT if configured
            self.event_logger.log_anomaly(abnormal_tag, abnormal_prob)
            
            # Send to WebSocket frontend
            self.websocket_server.send_anomaly_detection(
                abnormal_tag, classification, abnormal_prob
            )
    
    def _handle_speech_session_start(self):
        """Handle start of speech session"""
        self.speech_sessions_detected += 1
        self.logger.info(f"Speech session #{self.speech_sessions_detected} started")
    
    def _try_transcription(self, trigger: str):
        """Try transcription with debugging info"""
        print(f"🔄 Attempting transcription (trigger: {trigger})...")
        
        # Get buffered audio
        buffered_audio = self.audio_processor.get_buffered_audio()
        
        if len(buffered_audio) > 8000:  # At least 0.5 seconds
            print(f"🔄 Transcribing {len(buffered_audio)/16000:.1f}s of buffered audio...")
            
            transcript = self.classifier.transcribe_speech(buffered_audio)
            
            if transcript:
                print(f"🗣 Speech ({trigger}): {transcript}")
                
                # Analyze sentiment
                sentiment = self.classifier.analyze_sentiment(transcript)
                
                # Calculate session duration
                session_info = self.speech_session_manager.get_session_info()
                session_duration = session_info.get('session_duration', 0)
                
                # Store speech locally
                self.local_storage.store_speech(
                    transcript=transcript,
                    sentiment=sentiment,
                    confidence=0.8,
                    duration=session_duration
                )
                
                # Also log to MQTT if configured
                self.event_logger.log_speech(transcript, sentiment)
                
                # Send to WebSocket frontend
                self.websocket_server.send_speech_detection(
                    transcript, confidence=0.8, trigger=trigger
                )
                
                # Check for negative sentiment as anomaly
                if sentiment["sentiment"] in ["negative", "mixed"]:
                    self.local_storage.store_anomaly(
                        anomaly_type="negative_speech",
                        classification="Speech",
                        confidence=0.8,
                        description=f"Negative sentiment detected: {transcript[:50]}..."
                    )
                    
                    self.event_logger.log_anomaly(
                        "negative_speech", 
                        0.8,
                        transcript,
                        sentiment
                    )
                
                return True  # Success
            else:
                print(f"❌ No transcript from {trigger} attempt")
                return False
        else:
            print(f"⚠️  Buffer too short for transcription: {len(buffered_audio)} samples")
            return False
    
    def _handle_speech_session_end(self):
        """Handle end of speech session - trigger final transcription"""
        self.logger.info("Speech session ended - final transcription attempt")
        self._try_transcription("session_end")
    
    def run(self):
        """Run the live audio analysis pipeline"""
        print("🔴 Live Audio Pipeline Starting...")
        print("   📊 PANNs CNN14 for audio classification")
        print("   🗣️  Vosk for speech recognition") 
        print("   💾 Local storage enabled")
        print("   ⏸️  Ctrl-C to quit")
        print()
        
        self.logger.info("Starting live audio analysis pipeline")
        
        try:
            # Start WebSocket server
            self.websocket_server.start_in_thread()
            
            # Send initial status to frontend
            self.websocket_server.send_system_status(
                "Pipeline started",
                {"components": ["AudioProcessor", "AudioClassifier", "EventLogger", "LocalStorage"]}
            )
            
            # Start audio stream
            self.audio_processor.start_stream()
            
            # Process audio windows
            for wav32, wav16 in self.audio_processor.get_audio_windows():
                if not self.running:
                    break
                
                self.process_audio_window(wav32, wav16)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping pipeline...")
            self.running = False
        except Exception as e:
            self.logger.error(f"Error in main pipeline loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and save final statistics"""
        self.logger.info("Cleaning up pipeline resources...")
        
        # Stop audio stream
        self.audio_processor.stop_stream()
        
        # Close event logger
        self.event_logger.close()
        
        # Stop WebSocket server
        self.websocket_server.stop()
        
        # Save daily summary
        self.local_storage.save_daily_summary()
        
        # Print final statistics
        print(f"\n📊 Session Statistics:")
        print(f"   • Windows processed: {self.total_windows_processed}")
        print(f"   • Speech sessions: {self.speech_sessions_detected}")
        print(f"   • Anomalies detected: {self.anomalies_detected}")
        
        # Get daily stats
        daily_stats = self.local_storage.get_daily_stats()
        print(f"   • Today's events: {daily_stats['total_events']}")
        print(f"   • Speech transcripts: {daily_stats['speech_count']}")
        
        self.logger.info("Pipeline cleanup completed")
    
    def get_status(self) -> dict:
        """Get current pipeline status"""
        return {
            'running': self.running,
            'windows_processed': self.total_windows_processed,
            'speech_sessions': self.speech_sessions_detected,
            'anomalies_detected': self.anomalies_detected,
            'session_info': self.speech_session_manager.get_session_info()
        }
