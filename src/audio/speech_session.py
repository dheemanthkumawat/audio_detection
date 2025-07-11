import time
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class SpeechSessionState(Enum):
    SILENCE = "silence"
    SPEECH_ACTIVE = "speech_active"
    PENDING_END = "pending_end"

class SpeechSessionManager:
    """Manages speech session detection to prevent bouncing transcription calls"""
    
    def __init__(self, config):
        self.config = config
        self.state = SpeechSessionState.SILENCE
        
        # Session parameters
        self.min_session_length = config.get("speech_session.min_length", 1.0)  # seconds
        self.max_silence_gap = config.get("speech_session.max_silence_gap", 1.5)  # seconds
        self.silence_threshold = config.get("speech_session.silence_threshold", 3)  # windows
        
        # Session tracking
        self.session_start_time = None
        self.last_speech_time = None
        self.silence_counter = 0
        
        logger.info(f"Speech session manager initialized:")
        logger.info(f"  Min session length: {self.min_session_length}s")
        logger.info(f"  Max silence gap: {self.max_silence_gap}s")
        logger.info(f"  Silence threshold: {self.silence_threshold} windows")
    
    def process_window(self, is_speech: bool, window_duration: float = 0.5) -> Optional[str]:
        """
        Process a window and return action to take
        
        Args:
            is_speech: Whether current window contains speech
            window_duration: Duration of current window in seconds
            
        Returns:
            'start_session' if speech session started
            'end_session' if speech session ended (trigger transcription)
            None if no action needed
        """
        current_time = time.time()
        
        if is_speech:
            self.last_speech_time = current_time
            self.silence_counter = 0
            
            if self.state == SpeechSessionState.SILENCE:
                # Start new speech session
                self.state = SpeechSessionState.SPEECH_ACTIVE
                self.session_start_time = current_time
                logger.debug("Speech session started")
                return 'start_session'
                
            elif self.state == SpeechSessionState.PENDING_END:
                # Speech resumed, continue session
                self.state = SpeechSessionState.SPEECH_ACTIVE
                logger.debug("Speech session resumed")
                
        else:  # Silence detected
            if self.state == SpeechSessionState.SPEECH_ACTIVE:
                self.silence_counter += 1
                
                # Check if we should transition to pending end
                silence_duration = self.silence_counter * window_duration
                if silence_duration >= self.max_silence_gap:
                    self.state = SpeechSessionState.PENDING_END
                    logger.debug("Speech session pending end due to silence")
                    
            elif self.state == SpeechSessionState.PENDING_END:
                self.silence_counter += 1
                
                # Check if session should end
                if self._should_end_session(current_time):
                    return self._end_session()
        
        return None
    
    def _should_end_session(self, current_time: float) -> bool:
        """Check if current session should end"""
        if self.session_start_time is None:
            return False
            
        # Session must be minimum length
        session_duration = current_time - self.session_start_time
        if session_duration < self.min_session_length:
            return False
            
        # Must have enough silence
        return self.silence_counter >= self.silence_threshold
    
    def _end_session(self) -> str:
        """End current speech session"""
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            logger.info(f"Speech session ended (duration: {session_duration:.1f}s)")
        
        self.state = SpeechSessionState.SILENCE
        self.session_start_time = None
        self.last_speech_time = None
        self.silence_counter = 0
        
        return 'end_session'
    
    def force_end_session(self) -> Optional[str]:
        """Force end current session (e.g., on shutdown)"""
        if self.state != SpeechSessionState.SILENCE:
            return self._end_session()
        return None
    
    def get_session_info(self) -> dict:
        """Get current session information"""
        current_time = time.time()
        info = {
            'state': self.state.value,
            'session_duration': 0,
            'silence_duration': 0
        }
        
        if self.session_start_time:
            info['session_duration'] = current_time - self.session_start_time
            
        if self.last_speech_time:
            info['silence_duration'] = current_time - self.last_speech_time
            
        return info