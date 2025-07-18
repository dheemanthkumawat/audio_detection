import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import io
import soundfile as sf

logger = logging.getLogger(__name__)

class SpeakerAnalyzer:
    """Speaker diarization using pyannote.audio"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.get("speaker_analysis.enabled", True)
        self.model_name = config.get("speaker_analysis.model", "pyannote/speaker-diarization-3.1")
        
        self.pipeline = None
        if self.enabled:
            self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the pyannote speaker diarization pipeline"""
        try:
            logger.info(f"Loading speaker diarization pipeline: {self.model_name}")
            
            # Try to get HuggingFace token from environment
            import os
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            if hf_token:
                logger.info("Using HuggingFace authentication token")
                self.pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=hf_token)
            else:
                logger.warning("No HuggingFace token found, trying without authentication")
                self.pipeline = Pipeline.from_pretrained(self.model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("Speaker diarization pipeline loaded on GPU")
            else:
                logger.info("Speaker diarization pipeline loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load speaker diarization pipeline: {e}")
            logger.info("Speaker analysis will be disabled")
            logger.info("To enable speaker analysis:")
            logger.info("1. Get HF token: https://hf.co/settings/tokens")
            logger.info("2. Accept terms: https://hf.co/pyannote/speaker-diarization-3.1")
            logger.info("3. Set environment variable: export HUGGINGFACE_TOKEN=your_token")
            self.enabled = False
    
    def analyze_session(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> Optional[Dict]:
        """
        Analyze a speech session for speaker diarization
        
        Args:
            audio_buffer: Audio data as numpy array
            sample_rate: Sample rate of audio (default 16kHz)
            
        Returns:
            Dict containing speaker segments or None if disabled/failed
        """
        if not self.enabled or self.pipeline is None:
            return None
        
        if len(audio_buffer) == 0:
            return None
        
        try:
            # Convert numpy array to audio format that pyannote expects
            audio_file = self._numpy_to_audio_file(audio_buffer, sample_rate)
            
            # Run diarization
            diarization = self.pipeline(audio_file)
            
            # Convert results to our format
            return self._format_diarization_results(diarization)
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            return None
    
    def _numpy_to_audio_file(self, audio_buffer: np.ndarray, sample_rate: int):
        """Convert numpy array to audio file format for pyannote"""
        # Ensure audio is float32 and in correct range
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_buffer)) > 1.0:
            audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
        
        # Create in-memory audio file
        audio_io = io.BytesIO()
        sf.write(audio_io, audio_buffer, sample_rate, format='WAV')
        audio_io.seek(0)
        
        return {"uri": "session_audio", "audio": audio_io}
    
    def _format_diarization_results(self, diarization: Annotation) -> Dict:
        """Format pyannote diarization results to our data structure"""
        speakers = []
        speaker_stats = {}
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_info = {
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.end - segment.start)
            }
            speakers.append(speaker_info)
            
            # Track speaker statistics
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0.0,
                    "segments": 0
                }
            
            speaker_stats[speaker]["total_duration"] += speaker_info["duration"]
            speaker_stats[speaker]["segments"] += 1
        
        # Sort speakers by start time
        speakers.sort(key=lambda x: x["start"])
        
        return {
            "speakers": speakers,
            "speaker_stats": speaker_stats,
            "total_speakers": len(speaker_stats),
            "dominant_speaker": self._get_dominant_speaker(speaker_stats)
        }
    
    def _get_dominant_speaker(self, speaker_stats: Dict) -> Optional[str]:
        """Get the speaker who talked the most"""
        if not speaker_stats:
            return None
        
        return max(speaker_stats.keys(), 
                  key=lambda x: speaker_stats[x]["total_duration"])
    
    def assign_text_to_speakers(self, speaker_data: Dict, transcript: str, 
                               session_start: float, session_duration: float) -> List[Dict]:
        """
        Assign transcript text to speaker segments
        
        Args:
            speaker_data: Result from analyze_session
            transcript: Full transcript text
            session_start: When the session started
            session_duration: Total session duration
            
        Returns:
            List of speaker segments with assigned text
        """
        if not speaker_data or not speaker_data.get("speakers"):
            return [{
                "speaker": "UNKNOWN",
                "start": 0.0,
                "end": session_duration,
                "text": transcript,
                "duration": session_duration
            }]
        
        speakers = speaker_data["speakers"]
        
        # Simple text assignment - split transcript roughly by time proportions
        # This is a basic implementation - more sophisticated text alignment would be better
        segments_with_text = []
        
        if len(speakers) == 1:
            # Single speaker - assign all text
            speaker = speakers[0]
            segments_with_text.append({
                "speaker": speaker["speaker"],
                "start": speaker["start"],
                "end": speaker["end"],
                "text": transcript,
                "duration": speaker["duration"]
            })
        else:
            # Multiple speakers - split text proportionally
            total_duration = sum(s["duration"] for s in speakers)
            words = transcript.split()
            
            word_idx = 0
            for speaker in speakers:
                # Calculate proportion of text for this speaker
                proportion = speaker["duration"] / total_duration if total_duration > 0 else 0
                words_for_speaker = int(len(words) * proportion)
                
                # Assign words to this speaker
                speaker_words = words[word_idx:word_idx + words_for_speaker]
                word_idx += words_for_speaker
                
                segments_with_text.append({
                    "speaker": speaker["speaker"],
                    "start": speaker["start"],
                    "end": speaker["end"],
                    "text": " ".join(speaker_words),
                    "duration": speaker["duration"]
                })
            
            # Assign remaining words to last speaker
            if word_idx < len(words):
                remaining_words = words[word_idx:]
                if segments_with_text:
                    segments_with_text[-1]["text"] += " " + " ".join(remaining_words)
        
        return segments_with_text