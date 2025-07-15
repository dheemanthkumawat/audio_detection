import numpy as np
import torch
import vosk
import json
import logging
from typing import Dict, Tuple, Optional
from panns_inference import AudioTagging, labels as panns_labels
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class AudioClassifier:
    def __init__(self, config):
        self.config = config
        self.backend = config.get("analysis.backend", "panns")
        self.speech_threshold = config.get("analysis.thresholds.speech_threshold", 0.3)
        self.abnormal_threshold = config.get("analysis.thresholds.abnormal_threshold", 0.35)
        
        # Initialize models
        self._init_classifier()
        self._init_stt()
        self._setup_abnormal_detection()
        
    def _init_classifier(self):
        if self.backend == "panns":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading PANNs CNN14 on {device}")
            self.model = AudioTagging(device=device)
            self.class_names = panns_labels
            self.speech_idx = self.class_names.index("Speech")
            logger.info("PANNs CNN14 loaded successfully")
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")
    
    def _setup_abnormal_detection(self):
        """Setup abnormal sound detection based on label groups"""
        abnormal_label_groups = self.config.get("analysis.abnormal_label_groups", {
            "siren": ["Siren", "Civil defense siren"],
            "alarm": ["Alarm", "Smoke detector, smoke alarm", "Car alarm", "Buzzer", "Reversing beeps", "Air horn, truck horn"],
            "glass": ["Breaking", "Glass", "Shatter"],
            "scream": ["Scream", "Yell", "Child shriek", "Screech"],
            "gunshot": ["Gunshot, gunfire"]
        })
        
        self.abnormal_indices = {}
        for tag, labels in abnormal_label_groups.items():
            indices = []
            for label in labels:
                if label in self.class_names:
                    indices.append(self.class_names.index(label))
            self.abnormal_indices[tag] = indices
            
        logger.info(f"Loaded {len(self.abnormal_indices)} abnormal label groups")
    
    def _init_stt(self):
        """Initialize STT model"""
        self.stt_backend = self.config.get("stt.backend", "vosk")
        
        if self.stt_backend == "vosk":
            model_path = self.config.get("stt.model_path", "models/vosk-model-small-en-us-0.15")
            try:
                self.stt_model = vosk.Model(model_path)
                self.stt_recognizer = vosk.KaldiRecognizer(self.stt_model, 16000)
                logger.info("Vosk STT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Vosk model: {e}")
                self.stt_model = None
                
        elif self.stt_backend == "whisper":
            model_size = self.config.get("stt.whisper.model_size", "base")
            device = self.config.get("stt.whisper.device", "auto")
            compute_type = self.config.get("stt.whisper.compute_type", "int8")
            self.whisper_language = self.config.get("stt.whisper.language", "auto")
            self.whisper_task = self.config.get("stt.whisper.task", "transcribe")
            
            try:
                self.whisper_model = WhisperModel(
                    model_size, 
                    device=device, 
                    compute_type=compute_type
                )
                logger.info(f"Faster-Whisper model '{model_size}' loaded successfully on {device}")
                logger.info(f"Language setting: {self.whisper_language}, Task: {self.whisper_task}")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
        else:
            logger.warning(f"Unknown STT backend: {self.stt_backend}")
            self.stt_model = None
    
    def classify_audio(self, audio_data: np.ndarray) -> Tuple[str, float, bool, Optional[str]]:
        """Classify audio and return top prediction, confidence, anomaly flag, and abnormal type"""
        if self.backend == "panns":
            clipwise, _ = self.model.inference(audio_data[None, :])
            scores = clipwise.squeeze(0).cpu().numpy()
            
            top_class_idx = int(scores.argmax())
            top_class = self.class_names[top_class_idx]
            confidence = float(scores[top_class_idx])
            
            # Check for abnormal sounds
            abnormal_tag, abnormal_prob = None, 0.0
            for tag, indices in self.abnormal_indices.items():
                if indices:
                    prob = scores[indices].max()
                    if prob > abnormal_prob:
                        abnormal_prob, abnormal_tag = prob, tag
            
            is_abnormal = abnormal_prob > self.abnormal_threshold
            
            return top_class, confidence, is_abnormal, abnormal_tag if is_abnormal else None
        
        return "Unknown", 0.0, False, None
    
    def transcribe_speech(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe speech using STT"""
        if self.stt_backend == "vosk":
            if not self.stt_model:
                return None
            
            # Convert float32 to int16 for Vosk
            audio_int16 = (audio_data * 32767).astype(np.int16).tobytes()
            
            if self.stt_recognizer.AcceptWaveform(audio_int16):
                # Final result - complete utterance
                result = json.loads(self.stt_recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    return text
            
            # Check partial result for ongoing speech
            partial_result = json.loads(self.stt_recognizer.PartialResult())
            partial_text = partial_result.get("partial", "").strip()
            if partial_text:
                return partial_text
            
            return None
            
        elif self.stt_backend == "whisper":
            if not hasattr(self, 'whisper_model') or not self.whisper_model:
                return None
            
            try:
                # Determine language parameter
                language = None if self.whisper_language == "auto" else self.whisper_language
                
                # Whisper expects audio at 16kHz
                segments, info = self.whisper_model.transcribe(
                    audio_data, 
                    beam_size=5,
                    language=language,
                    task=self.whisper_task
                )
                
                # Log detected language if auto-detection was used
                if self.whisper_language == "auto" and hasattr(info, 'language'):
                    logger.debug(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
                
                # Collect all segments
                text_segments = []
                for segment in segments:
                    text_segments.append(segment.text)
                
                full_text = " ".join(text_segments).strip()
                return full_text if full_text else None
                
            except Exception as e:
                logger.error(f"Error in Whisper transcription: {e}")
                return None
        
        return None
    
    def is_speech(self, scores: np.ndarray) -> bool:
        """Check if audio contains speech based on PANNs scores"""
        return scores[self.speech_idx] > self.speech_threshold
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Simple keyword-based sentiment analysis"""
        if not text:
            return {"sentiment": "neutral", "keywords": []}
        
        text_lower = text.lower()
        negative_keywords = self.config.get("sentiment.keywords.negative", [])
        positive_keywords = self.config.get("sentiment.keywords.positive", [])
        
        found_negative = [kw for kw in negative_keywords if kw in text_lower]
        found_positive = [kw for kw in positive_keywords if kw in text_lower]
        
        if found_negative and not found_positive:
            sentiment = "negative"
        elif found_positive and not found_negative:
            sentiment = "positive"
        elif found_negative and found_positive:
            sentiment = "mixed"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "keywords": found_negative + found_positive
        }
