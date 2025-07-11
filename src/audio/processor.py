import sounddevice as sd
import numpy as np
import librosa
import queue
from collections import deque
from typing import Optional, Generator, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.device_sr = config.get("audio.device_sample_rate", 44100)
        self.panns_sr = config.get("audio.panns_sample_rate", 32000)
        self.vosk_sr = config.get("audio.vosk_sample_rate", 16000)
        self.window_size = config.get("audio.window_size", 1.0)
        self.stride = config.get("audio.stride", 0.5)
        self.blocksize = config.get("audio.blocksize", 0.1)
        self.buffer_duration = config.get("audio.buffer_duration", 5.0)  # 5 seconds
        
        # Audio device setup
        self.device_index = self._get_device_index()
        
        # Audio queue and sliding window buffer
        self.audio_queue = queue.Queue()
        self.window_buffer = np.zeros((0,), dtype=np.float32)
        
        # Rolling buffer for Vosk (5 seconds at 16kHz)
        self.vosk_buffer_size = int(self.buffer_duration * self.vosk_sr)
        self.vosk_buffer = deque(maxlen=self.vosk_buffer_size)
        
        # Configure sounddevice defaults
        sd.default.device = (self.device_index, None)
        sd.default.samplerate = self.device_sr
        sd.default.channels = 1
        
        logger.info(f"Audio device index: {self.device_index}")
        logger.info(f"Device sample rate: {self.device_sr}")
        logger.info(f"PANNs sample rate: {self.panns_sr}")
        logger.info(f"Vosk sample rate: {self.vosk_sr}")
        logger.info(f"Vosk buffer duration: {self.buffer_duration}s ({self.vosk_buffer_size} samples)")
    
    def _get_device_index(self) -> int:
        devices = sd.query_devices()
        
        # Check for config overrides
        device_index = self.config.get("audio.device_index")
        device_name = self.config.get("audio.device_name")
        
        if device_index is not None:
            logger.info(f"Using configured device index: {device_index}")
            return device_index
        
        if device_name:
            for i, device in enumerate(devices):
                if device_name.lower() in device["name"].lower():
                    logger.info(f"Found device by name: {device['name']} (index: {i})")
                    return i
        
        # Auto-select first device with input channels
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                logger.info(f"Auto-selected device: {device['name']} (index: {i})")
                return i
        
        raise RuntimeError("No suitable audio input device found")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback to fill the queue"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata[:, 0].copy())
    
    def start_stream(self):
        """Start the audio input stream"""
        self.stream = sd.InputStream(
            samplerate=self.device_sr,
            channels=1,
            blocksize=int(self.blocksize * self.device_sr),
            callback=self._audio_callback
        )
        self.stream.start()
        logger.info("Audio stream started")
    
    def stop_stream(self):
        """Stop the audio input stream"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped")
    
    def get_audio_windows(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Get audio windows for PANNs (32kHz) and continuously buffer for Vosk (16kHz)"""
        win_size = int(self.window_size * self.device_sr)
        stride = int(self.stride * self.device_sr)
        
        while True:
            try:
                # Get new audio data
                new_audio = self.audio_queue.get(timeout=0.1)
                self.window_buffer = np.concatenate([self.window_buffer, new_audio])
                
                # Resample new audio to 16kHz and add to rolling buffer
                wav16_chunk = librosa.resample(new_audio, orig_sr=self.device_sr, target_sr=self.vosk_sr)
                self.vosk_buffer.extend(wav16_chunk)
                
                # Check if we have enough data for a PANNs window
                if len(self.window_buffer) < win_size:
                    continue
                
                # Extract window for PANNs
                window = self.window_buffer[:win_size]
                self.window_buffer = self.window_buffer[stride:]
                
                # Resample for PANNs (32kHz)
                wav32 = librosa.resample(window, orig_sr=self.device_sr, target_sr=self.panns_sr)
                
                # Get current 16kHz window (same timing as PANNs window)
                wav16_window = librosa.resample(window, orig_sr=self.device_sr, target_sr=self.vosk_sr)
                
                yield wav32, wav16_window
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                continue
    
    def get_buffered_audio(self) -> np.ndarray:
        """Get the current buffered audio for Vosk transcription"""
        if len(self.vosk_buffer) == 0:
            return np.array([])
        return np.array(self.vosk_buffer)