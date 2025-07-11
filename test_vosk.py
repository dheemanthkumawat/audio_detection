#!/usr/bin/env python3
"""
Simple Vosk test script to verify the model is working
"""

import sys
import os
import numpy as np
import json
import wave

# Add src to path
sys.path.insert(0, 'src')

try:
    import vosk
    from src.utils.config import Config
    
    print("Testing Vosk STT...")
    
    # Load config and model
    config = Config()
    model_path = config.get("stt.model_path", "models/vosk-small-en")
    
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)
    
    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, 16000)
    
    print("✅ Model loaded successfully")
    
    # Create test audio (1 second of sine wave at 440Hz - sounds like "beep")
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_float = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    print(f"Created test audio: {len(audio_int16)} samples")
    print(f"Audio stats - Max: {np.max(np.abs(audio_float)):.3f}, Mean: {np.mean(np.abs(audio_float)):.3f}")
    
    # Test Vosk with generated audio
    audio_bytes = audio_int16.tobytes()
    
    print("Testing Vosk recognition...")
    accepted = rec.AcceptWaveform(audio_bytes)
    print(f"AcceptWaveform returned: {accepted}")
    
    if accepted:
        result = json.loads(rec.Result())
        print(f"Result: {result}")
    
    final_result = json.loads(rec.FinalResult())
    print(f"FinalResult: {final_result}")
    
    partial_result = json.loads(rec.PartialResult())
    print(f"PartialResult: {partial_result}")
    
    print("\n🎯 Test with actual speech-like patterns...")
    
    # Test with speech-like noise (random but structured)
    np.random.seed(42)  # Reproducible
    speech_like = np.random.normal(0, 0.1, sample_rate)  # 1 second of noise
    speech_int16 = (speech_like * 32767 * 0.5).astype(np.int16)
    
    rec2 = vosk.KaldiRecognizer(model, 16000)
    accepted2 = rec2.AcceptWaveform(speech_int16.tobytes())
    print(f"Speech-like AcceptWaveform: {accepted2}")
    
    if accepted2:
        result2 = json.loads(rec2.Result())
        print(f"Speech-like Result: {result2}")
    
    final2 = json.loads(rec2.FinalResult())
    print(f"Speech-like FinalResult: {final2}")
    
    print("\n✅ Vosk test completed")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure vosk is installed: pip install vosk")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()