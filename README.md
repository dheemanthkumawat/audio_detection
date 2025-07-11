# Live Audio Analysis Pipeline

Real-time audio processing pipeline using PANNs CNN14 for audio classification, Vosk for speech recognition, and anomaly detection.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Vosk model:**
   ```bash
   mkdir -p models
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O models/vosk-model.zip
   cd models && unzip vosk-model.zip
   mv vosk-model-small-en-us-0.15 vosk-small-en
   rm vosk-model.zip
   ```

3. **List audio devices (optional):**
   ```bash
   python main.py --list-devices
   ```

4. **Run with default config:**
   ```bash
   python main.py
   ```

5. **Run with quiet-office profile:**
   ```bash
   python main.py --profile quiet-office
   ```

## Features

- **Real-time audio classification** using PANNs CNN14 (527 AudioSet classes)
- **Anomaly detection** for sirens, alarms, glass breaking, screams, gunshots
- **Speech-to-text** using Vosk for offline transcription  
- **Sentiment analysis** with keyword matching
- **MQTT publishing** for event streaming
- **JSON logging** to `logs/events.jsonl`

## Configuration

- Default config: `config/default.yaml`
- Profiles: `config/quiet-office.yaml`  
- Environment overrides: `.env` (see `.env.example`)

## Architecture

- `main.py` - Entry point and pipeline orchestration
- `config.py` - Configuration management with YAML + env overrides
- `audio_processor.py` - Audio capture, resampling, windowing
- `classifier.py` - PANNs classification, Vosk STT, sentiment analysis
- `event_logger.py` - Event logging and MQTT publishing

## Output

Events are logged in real-time to console and `logs/events.jsonl`:
```
[PANNs] Speech (0.847)
🗣 Speech: hello world
🚨 14:23:15  ABNORMAL ALARM (0.92)
```