# Live Audio Analysis Pipeline

Real-time audio processing pipeline using PANNs CNN14 for audio classification, Vosk for speech recognition, and anomaly detection.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Vosk model ( Skip if using Whisper model ):**
   ```bash
   mkdir -p models
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O models/vosk-model.zip
   cd models && unzip vosk-model.zip
   mv vosk-model-small-en-us-0.15 vosk-small-en
   rm vosk-model.zip
   ```
   
   **Copy env file:**
   ```bash
   cp .env.example .env
   ```
   
3. **List audio devices (optional):**
   ```bash
   python main.py --list-devices
   ```
   Enter the micprophone AUDIO_DEVICE_INDEX in .env file to the number you get for your connected device

4. **Run with default config:**
   ```bash
   python main_web.py
   ```

5. **Run with quiet-office profile:**
   ```bash
   python main_web.py --profile quiet-office
   ```

## Features

- **Real-time audio classification** using PANNs CNN14 (527 AudioSet classes)
- **Smart speech session management** - prevents bouncing, captures complete conversations
- **Buffered transcription** - 5-second rolling buffer ensures no speech is lost
- **Anomaly detection** for sirens, alarms, glass breaking, screams, gunshots
- **Speech-to-text** using Vosk for offline transcription  
- **Sentiment analysis** with keyword matching
- **Local data storage** - CSV files for analysis, JSON for events
- **MQTT publishing** for event streaming

## Configuration

- Default config: `config/default.yaml`
- Profiles: `config/quiet-office.yaml`  
- Environment overrides: `.env` (see `.env.example`)

## Architecture

```
live_audio/
├── main.py                 # Entry point
├── src/
│   ├── pipeline.py         # Main pipeline orchestration
│   ├── audio/
│   │   ├── processor.py    # Audio capture & buffering
│   │   └── speech_session.py # Speech session management
│   ├── classification/
│   │   └── audio_classifier.py # PANNs + Vosk integration
│   ├── events/
│   │   ├── logger.py       # MQTT/Event logging
│   │   └── storage.py      # Local data storage
│   └── utils/
│       └── config.py       # Configuration management
├── config/                 # YAML configuration files
├── data/                   # Local storage (CSV + JSON)
└── logs/                   # Application logs
```

## Output

Events are logged in real-time to console and stored locally:

**Console Output:**
```
🔴 Live Audio Pipeline Starting...
[PANNs] Speech (0.847) [Window 45]
🗣 Speech: hello world this is a complete sentence
🚨 14:23:15 ABNORMAL ALARM (0.92)
📊 Session Statistics: 245 windows, 3 speech sessions, 1 anomaly
```

**Local Storage:**
- `data/events.jsonl` - All events in JSON format
- `data/speech_transcripts.csv` - Speech transcriptions with metadata
- `data/anomalies.csv` - Detected anomalies with classifications
- `data/daily_summary.json` - Daily statistics and summaries
