# Live Audio Analysis Pipeline

Real-time audio processing pipeline using PANNs CNN14 for audio classification, Vosk for speech recognition, and intelligent anomaly detection with speech session management.

## Requirements

- **Python 3.9+** (tested on 3.9, 3.10, 3.11)
- **CUDA-capable GPU** (optional, for faster PANNs inference)
- **Microphone** with audio input capabilities

## Quick Start

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Vosk model:**
   ```bash
   mkdir -p models
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -O models/vosk-model.zip
   cd models && unzip vosk-model.zip
   mv vosk-model-small-en-us-0.15 vosk-small-en
   rm vosk-model.zip
   cd ..
   ```

4. **List audio devices (find your microphone):**
   ```bash
   python main.py --list-devices
   ```

5. **Configure audio device (optional):**
   ```bash
   cp .env.example .env
   # Edit .env and set AUDIO_DEVICE_INDEX to your microphone index
   ```

6. **Run with default config:**
   ```bash
   python main.py
   ```

7. **Run with quiet-office profile:**
   ```bash
   python main.py --profile quiet-office
   ```

## Features

- **🎯 Real-time audio classification** using PANNs CNN14 (527 AudioSet classes)
- **🧠 Intelligent speech session management** - prevents bouncing, captures complete conversations
- **🔄 Hybrid transcription approach** - multiple triggers for reliable speech capture
- **📊 5-second rolling buffer** - ensures no speech is lost during processing
- **⚠️ Anomaly detection** for sirens, alarms, glass breaking, screams, gunshots
- **🗣️ Offline speech-to-text** using Vosk (no internet required)
- **😊 Sentiment analysis** with configurable keyword matching
- **💾 Local data storage** - CSV files for analysis, JSON for events
- **📡 MQTT publishing** for real-time event streaming
- **⚙️ Configurable profiles** - default and quiet-office presets

## Configuration

- Default config: `config/default.yaml`
- Profiles: `config/quiet-office.yaml`  
- Environment overrides: `.env` (see `.env.example`)

## Architecture

```
audio_detection/
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
├── logs/                   # Application logs
├── models/                 # Vosk STT models
└── requirements.txt        # Python dependencies
```

## Output

Events are logged in real-time to console and stored locally:

**Console Output:**
```
🔴 Live Audio Pipeline Starting...
   📊 PANNs CNN14 for audio classification
   🗣️  Vosk for speech recognition
   💾 Local storage enabled
   ⏸️  Ctrl-C to quit

[PANNs] Speech (0.847) [Window 45]
🗣 Speech (session_active): hello world this is a complete sentence
🚨 14:23:15 ABNORMAL ALARM (0.92)

📊 Session Statistics:
   • Windows processed: 245
   • Speech sessions: 3  
   • Anomalies detected: 1
   • Today's events: 12
   • Speech transcripts: 8
```

**Local Storage:**
- `data/events.jsonl` - All events in JSON format
- `data/speech_transcripts.csv` - Speech transcriptions with metadata  
- `data/anomalies.csv` - Detected anomalies with classifications
- `data/daily_summary.json` - Daily statistics and summaries

## Troubleshooting

**Common Issues:**

1. **No audio device found:**
   ```bash
   python main.py --list-devices
   # Copy the correct device index to .env file
   ```

2. **Vosk model not found:**
   ```bash
   # Re-download and extract the model
   cd models && wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   ```

3. **MQTT connection failed:**
   ```bash
   # Install and start MQTT broker (optional)
   sudo apt install mosquitto mosquitto-clients
   sudo systemctl start mosquitto
   ```

4. **GPU not detected:**
   ```bash
   # Install CUDA-enabled PyTorch (optional, for faster inference)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.