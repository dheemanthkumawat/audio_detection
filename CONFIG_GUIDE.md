# Audio Pipeline Configuration Guide

## Available Configuration Profiles

The live audio pipeline supports multiple context-aware configuration profiles optimized for different environments and use cases.

## How to Use Different Profiles

### Command Line Usage

```bash
# Use default configuration
python main_new.py

# Use a specific profile
python main_new.py --config-profile quiet-office
python main_new.py --config-profile home-security
python main_new.py --config-profile baby-monitor

# With web interface
python main_web.py --config-profile elderly-care
```

### Programmatic Usage

```python
from src.pipeline import LiveAudioPipeline

# Use specific profile
pipeline = LiveAudioPipeline(config_profile="workshop-safety")
pipeline.run()
```

## Available Profiles

### 1. **default.yaml** - General Purpose
- **Use case**: General audio monitoring
- **Speech threshold**: 0.3 (moderate)
- **Anomaly threshold**: 0.35 (moderate)
- **Best for**: Testing, general monitoring, mixed environments

### 2. **quiet-office.yaml** - Office Environment
- **Use case**: Professional office spaces
- **Speech threshold**: 0.2 (sensitive)
- **Anomaly threshold**: 0.15 (very sensitive)
- **Features**: 
  - Longer buffer for better transcription
  - Sensitive to disruptions
  - Debug logging enabled
- **Best for**: Meeting rooms, open offices, professional spaces

### 3. **home-security.yaml** - Home Security
- **Use case**: Residential security monitoring
- **Speech threshold**: 0.25 (moderate)
- **Anomaly threshold**: 0.20 (sensitive)
- **Features**:
  - Break-in detection (glass, doors, footsteps)
  - Emergency sound recognition
  - Security-focused storage (90 days retention)
- **Best for**: Home security systems, break-in detection, emergency monitoring

### 4. **baby-monitor.yaml** - Baby Monitoring
- **Use case**: Infant and child monitoring
- **Speech threshold**: 0.15 (very sensitive)
- **Anomaly threshold**: 0.25 (moderate)
- **Features**:
  - Crying and distress detection
  - Safety concern monitoring
  - Quick response (0.3s minimum detection)
  - Remote access enabled
- **Best for**: Nurseries, baby rooms, childcare monitoring

### 5. **elderly-care.yaml** - Elderly Care
- **Use case**: Senior citizen monitoring and care
- **Speech threshold**: 0.30 (clear speech)
- **Anomaly threshold**: 0.18 (sensitive)
- **Features**:
  - Fall detection
  - Medical emergency recognition
  - Patient speech patterns (longer pauses)
  - Health data retention (180 days)
  - MQTT integration for healthcare systems
- **Best for**: Assisted living, senior care, medical monitoring

### 6. **workshop-safety.yaml** - Industrial/Workshop Safety
- **Use case**: Workshop and industrial environments
- **Speech threshold**: 0.40 (high - cuts through machine noise)
- **Anomaly threshold**: 0.30 (clear events only)
- **Features**:
  - Equipment failure detection
  - Workplace accident monitoring
  - Safety compliance logging (365 days)
  - Integration with safety systems
- **Best for**: Workshops, factories, industrial sites, construction

### 7. **classroom-monitor.yaml** - Educational Monitoring
- **Use case**: Classroom and educational environments
- **Speech threshold**: 0.20 (sensitive to student voices)
- **Anomaly threshold**: 0.35 (moderate - classrooms are noisy)
- **Features**:
  - Student safety monitoring
  - Disruption detection
  - Privacy-focused (local only, short retention)
  - Emergency alert recognition
- **Best for**: Schools, classrooms, educational facilities

### 8. **outdoor-surveillance.yaml** - Outdoor Security
- **Use case**: Perimeter and outdoor monitoring
- **Speech threshold**: 0.35 (high - outdoor noise)
- **Anomaly threshold**: 0.25 (moderate)
- **Features**:
  - Vehicle and intrusion detection
  - Perimeter security monitoring
  - Weather-resistant audio processing
  - Remote monitoring enabled
- **Best for**: Parking lots, building perimeters, outdoor security

## Customizing Configurations

### Key Configuration Sections

#### 1. **Audio Settings**
```yaml
audio:
  buffer_duration: 5.0      # Buffer length for speech recognition
  device_index: null        # Microphone device (null = auto-detect)
```

#### 2. **Detection Thresholds**
```yaml
analysis:
  thresholds:
    speech_threshold: 0.3   # Lower = more sensitive to speech
    abnormal_threshold: 0.35 # Lower = more sensitive to anomalies
```

#### 3. **Anomaly Detection Groups**
```yaml
analysis:
  abnormal_label_groups:
    custom_group: ["Sound1", "Sound2", "Sound3"]
```

#### 4. **Speech Session Management**
```yaml
speech_session:
  min_length: 1.0          # Minimum session duration
  max_silence_gap: 1.5     # Max silence before ending session
  silence_threshold: 3     # Silent windows before session end
```

#### 5. **Storage and Logging**
```yaml
storage:
  base_dir: "data"         # Data storage directory
  cleanup_days: 30         # Data retention period

logging:
  level: "INFO"            # DEBUG, INFO, WARNING, ERROR
```

#### 6. **WebSocket Server**
```yaml
websocket:
  enabled: true
  host: "localhost"        # "0.0.0.0" for remote access
  port: 8765
```

## Creating Custom Profiles

1. **Copy an existing profile**:
   ```bash
   cp config/default.yaml config/my-custom.yaml
   ```

2. **Modify the settings** for your specific use case

3. **Use your custom profile**:
   ```bash
   python main_new.py --config-profile my-custom
   ```

## Profile Selection Guidelines

| Environment | Recommended Profile | Key Features |
|-------------|-------------------|--------------|
| Office/Meeting Room | `quiet-office` | Sensitive speech detection, disruption alerts |
| Home Security | `home-security` | Break-in detection, emergency alerts |
| Baby/Child Care | `baby-monitor` | Crying detection, quick response |
| Senior Care | `elderly-care` | Fall detection, medical emergencies |
| Workshop/Factory | `workshop-safety` | Equipment monitoring, workplace safety |
| School/Classroom | `classroom-monitor` | Student safety, privacy-focused |
| Outdoor Security | `outdoor-surveillance` | Perimeter monitoring, vehicle detection |
| General Purpose | `default` | Balanced settings for mixed use |

## Troubleshooting

### Common Issues

1. **Too many false alarms**: Increase thresholds
2. **Missing detections**: Decrease thresholds  
3. **Poor speech recognition**: Increase `buffer_duration`
4. **Choppy speech sessions**: Adjust `max_silence_gap`

### Performance Tuning

- **Quiet environments**: Lower thresholds (0.1-0.2)
- **Noisy environments**: Higher thresholds (0.4-0.5)
- **Real-time needs**: Shorter buffers (3-5s)
- **Accuracy needs**: Longer buffers (7-10s)