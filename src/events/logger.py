import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
from threading import Thread

logger = logging.getLogger(__name__)

class EventLogger:
    def __init__(self, config):
        self.config = config
        self.events_file = config.get("logging.events_file", "logs/events.jsonl")
        self.mqtt_client = None
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.events_file), exist_ok=True)
        
        # Initialize MQTT if configured
        if config.get("mqtt.enabled", False):
            self._init_mqtt()
    
    def _init_mqtt(self):
        """Initialize MQTT client"""
        try:
            broker = self.config.get("mqtt.broker", "localhost:1883")
            if ":" in broker:
                host, port = broker.split(":")
                port = int(port)
            else:
                host, port = broker, 1883
            
            self.mqtt_client = mqtt.Client()
            
            # Set credentials if provided
            username = self.config.get("mqtt.username")
            password = self.config.get("mqtt.password")
            if username and password:
                self.mqtt_client.username_pw_set(username, password)
            
            # Connect to broker
            self.mqtt_client.connect(host, port, 60)
            self.mqtt_client.loop_start()
            
            logger.info(f"MQTT client connected to {broker}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MQTT: {e}")
            self.mqtt_client = None
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to file and MQTT"""
        timestamp = datetime.now().isoformat()
        
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "data": data
        }
        
        # Log to file
        self._log_to_file(event)
        
        # Publish to MQTT
        if self.mqtt_client:
            self._publish_to_mqtt(event)
    
    def _log_to_file(self, event: Dict[str, Any]):
        """Append event to JSONL file"""
        try:
            with open(self.events_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")
    
    def _publish_to_mqtt(self, event: Dict[str, Any]):
        """Publish event to MQTT topic"""
        try:
            topic = self.config.get("mqtt.topic", "audio/events")
            payload = json.dumps(event)
            self.mqtt_client.publish(topic, payload)
            logger.debug(f"Published event to MQTT topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to publish to MQTT: {e}")
    
    def log_anomaly(self, classification: str, confidence: float, 
                   transcript: Optional[str] = None, 
                   sentiment: Optional[Dict] = None):
        """Log an anomaly event"""
        data = {
            "classification": classification,
            "confidence": confidence
        }
        
        if transcript:
            data["transcript"] = transcript
        
        if sentiment:
            data["sentiment"] = sentiment
        
        self.log_event("anomaly", data)
        logger.info(f"Anomaly detected: {classification} (confidence: {confidence:.2f})")
    
    def log_speech(self, transcript: str, sentiment: Dict[str, Any]):
        """Log a speech event"""
        data = {
            "transcript": transcript,
            "sentiment": sentiment
        }
        
        self.log_event("speech", data)
        logger.info(f"Speech detected: {transcript[:50]}...")
    
    def close(self):
        """Clean up resources"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()