import json
import os
import csv
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class LocalEventStorage:
    """Enhanced local storage for events with multiple formats"""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = config.get("storage.base_dir", "data")
        self.events_file = os.path.join(self.base_dir, "events.jsonl")
        self.speech_file = os.path.join(self.base_dir, "speech_transcripts.csv")
        self.anomalies_file = os.path.join(self.base_dir, "anomalies.csv")
        self.daily_summary_file = os.path.join(self.base_dir, "daily_summary.json")
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()
        
        logger.info(f"Local storage initialized in: {self.base_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        # Speech transcripts CSV
        if not os.path.exists(self.speech_file):
            with open(self.speech_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'transcript', 'sentiment', 'confidence', 'duration'])
        
        # Anomalies CSV
        if not os.path.exists(self.anomalies_file):
            with open(self.anomalies_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'type', 'classification', 'confidence', 'description'])
    
    def store_event(self, event_type: str, data: Dict[str, Any]):
        """Store event in JSONL format"""
        timestamp = datetime.now().isoformat()
        
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "data": data
        }
        
        try:
            with open(self.events_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
    
    def store_speech(self, transcript: str, sentiment: Dict[str, Any], 
                    confidence: float = 0.0, duration: float = 0.0):
        """Store speech transcript in CSV format"""
        timestamp = datetime.now().isoformat()
        
        try:
            with open(self.speech_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    transcript,
                    sentiment.get('sentiment', 'neutral'),
                    confidence,
                    duration
                ])
            
            # Also store in JSONL
            self.store_event("speech", {
                "transcript": transcript,
                "sentiment": sentiment,
                "confidence": confidence,
                "duration": duration
            })
            
            logger.info(f"Speech stored: {transcript[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to store speech: {e}")
    
    def store_anomaly(self, anomaly_type: str, classification: str, 
                     confidence: float, description: str = ""):
        """Store anomaly in CSV format"""
        timestamp = datetime.now().isoformat()
        
        try:
            with open(self.anomalies_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    anomaly_type,
                    classification,
                    confidence,
                    description
                ])
            
            # Also store in JSONL
            self.store_event("anomaly", {
                "anomaly_type": anomaly_type,
                "classification": classification,
                "confidence": confidence,
                "description": description
            })
            
            logger.info(f"Anomaly stored: {anomaly_type} - {classification}")
            
        except Exception as e:
            logger.error(f"Failed to store anomaly: {e}")
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from JSONL file"""
        events = []
        
        try:
            if os.path.exists(self.events_file):
                with open(self.events_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Get last N lines
                for line in lines[-limit:]:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to read recent events: {e}")
            
        return events
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get statistics for current day"""
        today = datetime.now().date().isoformat()
        stats = {
            'date': today,
            'total_events': 0,
            'speech_count': 0,
            'anomaly_count': 0,
            'anomaly_types': {},
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0}
        }
        
        try:
            events = self.get_recent_events(1000)  # Check last 1000 events
            
            for event in events:
                # Only count today's events
                event_date = event['timestamp'][:10]  # YYYY-MM-DD
                if event_date != today:
                    continue
                    
                stats['total_events'] += 1
                
                if event['type'] == 'speech':
                    stats['speech_count'] += 1
                    sentiment = event['data'].get('sentiment', {}).get('sentiment', 'neutral')
                    if sentiment in stats['sentiment_distribution']:
                        stats['sentiment_distribution'][sentiment] += 1
                        
                elif event['type'] == 'anomaly':
                    stats['anomaly_count'] += 1
                    anomaly_type = event['data'].get('anomaly_type', 'unknown')
                    stats['anomaly_types'][anomaly_type] = stats['anomaly_types'].get(anomaly_type, 0) + 1
                    
        except Exception as e:
            logger.error(f"Failed to calculate daily stats: {e}")
            
        return stats
    
    def save_daily_summary(self):
        """Save daily summary to file"""
        stats = self.get_daily_stats()
        
        try:
            with open(self.daily_summary_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Daily summary saved: {stats['total_events']} events")
            
        except Exception as e:
            logger.error(f"Failed to save daily summary: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up data older than specified days"""
        # Implementation for cleanup if needed
        logger.info(f"Cleanup routine - keeping last {days_to_keep} days")
        # TODO: Implement cleanup logic