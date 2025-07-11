import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any

class Config:
    def __init__(self, profile: str = None):
        load_dotenv()
        self.config = self._load_config(profile)
        self._apply_env_overrides()
    
    def _load_config(self, profile: str) -> Dict[str, Any]:
        # Load default config
        with open("config/default.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Override with profile if specified
        if profile:
            profile_path = f"config/{profile}.yaml"
            if os.path.exists(profile_path):
                with open(profile_path, "r") as f:
                    profile_config = yaml.safe_load(f)
                    config = self._merge_configs(config, profile_config)
        
        return config
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        # Apply environment variable overrides
        if os.getenv("AUDIO_DEVICE_INDEX"):
            self.config["audio"]["device_index"] = int(os.getenv("AUDIO_DEVICE_INDEX"))
        if os.getenv("AUDIO_DEVICE_NAME"):
            self.config["audio"]["device_name"] = os.getenv("AUDIO_DEVICE_NAME")
        if os.getenv("MQTT_BROKER"):
            self.config["mqtt"]["enabled"] = True
            self.config["mqtt"]["broker"] = os.getenv("MQTT_BROKER")
            self.config["mqtt"]["username"] = os.getenv("MQTT_USERNAME")
            self.config["mqtt"]["password"] = os.getenv("MQTT_PASSWORD")
    
    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value