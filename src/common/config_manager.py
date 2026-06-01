import yaml
import os
import glob
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages loading, saving, and validating system configuration.
    Handles field geometry calculations and profile management.
    """
    
    def __init__(self, default_config_path: str = "config/system_config.yaml", profiles_dir: str = "config/profiles"):
        self.default_config_path = default_config_path
        self.profiles_dir = profiles_dir
        self.config: Dict[str, Any] = {}
        
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir, exist_ok=True)

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Loads configuration from the specified path or default path."""
        load_path = path or self.default_config_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Configuration file not found: {load_path}")
            
        with open(load_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
            
        logger.info(f"Loaded configuration from {load_path}")
        return self.config

    def save(self, config: Dict[str, Any], path: Optional[str] = None) -> None:
        """Saves configuration to the specified path or default path."""
        save_path = path or self.default_config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        logger.info(f"Saved configuration to {save_path}")

    def derive_stereo_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates derived parameters like focal_length_px from physical lens parameters.
        Modifies the config dictionary in-place and returns it.
        """
        if "field_geometry" in config and "camera" in config and "stereo" in config:
            fg = config["field_geometry"]
            cam = config["camera"]["resolution"]
            
            # Calculate focal length in pixels
            # formula: focal_px = (focal_mm / sensor_width_mm) * image_width_px
            if fg.get("focal_length_mm") and fg.get("sensor_width_mm") and cam.get("width"):
                focal_px = (fg["focal_length_mm"] / fg["sensor_width_mm"]) * cam["width"]
                config["stereo"]["focal_length_px"] = round(focal_px, 2)
            
            # Set principal point to center of image
            if cam.get("width") and cam.get("height"):
                config["stereo"]["principal_point_x"] = cam["width"] / 2.0
                config["stereo"]["principal_point_y"] = cam["height"] / 2.0
                
            # Keep baseline synced if it's in field_geometry
            if "baseline_mm" in fg:
                config["stereo"]["baseline_mm"] = fg["baseline_mm"]
                
        return config

    def list_profiles(self) -> List[str]:
        """Returns a list of available configuration profile names."""
        profiles = []
        for file_path in glob.glob(os.path.join(self.profiles_dir, "*.yaml")):
            basename = os.path.basename(file_path)
            profiles.append(os.path.splitext(basename)[0])
        return sorted(profiles)

    def load_profile(self, name: str) -> Dict[str, Any]:
        """Loads a specific profile by name."""
        path = os.path.join(self.profiles_dir, f"{name}.yaml")
        return self.load(path)

    def save_profile(self, config: Dict[str, Any], name: str) -> None:
        """Saves a configuration as a profile with the given name."""
        path = os.path.join(self.profiles_dir, f"{name}.yaml")
        self.save(config, path)

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validates the configuration and returns a list of error messages."""
        errors = []
        
        # Check required sections
        required_sections = ["network", "camera", "stereo", "field_geometry", "detection"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
                
        if errors:
            return errors
            
        # Field geometry validation
        fg = config.get("field_geometry", {})
        if fg.get("baseline_mm", 0) <= 0:
            errors.append("baseline_mm must be positive")
        if fg.get("focal_length_mm", 0) <= 0:
            errors.append("focal_length_mm must be positive")
            
        # Network validation
        net = config.get("network", {})
        port = net.get("port", 0)
        if not (1024 <= port <= 65535):
            errors.append(f"Invalid port: {port}. Must be between 1024 and 65535.")
            
        return errors
