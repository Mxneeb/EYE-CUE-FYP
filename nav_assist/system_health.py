"""
System Health Monitor Module.

Monitors all system components for failures and anomalies:
    - Model output validation
    - Performance degradation detection
    - Graceful degradation on failures
    - Environmental challenge detection (low light, glare)
    - Auto-recovery mechanisms

Also includes configuration loading from YAML/JSON files.
"""

import os
import json
import time
import threading
from collections import deque
from enum import Enum
import numpy as np
import cv2


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"


class SystemHealthMonitor:
    """
    Monitors system health and provides graceful degradation.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Health history
        self.health_history = deque(maxlen=100)
        self.module_health = {
            'depth_model': HealthStatus.HEALTHY,
            'seg_model': HealthStatus.HEALTHY,
            'fusion': HealthStatus.HEALTHY,
            'path_planner': HealthStatus.HEALTHY,
            'audio': HealthStatus.HEALTHY,
        }
        
        # Failure counters
        self.failure_count = {
            'depth': 0,
            'segmentation': 0,
            'fusion': 0,
        }
        
        # Graceful degradation flags
        self.degradation_mode = False
        self.fallback_depth_enabled = False
        self.fallback_seg_enabled = False
        
        # Environmental conditions
        self.environmental_flags = {
            'low_light': False,
            'bright_light': False,
            'glare_detected': False,
            'motion_blur': False,
        }
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'failed_frames': 0,
            'degradation_events': 0,
            'recovery_events': 0,
            'avg_fps': 0,
        }
        
        self._lock = threading.Lock()
        self._last_check_time = time.time()
    
    def check_depth_health(self, depth_map):
        """
        Validate depth model output.
        
        Checks:
        - Valid value range
        - Not all zeros (model failure)
        - Not all max values (saturation)
        - Reasonable variance
        """
        if depth_map is None or depth_map.size == 0:
            return False, "Empty depth map"
        
        # Check for all zeros
        if depth_map.max() < 1e-6:
            return False, "Depth model producing zeros"
        
        # Check for saturation (all max values)
        max_val = depth_map.max()
        sat_ratio = (depth_map >= max_val * 0.99).mean()
        if sat_ratio > 0.5:
            return False, "Depth saturation detected"
        
        # Check for NaN or Inf
        if np.any(np.isnan(depth_map)) or np.any(np.isinf(depth_map)):
            return False, "Depth contains NaN/Inf"
        
        # Check variance (too low = failure)
        variance = depth_map.var()
        if variance < 1e-4:
            return False, "Depth variance too low"
        
        return True, "OK"
    
    def check_seg_health(self, seg_mask):
        """Validate segmentation model output."""
        if seg_mask is None or seg_mask.size == 0:
            return False, "Empty segmentation mask"
        
        # Check for all same class
        unique_classes = len(np.unique(seg_mask))
        if unique_classes < 3:
            return False, f"Too few classes ({unique_classes})"
        
        # Check for NaN
        if np.any(np.isnan(seg_mask.astype(float))):
            return False, "Segmentation contains NaN"
        
        return True, "OK"
    
    def check_fusion_health(self, obstacle_mask, depth_map):
        """Validate fusion output."""
        if obstacle_mask is None:
            return False, "Empty obstacle mask"
        
        # If depth is healthy but fusion produces nothing, might be an issue
        if depth_map.max() > 1e-3 and obstacle_mask.sum() == 0:
            # This could be valid in clear scenes, but track it
            return True, "Clear scene (no obstacles)"
        
        return True, "OK"
    
    def update_health(self, depth_map=None, seg_mask=None, obstacle_mask=None,
                      fps_dict=None):
        """
        Update overall system health based on component outputs.
        """
        with self._lock:
            self.stats['total_frames'] += 1
            
            healthy = True
            
            # Check depth
            if depth_map is not None:
                ok, msg = self.check_depth_health(depth_map)
                if ok:
                    self.failure_count['depth'] = 0
                    self.module_health['depth_model'] = HealthStatus.HEALTHY
                else:
                    self.failure_count['depth'] += 1
                    self.module_health['depth_model'] = HealthStatus.DEGRADED
                    if self.failure_count['depth'] > 5:
                        self.module_health['depth_model'] = HealthStatus.FAILING
                        self.trigger_degradation('depth')
                    healthy = False
            
            # Check segmentation
            if seg_mask is not None:
                ok, msg = self.check_seg_health(seg_mask)
                if ok:
                    self.failure_count['segmentation'] = 0
                    self.module_health['seg_model'] = HealthStatus.HEALTHY
                else:
                    self.failure_count['segmentation'] += 1
                    self.module_health['seg_model'] = HealthStatus.DEGRADED
                    if self.failure_count['segmentation'] > 5:
                        self.module_health['seg_model'] = HealthStatus.FAILING
                        self.trigger_degradation('segmentation')
                    healthy = False
            
            # Check fusion
            if obstacle_mask is not None and depth_map is not None:
                ok, msg = self.check_fusion_health(obstacle_mask, depth_map)
                if ok:
                    self.failure_count['fusion'] = 0
                    self.module_health['fusion'] = HealthStatus.HEALTHY
                else:
                    self.failure_count['fusion'] += 1
                    self.module_health['fusion'] = HealthStatus.DEGRADED
            
            # Update FPS
            if fps_dict:
                self.stats['avg_fps'] = np.mean(list(fps_dict.values()))
            
            # Record health
            overall = self.get_overall_health()
            self.health_history.append({
                'time': time.time(),
                'status': overall,
                'modules': dict(self.module_health),
            })
            
            if not healthy:
                self.stats['failed_frames'] += 1
            
            # Check for recovery
            if self.degradation_mode and healthy:
                if all(c < 2 for c in self.failure_count.values()):
                    self.recover_from_degradation()
    
    def trigger_degradation(self, module):
        """
        Trigger graceful degradation when a module fails.
        """
        with self._lock:
            if not self.degradation_mode:
                self.degradation_mode = True
                self.stats['degradation_events'] += 1
                print(f'[Health] Entering degraded mode - {module} failing')
            
            if module == 'depth':
                self.fallback_depth_enabled = True
            elif module == 'segmentation':
                self.fallback_seg_enabled = True
    
    def recover_from_degradation(self):
        """
        Recover from degradation mode when all modules are healthy.
        """
        with self._lock:
            if self.degradation_mode:
                self.degradation_mode = False
                self.fallback_depth_enabled = False
                self.fallback_seg_enabled = False
                self.stats['recovery_events'] += 1
                print('[Health] Recovered from degraded mode')
    
    def get_overall_health(self):
        """Get overall system health status."""
        statuses = list(self.module_health.values())
        
        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        elif any(s == HealthStatus.FAILING for s in statuses):
            return HealthStatus.FAILING
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def is_safe_mode(self):
        """Check if system should enter extra-safe mode."""
        return (self.get_overall_health() != HealthStatus.HEALTHY or
                any(self.environmental_flags.values()))
    
    def get_degradation_warning(self):
        """Get warning message for current degradation state."""
        if not self.degradation_mode:
            return None
        
        warnings = []
        if self.fallback_depth_enabled:
            warnings.append("Depth estimation degraded")
        if self.fallback_seg_enabled:
            warnings.append("Segmentation degraded")
        
        if warnings:
            return "System running in fallback mode: " + ", ".join(warnings)
        return None
    
    def get_stats(self):
        """Get system statistics."""
        with self._lock:
            stats = dict(self.stats)
            stats['health_score'] = self.calculate_health_score()
            stats['reliability'] = (
                (stats['total_frames'] - stats['failed_frames']) /
                max(stats['total_frames'], 1)
            )
            return stats
    
    def calculate_health_score(self):
        """Calculate overall health score (0-100)."""
        base_score = 100
        
        # Deduct for failures
        failure_rate = self.stats['failed_frames'] / max(self.stats['total_frames'], 1)
        base_score -= failure_rate * 30
        
        # Deduct for degradation events
        base_score -= min(self.stats['degradation_events'] * 5, 20)
        
        # Deduct for environmental challenges
        env_penalty = sum(self.environmental_flags.values()) * 5
        base_score -= min(env_penalty, 15)
        
        return max(0, min(100, int(base_score)))


class EnvironmentalDetector:
    """
    Detect environmental challenges like low light, glare, reflections.
    """
    
    def __init__(self):
        self.frame_history = deque(maxlen=10)
        self.last_detection_time = 0
        self.detection_interval = 0.5  # seconds
        
    def detect_low_light(self, frame):
        """
        Detect low light conditions.
        
        Returns:
            is_low_light (bool), brightness (0-255)
        """
        if frame is None:
            return False, 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        
        # Thresholds
        is_low = brightness < 50
        
        return is_low, brightness
    
    def detect_glare(self, frame):
        """
        Detect glare or bright light spots.
        
        Returns:
            has_glare (bool), glare_intensity (0-1)
        """
        if frame is None:
            return False, 0
        
        # Convert to HSV for better light detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect very bright regions (potential glare)
        bright_mask = (hsv[:, :, 2] > 230).astype(np.float32)
        
        # Calculate glare intensity
        glare_intensity = bright_mask.mean()
        has_glare = glare_intensity > 0.05  # More than 5% bright pixels
        
        return has_glare, glare_intensity
    
    def detect_motion_blur(self, frame):
        """
        Detect motion blur using Laplacian variance.
        
        Returns:
            is_blurry (bool), laplacian_var
        """
        if frame is None:
            return False, 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Lower variance = more blur
        is_blurry = variance < 50
        
        return is_blurry, variance
    
    def detect_reflections(self, frame, depth_map=None):
        """
        Detect reflective surfaces that may confuse depth estimation.
        
        Returns:
            has_reflections (bool), reflection_score (0-1)
        """
        if frame is None:
            return False, 0
        
        # High contrast regions might be reflections
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sobel for edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Sharp horizontal lines might be reflections
        horizontal_edges = (sobely > 50).sum() / sobely.size
        
        # Check if depth is inconsistent with edges (reflection indicator)
        if depth_map is not None:
            depth_gradient = np.gradient(depth_map)
            depth_inconsistency = np.abs(depth_gradient[0]) + np.abs(depth_gradient[1])
            depth_inconsistency = (depth_inconsistency / (depth_inconsistency.max() + 1e-6)).mean()
        else:
            depth_inconsistency = 0
        
        reflection_score = min(horizontal_edges * 10 + depth_inconsistency, 1.0)
        has_reflections = reflection_score > 0.3
        
        return has_reflections, reflection_score
    
    def analyze_environment(self, frame, depth_map=None):
        """
        Perform full environmental analysis.
        
        Returns:
            dict of environmental flags
        """
        now = time.time()
        if now - self.last_detection_time < self.detection_interval:
            return None  # Too soon to re-analyze
        
        self.last_detection_time = now
        
        flags = {}
        
        # Low light
        is_low_light, brightness = self.detect_low_light(frame)
        flags['low_light'] = is_low_light
        flags['brightness'] = brightness
        
        # Glare
        has_glare, glare_intensity = self.detect_glare(frame)
        flags['glare_detected'] = has_glare
        flags['glare_intensity'] = glare_intensity
        
        # Motion blur
        is_blurry, laplacian_var = self.detect_motion_blur(frame)
        flags['motion_blur'] = is_blurry
        flags['sharpness'] = laplacian_var
        
        # Reflections
        has_reflections, reflection_score = self.detect_reflections(frame, depth_map)
        flags['reflections'] = has_reflections
        flags['reflection_score'] = reflection_score
        
        return flags
    
    def apply_environmental_compensation(self, frame, depth_map, flags):
        """
        Apply compensation for environmental challenges.
        
        Parameters:
            frame: Original camera frame
            depth_map: Depth map to compensate
            flags: Environmental flags from analyze_environment
        
        Returns:
            compensated_frame, compensated_depth
        """
        compensated_frame = frame.copy()
        compensated_depth = depth_map.copy()
        
        if flags.get('low_light', False):
            # Boost brightness
            compensated_frame = cv2.convertScaleAbs(compensated_frame, alpha=1.5, beta=30)
        
        if flags.get('motion_blur', False):
            # Apply sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            compensated_frame = cv2.filter2D(compensated_frame, -1, kernel)
        
        if flags.get('glare_detected', False):
            # Reduce glare impact
            hsv = cv2.cvtColor(compensated_frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255)
            compensated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return compensated_frame, compensated_depth


# ════════════════════════════════════════════════════════════════════════════
# Configuration Loading
# ════════════════════════════════════════════════════════════════════════════

class ConfigLoader:
    """
    Load and save configuration from JSON/YAML files.
    """
    
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.config = {}
        self.defaults = self._get_defaults()
        
    def _get_defaults(self):
        """Get default configuration values."""
        return {
            'depth_model': {
                'encoder': 'vitb',
                'features': 128,
                'input_size': 308,
            },
            'fusion': {
                'confidence_alpha': 0.6,
                'confidence_beta': 0.4,
                'edge_sigma': 2.0,
            },
            'path_planning': {
                'variable_grid': True,
                'trajectory_prediction': True,
                'alternative_paths': True,
            },
            'audio': {
                'spatial_enabled': True,
                'distance_volume': True,
                'icon_enabled': True,
                'rhythm_enabled': True,
                'speech_rate': 160,
            },
            'display': {
                'show_depth_histogram': True,
                'show_trajectory': True,
                'show_hazard_heatmap': True,
                'show_ar_overlay': True,
            },
            'performance': {
                'enable_monitoring': True,
                'history_size': 100,
                'record_sessions': True,
            },
            'safety': {
                'enable_health_check': True,
                'graceful_degradation': True,
                'safety_mode_threshold': 0.3,
            },
        }
    
    def load(self, filename='config.json'):
        """Load configuration from file."""
        filepath = os.path.join(self.config_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    if filepath.endswith('.json'):
                        loaded = json.load(f)
                    else:
                        # Try YAML
                        import yaml
                        loaded = yaml.safe_load(f)
                
                # Merge with defaults
                self.config = self._deep_merge(self.defaults, loaded)
                print(f'[Config] Loaded configuration from {filepath}')
                return self.config
            except Exception as e:
                print(f'[Config] Failed to load config: {e}')
        
        # Use defaults
        self.config = self.defaults
        print('[Config] Using default configuration')
        return self.config
    
    def save(self, filename='config.json'):
        """Save current configuration to file."""
        os.makedirs(self.config_dir, exist_ok=True)
        filepath = os.path.join(self.config_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                if filepath.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    import yaml
                    yaml.dump(self.config, f, default_flow_style=False)
            
            print(f'[Config] Saved configuration to {filepath}')
            return True
        except Exception as e:
            print(f'[Config] Failed to save config: {e}')
            return False
    
    def get(self, key, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _deep_merge(self, base, override):
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


def create_default_config():
    """Create and save a default configuration file."""
    loader = ConfigLoader()
    loader.config = loader.defaults
    loader.save('config.json')
    return loader.config
