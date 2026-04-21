"""
Nav-Assist: Real-time obstacle detection and navigation for visually impaired.

Modules:
    - app: Main application (4-panel display)
    - path_planner: Zone-based path planning with voice guidance (Fig 5 & 6)
    - navigation_controller: Hysteresis-based instruction stability
    - obstacle: Depth-segmentation fusion obstacle detection
    - audio: Basic TTS audio feedback
    - visualization: Panel builders and overlay drawing
    - workers: Threaded depth & segmentation inference
    - enhanced_*: Enhanced versions with additional features
    - system_health: Health monitoring and environmental detection
"""

# ── Path planner module (new structured package) ──────────────────────
from nav_assist.path_planner import (
    PathPlanner,
    plan_path,
    NavigationSpeaker,
    compute_zones,
    compute_alert_zone,
    create_depth_gated_mask,
    compute_zone_occupancy,
    find_prominent_obstacle,
    evaluate_rules,
    defuzzify,
    classify_action,
    decide_navigation,
)

# ── Original modules ──────────────────────────────────────────────────
from nav_assist.app import main as app_main
from nav_assist.audio import AudioFeedback
from nav_assist.config import *
from nav_assist.navigation_controller import NavigationLogicController
from nav_assist.obstacle import detect_obstacles
from nav_assist.visualization import (
    build_navigation_overlay,
    build_camera_panel,
    build_depth_panel,
    build_seg_panel,
    build_obstacle_panel,
    build_status_bar,
)
from nav_assist.workers import DepthWorker, SegWorker

# ── Enhanced modules ──────────────────────────────────────────────────
from nav_assist.enhanced_app import main as enhanced_main
from nav_assist.enhanced_obstacle import (
    detect_obstacles_enhanced,
    ObstacleTracker,
    KalmanTracker,
    compute_depth_confidence,
    compute_seg_confidence,
    confidence_weighted_fusion,
    hierarchical_classification,
    get_nearest_obstacle_direction,
    create_obstacle_heatmap,
)
from nav_assist.enhanced_path_planner import (
    plan_path_enhanced,
    TrajectoryHistory,
    compute_variable_sector_bounds,
    generate_alternative_paths,
    draw_navigation_arrow,
)
from nav_assist.enhanced_visualization import (
    PerformanceMonitor,
    SessionRecorder,
    draw_depth_histogram,
    draw_trajectory,
    create_hazard_heatmap,
    draw_ar_navigation,
    build_confidence_overlay,
    build_metrics_panel,
    build_depth_histogram_panel,
    build_hazard_heatmap_panel,
)
from nav_assist.enhanced_audio import (
    EnhancedAudioFeedback,
    ComfortScorer,
    SpatialAudioMixer,
    AudioIconGenerator,
)
from nav_assist.system_health import (
    SystemHealthMonitor,
    EnvironmentalDetector,
    ConfigLoader,
    HealthStatus,
    create_default_config,
)

__all__ = [
    # Path planner module
    'PathPlanner',
    'plan_path',
    'NavigationSpeaker',
    'compute_zones',
    'compute_alert_zone',
    'create_depth_gated_mask',
    'decide_navigation',
    # Original
    'app_main',
    'AudioFeedback',
    'NavigationLogicController',
    'detect_obstacles',
    'build_navigation_overlay',
    'DepthWorker',
    'SegWorker',
    # Enhanced
    'enhanced_main',
    'detect_obstacles_enhanced',
    'ObstacleTracker',
    'KalmanTracker',
    'compute_depth_confidence',
    'compute_seg_confidence',
    'confidence_weighted_fusion',
    'hierarchical_classification',
    'get_nearest_obstacle_direction',
    'create_obstacle_heatmap',
    'plan_path_enhanced',
    'TrajectoryHistory',
    'compute_variable_sector_bounds',
    'generate_alternative_paths',
    'draw_navigation_arrow',
    'PerformanceMonitor',
    'SessionRecorder',
    'draw_depth_histogram',
    'draw_trajectory',
    'create_hazard_heatmap',
    'draw_ar_navigation',
    'build_confidence_overlay',
    'build_metrics_panel',
    'build_depth_histogram_panel',
    'build_hazard_heatmap_panel',
    'EnhancedAudioFeedback',
    'ComfortScorer',
    'SpatialAudioMixer',
    'AudioIconGenerator',
    'SystemHealthMonitor',
    'EnvironmentalDetector',
    'ConfigLoader',
    'HealthStatus',
    'create_default_config',
]
