"""
Path Planner Module — Zone-based obstacle analysis with fuzzy-logic
navigation and voice guidance for visually impaired users.

Based on:
  - Fig 5: 3m radius zone extraction (portrait/landscape adaptive)
  - Fig 6: Ground (0-0.4H) and overhead (0.6-0.9H) obstacle regions

Submodules:
  zones        — Frame zone division (Fig 5 & 6)
  analyzer     — Depth-gated obstacle detection and zone occupancy
  fuzzy_logic  — Trapezoidal membership functions and rule engine
  guidance     — Navigation decision logic (left / right / stop)
  speaker      — TTS voice feedback
  planner      — Main PathPlanner orchestrator
"""

from nav_assist.path_planner.planner import PathPlanner, plan_path
from nav_assist.path_planner.zones import compute_zones, compute_alert_zone
from nav_assist.path_planner.analyzer import (
    create_depth_gated_mask,
    compute_zone_occupancy,
    find_prominent_obstacle,
)
from nav_assist.path_planner.fuzzy_logic import (
    evaluate_rules,
    defuzzify,
    classify_action,
    mu_free,
    mu_blocked,
)
from nav_assist.path_planner.guidance import decide_navigation
from nav_assist.path_planner.speaker import NavigationSpeaker

__all__ = [
    'PathPlanner',
    'plan_path',
    'NavigationSpeaker',
    'compute_zones',
    'compute_alert_zone',
    'create_depth_gated_mask',
    'compute_zone_occupancy',
    'find_prominent_obstacle',
    'evaluate_rules',
    'defuzzify',
    'classify_action',
    'decide_navigation',
    'mu_free',
    'mu_blocked',
]
