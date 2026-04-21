"""
Main PathPlanner: orchestrates zone division, obstacle analysis,
fuzzy logic, guidance decision, and voice feedback.

Usage:
    from nav_assist.path_planner import PathPlanner

    planner = PathPlanner(speaker_enabled=True)
    instruction, details = planner.plan(seg_mask, depth_map)
    # ...
    planner.shutdown()
"""

import numpy as np

from nav_assist.path_planner.zones import compute_zones, compute_alert_zone
from nav_assist.path_planner.analyzer import (
    create_depth_gated_mask,
    compute_zone_occupancy,
    find_prominent_obstacle,
)
from nav_assist.path_planner.fuzzy_logic import (
    evaluate_rules, defuzzify, classify_action,
)
from nav_assist.path_planner.guidance import decide_navigation
from nav_assist.path_planner.speaker import NavigationSpeaker


class PathPlanner:
    """
    Full path planning pipeline with integrated voice guidance.

    Combines depth-gated obstacle detection, zone-based occupancy
    analysis, fuzzy logic navigation, and TTS speaker feedback.

    Parameters
    ----------
    speaker_enabled : bool
        Enable voice guidance via NavigationSpeaker.
    speaker_cooldown : float
        Minimum seconds between spoken instructions.
    """

    def __init__(self, speaker_enabled=True, speaker_cooldown=2.5):
        self.speaker = None
        if speaker_enabled:
            self.speaker = NavigationSpeaker(cooldown=speaker_cooldown)

    def plan(self, seg_mask, depth_map):
        """
        Run the full path planning pipeline.

        Parameters
        ----------
        seg_mask : np.ndarray (H, W) uint8
            Semantic segmentation class indices (ADE20K, 0..149).
        depth_map : np.ndarray (H, W) float32
            Disparity / inverse-depth map (higher = nearer).

        Returns
        -------
        instruction : str
            Navigation instruction for the user.
        details : dict
            Full pipeline outputs for visualization/debugging.
        """
        h, w = seg_mask.shape[:2]

        # 1. Depth-gated obstacle mask
        obstacle_mask, obstacle_labels = create_depth_gated_mask(
            seg_mask, depth_map)

        # 2. Zone division (Fig 5 & 6)
        zones = compute_zones(h, w)

        # 3. Per-zone occupancy (OStatus)
        occupancy, zone_labels = compute_zone_occupancy(
            obstacle_mask, obstacle_labels, zones)

        # 4. Prominent obstacle in center zones
        obstacle_name, obstacle_id, obstacle_pos = find_prominent_obstacle(
            zone_labels)

        # 5. Fuzzy logic evaluation
        rule_strengths = evaluate_rules(occupancy)
        centroid = defuzzify(rule_strengths)
        fuzzy_action = classify_action(centroid)

        # 6. Navigation guidance decision
        instruction, action, severity = decide_navigation(
            occupancy, obstacle_name, obstacle_pos)

        # 7. STOP override: all ground zones heavily blocked
        if all(occupancy.get(z, 0) > 0.50
               for z in ('ground_left', 'ground_center', 'ground_right')):
            instruction = 'Stop. Path completely blocked.'
            action = 'STOP'
            severity = 'emergency'

        # 8. Speak the instruction
        if self.speaker:
            self.speaker.speak(instruction, severity)

        # Build details dict (with backward-compat keys for visualization)
        details = {
            'action': action,
            'severity': severity,
            'prominent_obstacle': obstacle_name,
            'obstacle_position': obstacle_pos,
            'zones': zones,
            'occupancy': occupancy,
            'zone_labels': zone_labels,
            'rule_strengths': rule_strengths,
            'centroid': centroid,
            'fuzzy_action': fuzzy_action,
            'obstacle_mask': obstacle_mask,
            'obstacle_labels': obstacle_labels,
            # Backward-compat keys used by build_navigation_overlay
            'sector_bounds': _zones_to_sector_bounds(zones),
            'ostatus': _occupancy_to_ostatus(occupancy),
            'valid_mask': obstacle_mask,
        }

        return instruction, details

    def toggle_speaker(self):
        """Toggle speaker on/off. Returns the new state."""
        if self.speaker:
            return self.speaker.toggle()
        return False

    def shutdown(self):
        """Clean up resources (speaker thread)."""
        if self.speaker:
            self.speaker.shutdown()


# ── Backward-compatibility helpers ─────────────────────────────────────
# The visualization module expects the old sector/ostatus naming scheme.

_ZONE_TO_SECTOR = {
    'overhead_left':   'top_left',
    'overhead_center': 'top_mid',
    'overhead_right':  'top_right',
    'ground_left':     'bot_left',
    'ground_center':   'bot_mid',
    'ground_right':    'bot_right',
}


def _zones_to_sector_bounds(zones):
    """Convert zone dict to the old sector_bounds format."""
    return {_ZONE_TO_SECTOR[k]: v for k, v in zones.items()
            if k in _ZONE_TO_SECTOR}


def _occupancy_to_ostatus(occupancy):
    """Convert occupancy dict to the old ostatus format."""
    return {_ZONE_TO_SECTOR[k]: v for k, v in occupancy.items()
            if k in _ZONE_TO_SECTOR}


# ── Functional API (backward compatible) ──────────────────────────────
_default_planner = None


def plan_path(seg_mask, depth_map):
    """
    Backward-compatible functional API.

    Same as the old ``nav_assist.path_planner.plan_path``.
    Uses a module-level PathPlanner without speaker.
    """
    global _default_planner
    if _default_planner is None:
        _default_planner = PathPlanner(speaker_enabled=False)
    return _default_planner.plan(seg_mask, depth_map)
