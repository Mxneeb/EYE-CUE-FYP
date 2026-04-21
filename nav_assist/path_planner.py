"""
Path Planner Module — Fuzzy-logic based navigation guidance.

Evaluates obstacle risk in LEFT / CENTRE / RIGHT zones and produces
a navigation instruction: GO STRAIGHT, GO LEFT, GO RIGHT, or STOP.

Fuzzy membership functions:
    - Obstacle density  -> {low, medium, high}
    - Obstacle proximity -> {far, moderate, near}
    - Risk level         -> {safe, caution, danger}

Rules (simplified Mamdani-style):
    IF density=high AND proximity=near  THEN risk=danger
    IF density=medium OR proximity=moderate THEN risk=caution
    IF density=low AND proximity=far    THEN risk=safe

Navigation decision:
    1. If all zones are danger -> STOP
    2. If centre is safe -> GO STRAIGHT
    3. Pick the zone with the lowest risk -> GO LEFT / GO RIGHT
    4. If tie, prefer centre > right > left
"""

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# Fuzzy membership functions
# ════════════════════════════════════════════════════════════════════════════

def _mu_density_low(d):
    """Membership for 'low obstacle density'. Full at 0, zero at 0.15."""
    if d <= 0.02:
        return 1.0
    if d >= 0.15:
        return 0.0
    return (0.15 - d) / 0.13


def _mu_density_medium(d):
    """Membership for 'medium obstacle density'. Peak at 0.15, zero at 0/0.35."""
    if d <= 0.02 or d >= 0.35:
        return 0.0
    if d <= 0.15:
        return (d - 0.02) / 0.13
    return (0.35 - d) / 0.20


def _mu_density_high(d):
    """Membership for 'high obstacle density'. Zero at 0.20, full at 0.40."""
    if d <= 0.20:
        return 0.0
    if d >= 0.40:
        return 1.0
    return (d - 0.20) / 0.20


def _mu_proximity_far(p):
    """Membership for 'far' proximity. p is normalised disparity (0=far, 1=near)."""
    if p <= 0.3:
        return 1.0
    if p >= 0.6:
        return 0.0
    return (0.6 - p) / 0.3


def _mu_proximity_moderate(p):
    """Membership for 'moderate' proximity."""
    if p <= 0.2 or p >= 0.8:
        return 0.0
    if p <= 0.5:
        return (p - 0.2) / 0.3
    return (0.8 - p) / 0.3


def _mu_proximity_near(p):
    """Membership for 'near' proximity."""
    if p <= 0.5:
        return 0.0
    if p >= 0.8:
        return 1.0
    return (p - 0.5) / 0.3


# ════════════════════════════════════════════════════════════════════════════
# Fuzzy inference
# ════════════════════════════════════════════════════════════════════════════

def compute_zone_risk(density, max_disparity_normalised):
    """
    Compute risk score for a zone using fuzzy inference.

    Parameters
    ----------
    density : float
        Fraction of zone covered by obstacles (0..1).
    max_disparity_normalised : float
        Normalised max disparity of obstacles (0=far, 1=near).

    Returns
    -------
    risk : float
        Risk score in [0, 1]. 0 = safe, 1 = danger.
    risk_label : str
        One of 'safe', 'caution', 'danger'.
    """
    # Fuzzy memberships
    d_lo = _mu_density_low(density)
    d_md = _mu_density_medium(density)
    d_hi = _mu_density_high(density)

    p_far = _mu_proximity_far(max_disparity_normalised)
    p_mod = _mu_proximity_moderate(max_disparity_normalised)
    p_near = _mu_proximity_near(max_disparity_normalised)

    # Fuzzy rules -> risk memberships
    #   Rule 1: IF density=low  AND proximity=far     THEN risk=safe       (0.0)
    #   Rule 2: IF density=low  AND proximity=moderate THEN risk=safe      (0.1)
    #   Rule 3: IF density=medium AND proximity=far    THEN risk=caution   (0.4)
    #   Rule 4: IF density=medium AND proximity=moderate THEN risk=caution (0.5)
    #   Rule 5: IF density=medium AND proximity=near   THEN risk=danger    (0.7)
    #   Rule 6: IF density=high AND proximity=far      THEN risk=caution   (0.5)
    #   Rule 7: IF density=high AND proximity=moderate THEN risk=danger    (0.8)
    #   Rule 8: IF density=high AND proximity=near     THEN risk=danger    (1.0)
    #   Rule 9: IF density=low  AND proximity=near     THEN risk=caution   (0.4)

    rules = [
        (min(d_lo, p_far),  0.0),
        (min(d_lo, p_mod),  0.1),
        (min(d_md, p_far),  0.4),
        (min(d_md, p_mod),  0.5),
        (min(d_md, p_near), 0.7),
        (min(d_hi, p_far),  0.5),
        (min(d_hi, p_mod),  0.8),
        (min(d_hi, p_near), 1.0),
        (min(d_lo, p_near), 0.4),
    ]

    # Defuzzification: weighted average
    numerator = sum(w * v for w, v in rules)
    denominator = sum(w for w, _ in rules)

    if denominator < 1e-8:
        risk = 0.0
    else:
        risk = numerator / denominator

    # Classify
    if risk < 0.30:
        risk_label = 'safe'
    elif risk < 0.65:
        risk_label = 'caution'
    else:
        risk_label = 'danger'

    return risk, risk_label


def plan_path(zones, max_global_disparity):
    """
    Generate a navigation instruction from zone obstacle data.

    Parameters
    ----------
    zones : dict
        Output from obstacle.get_zone_obstacles(). Keys: 'left', 'center', 'right'.
    max_global_disparity : float
        Maximum disparity in the full depth image (for normalisation).

    Returns
    -------
    instruction : str
        Navigation instruction string.
    zone_risks : dict
        Risk score and label for each zone.
    """
    if max_global_disparity < 1e-6:
        return 'GO STRAIGHT — path clear', {
            z: {'risk': 0.0, 'label': 'safe'} for z in ('left', 'center', 'right')
        }

    zone_risks = {}
    for zone_name in ('left', 'center', 'right'):
        z = zones[zone_name]
        norm_disp = z['max_disparity'] / max_global_disparity
        risk, label = compute_zone_risk(z['density'], norm_disp)
        zone_risks[zone_name] = {'risk': risk, 'label': label}

    left_r = zone_risks['left']['risk']
    center_r = zone_risks['center']['risk']
    right_r = zone_risks['right']['risk']

    # Decision logic
    all_danger = all(zone_risks[z]['label'] == 'danger'
                     for z in ('left', 'center', 'right'))
    if all_danger:
        return 'STOP — obstacles ahead', zone_risks

    # If centre is safe, go straight
    if zone_risks['center']['label'] == 'safe':
        return 'GO STRAIGHT — path clear', zone_risks

    # Find safest zone
    min_risk = min(left_r, center_r, right_r)

    if center_r == min_risk:
        if zone_risks['center']['label'] == 'caution':
            return 'GO STRAIGHT — caution ahead', zone_risks
        return 'GO STRAIGHT — path clear', zone_risks
    elif right_r < left_r:
        return 'GO RIGHT — obstacle on left/center', zone_risks
    elif left_r < right_r:
        return 'GO LEFT — obstacle on right/center', zone_risks
    else:
        # Tie between left and right — prefer the one with fewer obstacles
        if len(zones['right']['obstacles']) <= len(zones['left']['obstacles']):
            return 'GO RIGHT — fewer obstacles', zone_risks
        return 'GO LEFT — fewer obstacles', zone_risks
