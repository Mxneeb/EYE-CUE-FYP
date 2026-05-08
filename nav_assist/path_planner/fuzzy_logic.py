"""
Fuzzy logic engine for navigation decisions.

Uses trapezoidal membership functions and a 3-rule Mamdani inference
system to produce a continuous navigation signal in [-1, +1]:
  -1 = turn left,  0 = go ahead,  +1 = turn right.
"""

import numpy as np


# ════════════════════════════════════════════════════════════════════════
# Trapezoidal Membership Function
# ════════════════════════════════════════════════════════════════════════

def _trapz(x, a, b, c, d):
    """Trapezoidal MF: ramps up a->b, flat b->c, ramps down c->d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


# ════════════════════════════════════════════════════════════════════════
# Input MFs — Zone Occupancy (0..1)
# ════════════════════════════════════════════════════════════════════════

def mu_free(v):
    """Zone is FREE.  Peak at v<=0.10, drops to 0 at v=0.40."""
    return _trapz(v, -0.01, 0.0, 0.10, 0.40)


def mu_blocked(v):
    """Zone is BLOCKED.  0 at v<=0.10, rises to 1 at v=0.40."""
    return _trapz(v, 0.10, 0.40, 1.0, 1.01)


# ════════════════════════════════════════════════════════════════════════
# Output MFs — Navigation Space [-1, +1]
# ════════════════════════════════════════════════════════════════════════

def _mu_out_left(x):
    return _trapz(x, -1.01, -1.0, -0.5, -0.1)


def _mu_out_ahead(x):
    return _trapz(x, -0.35, -0.05, 0.05, 0.35)


def _mu_out_right(x):
    return _trapz(x, 0.1, 0.5, 1.0, 1.01)


# ════════════════════════════════════════════════════════════════════════
# Rule Evaluation
# ════════════════════════════════════════════════════════════════════════

def evaluate_rules(occupancy):
    """
    Evaluate 3 fuzzy rules using ground and overhead zones.

    Rule 1 (Ahead):  center ground AND center overhead are free
    Rule 2 (Left):   center blocked AND left side free
    Rule 3 (Right):  center blocked AND right side free

    Parameters
    ----------
    occupancy : dict
        Zone occupancies: ground_center, overhead_center,
        ground_left, overhead_left, ground_right, overhead_right.

    Returns
    -------
    dict : move_ahead, move_left, move_right firing strengths
    """
    gc = occupancy.get('ground_center', 0.0)
    oc = occupancy.get('overhead_center', 0.0)
    gl = occupancy.get('ground_left', 0.0)
    ol = occupancy.get('overhead_left', 0.0)
    gr = occupancy.get('ground_right', 0.0)
    or_ = occupancy.get('overhead_right', 0.0)

    # R1: both center zones are free
    r1 = min(mu_free(gc), mu_free(oc))

    # Center blocked strength
    center_blocked = max(mu_blocked(gc), mu_blocked(oc))

    # R2: center blocked AND left side free
    left_free = min(mu_free(gl), mu_free(ol))
    r2 = min(center_blocked, left_free)

    # R3: center blocked AND right side free
    right_free = min(mu_free(gr), mu_free(or_))
    r3 = min(center_blocked, right_free)

    return {'move_ahead': r1, 'move_left': r2, 'move_right': r3}


# ════════════════════════════════════════════════════════════════════════
# Defuzzification
# ════════════════════════════════════════════════════════════════════════

def defuzzify(rule_strengths):
    """Centroid (Mamdani) defuzzification over [-1, 1] navigation space."""
    x_pts = np.linspace(-1.0, 1.0, 201)
    aggregated = np.zeros_like(x_pts)

    for i, x in enumerate(x_pts):
        ml = min(rule_strengths['move_left'],  _mu_out_left(x))
        ma = min(rule_strengths['move_ahead'], _mu_out_ahead(x))
        mr = min(rule_strengths['move_right'], _mu_out_right(x))
        aggregated[i] = max(ml, ma, mr)

    total = aggregated.sum()
    if total < 1e-8:
        return 0.0
    return float((x_pts * aggregated).sum() / total)


def classify_action(centroid):
    """Map defuzzified centroid to a 5-level action string."""
    if centroid < -0.40:
        return 'MOVE LEFT'
    elif centroid < -0.15:
        return 'MOVE SLIGHT LEFT'
    elif centroid <= 0.15:
        return 'MOVE AHEAD'
    elif centroid <= 0.40:
        return 'MOVE SLIGHT RIGHT'
    else:
        return 'MOVE RIGHT'
