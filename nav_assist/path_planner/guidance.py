"""
Navigation guidance: decides what direction to give based on zone analysis.

Core logic:
  1. If center is clear            -> "Clear ahead"
  2. If obstacle ahead, left clear  -> "Go left"
  3. If obstacle ahead, right clear -> "Go right"
  4. If both sides clear            -> pick the clearer side
  5. If neither side is clear       -> "Stop"
"""

# Zone occupancy above this value is considered blocked
BLOCKED_THRESHOLD = 0.25

# Zone occupancy below this value is considered safe to walk through
SAFE_THRESHOLD = 0.15


def decide_navigation(occupancy, obstacle_name=None, obstacle_position='ahead'):
    """
    Determine the navigation instruction based on zone occupancies.

    Parameters
    ----------
    occupancy : dict
        Per-zone occupancy values (0..1).
    obstacle_name : str or None
        Prominent obstacle class name (e.g. "wall", "chair").
    obstacle_position : str
        'ahead' or 'overhead'.

    Returns
    -------
    instruction : str
        Human-readable guidance for the user.
    action : str
        Machine-readable action code (MOVE AHEAD / MOVE LEFT / MOVE RIGHT / STOP).
    severity : str
        'safe', 'warning', 'critical', or 'emergency'.
    """
    gc = occupancy.get('ground_center', 0.0)
    oc = occupancy.get('overhead_center', 0.0)
    gl = occupancy.get('ground_left', 0.0)
    ol = occupancy.get('overhead_left', 0.0)
    gr = occupancy.get('ground_right', 0.0)
    or_ = occupancy.get('overhead_right', 0.0)

    center_blocked = gc > BLOCKED_THRESHOLD or oc > BLOCKED_THRESHOLD
    left_safe = gl < SAFE_THRESHOLD and ol < SAFE_THRESHOLD
    right_safe = gr < SAFE_THRESHOLD and or_ < SAFE_THRESHOLD

    left_occ = max(gl, ol)
    right_occ = max(gr, or_)

    # ── All ground zones heavily blocked → emergency ──────────────────
    if gc > 0.50 and gl > 0.40 and gr > 0.40:
        return 'Stop. Path blocked on all sides.', 'STOP', 'emergency'

    # ── Center clear → move ahead ─────────────────────────────────────
    if not center_blocked:
        if obstacle_name and obstacle_position == 'overhead':
            return (f'Clear ahead. Watch for {obstacle_name} overhead.',
                    'MOVE AHEAD', 'warning')
        return 'Clear ahead.', 'MOVE AHEAD', 'safe'

    # ── Center blocked → find escape route ────────────────────────────
    obs_text = obstacle_name.capitalize() if obstacle_name else 'Obstacle'
    pos_text = 'overhead' if obstacle_position == 'overhead' else 'ahead'

    if left_safe and right_safe:
        # Both sides clear — pick the one with less occupancy
        if left_occ <= right_occ:
            return f'{obs_text} {pos_text}. Go left.', 'MOVE LEFT', 'critical'
        else:
            return f'{obs_text} {pos_text}. Go right.', 'MOVE RIGHT', 'critical'

    if left_safe:
        return f'{obs_text} {pos_text}. Go left.', 'MOVE LEFT', 'critical'

    if right_safe:
        return f'{obs_text} {pos_text}. Go right.', 'MOVE RIGHT', 'critical'

    # Neither side is fully safe — check partial clearance
    if left_occ < right_occ and left_occ < BLOCKED_THRESHOLD:
        return (f'{obs_text} {pos_text}. Carefully go left.',
                'MOVE LEFT', 'critical')

    if right_occ < left_occ and right_occ < BLOCKED_THRESHOLD:
        return (f'{obs_text} {pos_text}. Carefully go right.',
                'MOVE RIGHT', 'critical')

    # Completely blocked
    return f'{obs_text} {pos_text}. Stop. No clear path.', 'STOP', 'emergency'
