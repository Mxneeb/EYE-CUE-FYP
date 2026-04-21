"""
Navigation Logic Controller — Stable, debounced navigation instructions.

Sits downstream of the PPM's per-sector OStatus scores and applies:
  1. Rolling-average smoothing  (ignores single-frame glitches)
  2. Multi-tiered threat levels (SAFE / WARNING / CRITICAL / EMERGENCY)
  3. Hysteresis on state transitions (prevents flickering)
  4. Absolute escape-route validation (never suggests an unsafe path)
"""

from collections import deque


# ── Tunable thresholds ────────────────────────────────────────────────────
# Each pair is (enter_threshold, exit_threshold).
# "enter" = score must exceed this to move UP to the next state.
# "exit"  = score must drop below this to fall BACK to the previous state.
# The gap between them is the hysteresis band.

DEFAULT_THRESHOLDS = {
    'safe_to_warning':     (0.35, 0.30),   # enter WARNING at 0.35, exit at 0.30
    'warning_to_critical': (0.55, 0.45),   # enter CRITICAL at 0.55, exit at 0.45
    'critical_to_emergency': (0.85, 0.75), # enter EMERGENCY at 0.85, exit at 0.75
}

# An escape route (Left or Right) is only valid if its smoothed
# score is strictly below this value.
DEFAULT_SAFE_PATH_THRESHOLD = 0.40

# Number of past frames to average over.
DEFAULT_WINDOW_SIZE = 5


class NavigationLogicController:
    """
    Consumes raw per-frame zone scores (left, center, right) and
    produces a stable navigation instruction string.

    Parameters
    ----------
    window_size : int
        Rolling-average window length (frames).  Larger = smoother
        but slower to react.  5 is a good starting point for 10-15 fps.
    thresholds : dict, optional
        Override the enter/exit pairs for each state transition.
        Keys: 'safe_to_warning', 'warning_to_critical',
              'critical_to_emergency'.
        Values: (enter_float, exit_float).
    safe_path_threshold : float
        Maximum smoothed score for a side zone to be considered a
        viable escape route.
    """

    # Possible center-zone states (ordered by severity)
    SAFE = 'SAFE'
    WARNING = 'WARNING'
    CRITICAL = 'CRITICAL'
    EMERGENCY = 'EMERGENCY'

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE,
                 thresholds=None,
                 safe_path_threshold=DEFAULT_SAFE_PATH_THRESHOLD):
        self.window_size = window_size
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.safe_path_threshold = safe_path_threshold

        # Rolling queues for each zone
        self._left_q = deque(maxlen=window_size)
        self._center_q = deque(maxlen=window_size)
        self._right_q = deque(maxlen=window_size)

        # Current hysteresis state for the center zone
        self._state = self.SAFE

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, left, center, right):
        """
        Feed one frame's raw zone scores and get back a navigation
        instruction.

        Parameters
        ----------
        left : float   (0..1)  OStatus of the left zone(s).
        center : float (0..1)  OStatus of the center zone(s).
        right : float  (0..1)  OStatus of the right zone(s).

        Returns
        -------
        instruction : str
            Human-readable navigation instruction.
        state : str
            One of 'SAFE', 'WARNING', 'CRITICAL', 'EMERGENCY'.
        smoothed : dict
            {'left': float, 'center': float, 'right': float} after
            rolling average — useful for debugging / overlay display.
        """
        # 1. Push raw scores and compute rolling averages
        self._left_q.append(left)
        self._center_q.append(center)
        self._right_q.append(right)

        sm_left = sum(self._left_q) / len(self._left_q)
        sm_center = sum(self._center_q) / len(self._center_q)
        sm_right = sum(self._right_q) / len(self._right_q)

        smoothed = {'left': sm_left, 'center': sm_center, 'right': sm_right}

        # 2. Determine center state with hysteresis
        self._state = self._next_state(sm_center)

        # 3. Map state to instruction
        instruction = self._decide(self._state, sm_left, sm_right)

        return instruction, self._state, smoothed

    def reset(self):
        """Clear history (e.g. on scene change or camera restart)."""
        self._left_q.clear()
        self._center_q.clear()
        self._right_q.clear()
        self._state = self.SAFE

    @property
    def state(self):
        """Current hysteresis state (read-only)."""
        return self._state

    # ── Internals ─────────────────────────────────────────────────────────

    def _next_state(self, sm_center):
        """
        Apply hysteresis to decide the next state.

        Transitions UP use the 'enter' threshold; transitions DOWN use
        the lower 'exit' threshold.  This prevents rapid flickering
        around a single boundary.

        Loops until stable so a large score jump (e.g. SAFE → CRITICAL)
        is handled in one call.
        """
        sw_enter, sw_exit = self.thresholds['safe_to_warning']
        wc_enter, wc_exit = self.thresholds['warning_to_critical']
        ce_enter, ce_exit = self.thresholds['critical_to_emergency']

        state = self._state

        while True:
            prev = state
            if state == self.SAFE:
                if sm_center >= sw_enter:
                    state = self.WARNING
            elif state == self.WARNING:
                if sm_center < sw_exit:
                    state = self.SAFE
                elif sm_center >= wc_enter:
                    state = self.CRITICAL
            elif state == self.CRITICAL:
                if sm_center < wc_exit:
                    state = self.WARNING
                elif sm_center >= ce_enter:
                    state = self.EMERGENCY
            elif state == self.EMERGENCY:
                if sm_center < ce_exit:
                    state = self.CRITICAL
            if state == prev:
                break

        return state

    def _decide(self, state, sm_left, sm_right):
        """
        Produce a navigation instruction from the current state and
        the smoothed side-zone scores.
        """
        if state == self.SAFE:
            return 'Clear Ahead'

        if state == self.WARNING:
            return 'Caution. Obstacle Approaching Ahead'

        if state == self.EMERGENCY:
            return 'Stop'

        # ── CRITICAL: pick a validated escape route ───────────────────
        threshold = self.safe_path_threshold
        left_ok = sm_left < threshold
        right_ok = sm_right < threshold

        if left_ok and right_ok:
            # Both are safe — pick the clearer one
            if sm_right <= sm_left:
                return 'Move Right'
            else:
                return 'Move Left'
        elif right_ok:
            return 'Move Right'
        elif left_ok:
            return 'Move Left'
        else:
            # Neither side is safe enough
            return 'Path Blocked. Stop'
