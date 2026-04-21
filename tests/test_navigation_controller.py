"""Tests for NavigationLogicController — smoothing, hysteresis, escape validation."""

import pytest
from nav_assist.navigation_controller import NavigationLogicController


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def feed_n(ctrl, left, center, right, n=5):
    """Push the same scores for n frames, return last result."""
    for _ in range(n):
        result = ctrl.update(left, center, right)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 1. Rolling Average Smoothing
# ═══════════════════════════════════════════════════════════════════════════

class TestSmoothing:
    def test_single_frame_spike_dampened(self):
        """One noisy frame shouldn't flip the instruction."""
        ctrl = NavigationLogicController(window_size=5)
        # Fill with safe frames
        feed_n(ctrl, 0.1, 0.1, 0.1, n=4)
        # One spike frame
        instr, state, sm = ctrl.update(0.1, 0.90, 0.1)
        # Smoothed center: (0.1*4 + 0.9)/5 = 0.26 → still SAFE
        assert state == 'SAFE'
        assert sm['center'] == pytest.approx(0.26, abs=0.01)

    def test_sustained_high_triggers_state(self):
        """Consistent high scores should transition state."""
        ctrl = NavigationLogicController(window_size=5)
        instr, state, _ = feed_n(ctrl, 0.1, 0.60, 0.1, n=5)
        # Smoothed center = 0.60 → CRITICAL
        assert state == 'CRITICAL'

    def test_smoothed_values_correct(self):
        ctrl = NavigationLogicController(window_size=3)
        ctrl.update(0.0, 0.2, 0.0)
        ctrl.update(0.0, 0.5, 0.0)
        _, _, sm = ctrl.update(0.0, 0.8, 0.0)
        assert sm['center'] == pytest.approx(0.5, abs=0.01)

    def test_window_rolls_over(self):
        """Older frames drop out of the window."""
        ctrl = NavigationLogicController(window_size=3)
        ctrl.update(0.0, 0.9, 0.0)
        ctrl.update(0.0, 0.1, 0.0)
        ctrl.update(0.0, 0.1, 0.0)
        _, _, sm = ctrl.update(0.0, 0.1, 0.0)
        # Window now has [0.1, 0.1, 0.1], old 0.9 dropped
        assert sm['center'] == pytest.approx(0.1, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Multi-Tiered Thresholds
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholds:
    def test_safe_state(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, state, _ = ctrl.update(0.1, 0.10, 0.1)
        assert state == 'SAFE'
        assert instr == 'Clear Ahead'

    def test_warning_state(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, state, _ = ctrl.update(0.1, 0.40, 0.1)
        assert state == 'WARNING'
        assert 'Approaching' in instr

    def test_critical_state_with_right_escape(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, state, _ = ctrl.update(0.7, 0.65, 0.2)
        assert state == 'CRITICAL'
        assert instr == 'Move Right'

    def test_critical_state_with_left_escape(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, state, _ = ctrl.update(0.2, 0.65, 0.7)
        assert state == 'CRITICAL'
        assert instr == 'Move Left'

    def test_emergency_state(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, state, _ = ctrl.update(0.1, 0.90, 0.1)
        assert state == 'EMERGENCY'
        assert instr == 'Stop'


# ═══════════════════════════════════════════════════════════════════════════
# 3. Hysteresis (Debouncing)
# ═══════════════════════════════════════════════════════════════════════════

class TestHysteresis:
    def test_no_flicker_at_warning_boundary(self):
        """Score oscillating around 0.35 should not toggle states."""
        ctrl = NavigationLogicController(window_size=1)
        # Enter WARNING
        ctrl.update(0.0, 0.36, 0.0)
        assert ctrl.state == 'WARNING'
        # Drop slightly below enter threshold but above exit (0.30)
        ctrl.update(0.0, 0.33, 0.0)
        assert ctrl.state == 'WARNING'  # stays WARNING, not SAFE

    def test_drops_to_safe_below_exit(self):
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.40, 0.0)
        assert ctrl.state == 'WARNING'
        ctrl.update(0.0, 0.28, 0.0)
        assert ctrl.state == 'SAFE'

    def test_no_flicker_at_critical_boundary(self):
        ctrl = NavigationLogicController(window_size=1)
        # Enter CRITICAL
        ctrl.update(0.0, 0.60, 0.0)
        assert ctrl.state == 'CRITICAL'
        # Drop below enter (0.55) but above exit (0.45)
        ctrl.update(0.0, 0.50, 0.0)
        assert ctrl.state == 'CRITICAL'

    def test_drops_to_warning_below_critical_exit(self):
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.60, 0.0)
        assert ctrl.state == 'CRITICAL'
        ctrl.update(0.0, 0.40, 0.0)
        assert ctrl.state == 'WARNING'

    def test_no_flicker_at_emergency_boundary(self):
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.90, 0.0)
        assert ctrl.state == 'EMERGENCY'
        ctrl.update(0.0, 0.80, 0.0)
        assert ctrl.state == 'EMERGENCY'  # above exit (0.75)

    def test_drops_from_emergency_below_exit(self):
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.90, 0.0)
        assert ctrl.state == 'EMERGENCY'
        ctrl.update(0.0, 0.70, 0.0)
        assert ctrl.state == 'CRITICAL'

    def test_multi_level_climb(self):
        """Score rising steadily should walk through all states."""
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.10, 0.0)
        assert ctrl.state == 'SAFE'
        ctrl.update(0.0, 0.40, 0.0)
        assert ctrl.state == 'WARNING'
        ctrl.update(0.0, 0.60, 0.0)
        assert ctrl.state == 'CRITICAL'
        ctrl.update(0.0, 0.90, 0.0)
        assert ctrl.state == 'EMERGENCY'

    def test_multi_level_descent(self):
        """Score falling steadily should walk back down."""
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.90, 0.0)
        assert ctrl.state == 'EMERGENCY'
        ctrl.update(0.0, 0.70, 0.0)
        assert ctrl.state == 'CRITICAL'
        ctrl.update(0.0, 0.40, 0.0)
        assert ctrl.state == 'WARNING'
        ctrl.update(0.0, 0.25, 0.0)
        assert ctrl.state == 'SAFE'


# ═══════════════════════════════════════════════════════════════════════════
# 4. Escape Route Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestEscapeValidation:
    def test_both_sides_safe_picks_clearer(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.30, 0.65, 0.15)
        assert instr == 'Move Right'

    def test_both_sides_safe_picks_left(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.10, 0.65, 0.30)
        assert instr == 'Move Left'

    def test_only_left_safe(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.20, 0.65, 0.60)
        assert instr == 'Move Left'

    def test_only_right_safe(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.60, 0.65, 0.20)
        assert instr == 'Move Right'

    def test_neither_side_safe(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.50, 0.65, 0.55)
        assert instr == 'Path Blocked. Stop'

    def test_side_exactly_at_threshold_is_blocked(self):
        """Score == safe_path_threshold (0.40) is NOT safe."""
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.40, 0.65, 0.40)
        assert instr == 'Path Blocked. Stop'

    def test_side_just_below_threshold_is_safe(self):
        ctrl = NavigationLogicController(window_size=1)
        instr, _, _ = ctrl.update(0.39, 0.65, 0.50)
        assert instr == 'Move Left'


# ═══════════════════════════════════════════════════════════════════════════
# 5. Reset & Custom Config
# ═══════════════════════════════════════════════════════════════════════════

class TestMisc:
    def test_reset_clears_state(self):
        ctrl = NavigationLogicController(window_size=1)
        ctrl.update(0.0, 0.90, 0.0)
        assert ctrl.state == 'EMERGENCY'
        ctrl.reset()
        assert ctrl.state == 'SAFE'

    def test_reset_clears_history(self):
        ctrl = NavigationLogicController(window_size=3)
        feed_n(ctrl, 0.0, 0.90, 0.0, n=3)
        ctrl.reset()
        # First frame after reset should only use that one frame
        _, _, sm = ctrl.update(0.0, 0.10, 0.0)
        assert sm['center'] == pytest.approx(0.10, abs=0.01)

    def test_custom_thresholds(self):
        custom = {
            'safe_to_warning': (0.50, 0.40),
            'warning_to_critical': (0.70, 0.60),
            'critical_to_emergency': (0.90, 0.80),
        }
        ctrl = NavigationLogicController(window_size=1, thresholds=custom)
        ctrl.update(0.0, 0.45, 0.0)
        assert ctrl.state == 'SAFE'  # custom threshold is 0.50
        ctrl.update(0.0, 0.55, 0.0)
        assert ctrl.state == 'WARNING'

    def test_custom_safe_path_threshold(self):
        ctrl = NavigationLogicController(window_size=1, safe_path_threshold=0.30)
        # Left=0.35 is above custom threshold, right=0.25 is below
        instr, _, _ = ctrl.update(0.35, 0.65, 0.25)
        assert instr == 'Move Right'

    def test_first_frame_works(self):
        """Controller should produce output even before the window is full."""
        ctrl = NavigationLogicController(window_size=5)
        instr, state, sm = ctrl.update(0.1, 0.1, 0.1)
        assert state == 'SAFE'
        assert instr == 'Clear Ahead'
