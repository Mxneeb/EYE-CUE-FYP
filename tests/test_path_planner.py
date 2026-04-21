"""Tests for the fuzzy-logic Path Planner module."""

import pytest
from nav_assist.path_planner import compute_zone_risk, plan_path


class TestFuzzyMembership:
    """Test that risk computation produces sensible outputs."""

    def test_no_obstacles_is_safe(self):
        risk, label = compute_zone_risk(density=0.0,
                                        max_disparity_normalised=0.0)
        assert label == 'safe'
        assert risk < 0.30

    def test_high_density_near_is_danger(self):
        risk, label = compute_zone_risk(density=0.5,
                                        max_disparity_normalised=0.9)
        assert label == 'danger'
        assert risk >= 0.65

    def test_medium_density_moderate_proximity_is_caution(self):
        risk, label = compute_zone_risk(density=0.15,
                                        max_disparity_normalised=0.5)
        assert label == 'caution'

    def test_risk_increases_with_density(self):
        risk_low, _ = compute_zone_risk(0.05, 0.5)
        risk_high, _ = compute_zone_risk(0.40, 0.5)
        assert risk_high > risk_low

    def test_risk_increases_with_proximity(self):
        risk_far, _ = compute_zone_risk(0.15, 0.2)
        risk_near, _ = compute_zone_risk(0.15, 0.9)
        assert risk_near > risk_far

    def test_risk_bounded_zero_one(self):
        for d in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            for p in [0.0, 0.2, 0.5, 0.8, 1.0]:
                risk, _ = compute_zone_risk(d, p)
                assert 0.0 <= risk <= 1.0, f"Risk {risk} out of bounds at d={d}, p={p}"


class TestPlanPath:
    """Test navigation decision logic."""

    def _make_zones(self, left_d=0.0, left_disp=0.0,
                    center_d=0.0, center_disp=0.0,
                    right_d=0.0, right_disp=0.0):
        """Helper to create zone dict."""
        return {
            'left':   {'density': left_d, 'obstacles': [],
                       'max_disparity': left_disp},
            'center': {'density': center_d, 'obstacles': [],
                       'max_disparity': center_disp},
            'right':  {'density': right_d, 'obstacles': [],
                       'max_disparity': right_disp},
        }

    def test_clear_path_go_straight(self):
        zones = self._make_zones()
        instruction, risks = plan_path(zones, max_global_disparity=1.0)
        assert 'STRAIGHT' in instruction

    def test_obstacle_left_go_right(self):
        zones = self._make_zones(
            left_d=0.5, left_disp=0.9,   # heavy left
            center_d=0.3, center_disp=0.7,
            right_d=0.0, right_disp=0.0,  # clear right
        )
        instruction, risks = plan_path(zones, max_global_disparity=1.0)
        assert 'RIGHT' in instruction

    def test_obstacle_right_go_left(self):
        zones = self._make_zones(
            left_d=0.0, left_disp=0.0,
            center_d=0.3, center_disp=0.7,
            right_d=0.5, right_disp=0.9,
        )
        instruction, risks = plan_path(zones, max_global_disparity=1.0)
        assert 'LEFT' in instruction

    def test_all_blocked_stop(self):
        zones = self._make_zones(
            left_d=0.5, left_disp=0.95,
            center_d=0.5, center_disp=0.95,
            right_d=0.5, right_disp=0.95,
        )
        instruction, risks = plan_path(zones, max_global_disparity=1.0)
        assert 'STOP' in instruction

    def test_zero_disparity_clear(self):
        zones = self._make_zones()
        instruction, risks = plan_path(zones, max_global_disparity=0.0)
        assert 'STRAIGHT' in instruction

    def test_center_caution(self):
        zones = self._make_zones(
            left_d=0.0, left_disp=0.0,
            center_d=0.15, center_disp=0.5,
            right_d=0.0, right_disp=0.0,
        )
        instruction, risks = plan_path(zones, max_global_disparity=1.0)
        # Center has some risk but left/right are clear
        # Should go left or right, or straight with caution
        assert any(word in instruction for word in ('LEFT', 'RIGHT', 'STRAIGHT'))
