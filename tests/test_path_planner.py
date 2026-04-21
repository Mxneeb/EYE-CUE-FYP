"""Tests for the fuzzy-logic Path Planner module (paper Fig. 5-10)."""

import numpy as np
import pytest

from nav_assist.path_planner import (
    compute_alert_zone, compute_sector_bounds, compute_ostatus,
    find_prominent_obstacle, mu_free, mu_blocked,
    evaluate_rules, defuzzify, classify_action, plan_path,
)
from nav_assist.config import ADE20K_CLASS_TO_IDX


# ════════════════════════════════════════════════════════════════════════════
# Alert Zone (Fig. 5)
# ════════════════════════════════════════════════════════════════════════════

class TestAlertZone:

    def test_portrait_margins(self):
        y_start, x_start, x_end = compute_alert_zone(h=480, w=320)
        assert y_start == int(0.60 * 480)
        assert x_start == int(0.20 * 320)
        assert x_end == int(0.80 * 320)

    def test_landscape_margins(self):
        y_start, x_start, x_end = compute_alert_zone(h=480, w=640)
        assert y_start == int(0.60 * 480)
        assert x_start == int(0.30 * 640)
        assert x_end == int(0.70 * 640)

    def test_square_uses_landscape(self):
        y_start, x_start, x_end = compute_alert_zone(h=500, w=500)
        assert x_start == int(0.30 * 500)


# ════════════════════════════════════════════════════════════════════════════
# 6-Sector Grid (Fig. 7)
# ════════════════════════════════════════════════════════════════════════════

class TestSectorBounds:

    def test_six_sectors_returned(self):
        bounds = compute_sector_bounds(100, 300)
        assert len(bounds) == 6
        expected = {'top_left', 'top_mid', 'top_right',
                    'bot_left', 'bot_mid', 'bot_right'}
        assert set(bounds.keys()) == expected

    def test_sectors_cover_image(self):
        h, w = 100, 300
        bounds = compute_sector_bounds(h, w)
        covered = np.zeros((h, w), dtype=bool)
        for y0, y1, x0, x1 in bounds.values():
            covered[y0:y1, x0:x1] = True
        assert covered.all()

    def test_top_bottom_split_at_half(self):
        bounds = compute_sector_bounds(200, 300)
        assert bounds['top_mid'][1] == 100
        assert bounds['bot_mid'][0] == 100


# ════════════════════════════════════════════════════════════════════════════
# OStatus + Sector Labels
# ════════════════════════════════════════════════════════════════════════════

class TestOStatus:

    def test_empty_scene(self):
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)
        bounds = compute_sector_bounds(h, w)

        ostatus, sector_labels = compute_ostatus(mask, labels, bounds)

        for name in ostatus:
            assert ostatus[name] == 0.0
            assert sector_labels[name] == {}

    def test_full_obstacle_in_one_sector(self):
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)
        bounds = compute_sector_bounds(h, w)

        # Fill bot_mid entirely with obstacles
        y0, y1, x0, x1 = bounds['bot_mid']
        mask[y0:y1, x0:x1] = True
        labels[y0:y1, x0:x1] = 19  # chair

        ostatus, sector_labels = compute_ostatus(mask, labels, bounds)

        assert ostatus['bot_mid'] == pytest.approx(1.0)
        assert ostatus['bot_left'] == 0.0
        assert 19 in sector_labels['bot_mid']


# ════════════════════════════════════════════════════════════════════════════
# Prominent Obstacle (Fig. 7)
# ════════════════════════════════════════════════════════════════════════════

class TestProminentObstacle:

    def test_no_obstacles_returns_none(self):
        sector_labels = {s: {} for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        name, cls_id, pos = find_prominent_obstacle(sector_labels)
        assert name is None
        assert cls_id == -1

    def test_ahead_when_bottom_dominates(self):
        sector_labels = {s: {} for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        sector_labels['bot_mid'] = {12: 500}  # person, 500 pixels
        sector_labels['top_mid'] = {12: 100}

        name, cls_id, pos = find_prominent_obstacle(sector_labels)
        assert name == 'person'
        assert cls_id == 12
        assert pos == 'ahead'

    def test_overhead_when_top_dominates(self):
        sector_labels = {s: {} for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        sector_labels['top_mid'] = {4: 800}  # tree
        sector_labels['bot_mid'] = {4: 100}

        name, cls_id, pos = find_prominent_obstacle(sector_labels)
        assert name == 'tree'
        assert pos == 'overhead'


# ════════════════════════════════════════════════════════════════════════════
# Trapezoidal Membership Functions (Fig. 9)
# ════════════════════════════════════════════════════════════════════════════

class TestFuzzyMembership:

    def test_mu_free_at_zero(self):
        assert mu_free(0.0) == 1.0

    def test_mu_free_at_high(self):
        assert mu_free(0.5) == 0.0

    def test_mu_blocked_at_zero(self):
        assert mu_blocked(0.0) == 0.0

    def test_mu_blocked_at_high(self):
        assert mu_blocked(0.7) == 1.0

    def test_mu_free_transition(self):
        val = mu_free(0.225)  # midpoint of falloff (0.15..0.30)
        assert 0.0 < val < 1.0

    def test_mu_blocked_transition(self):
        val = mu_blocked(0.40)  # midpoint of rise (0.30..0.50)
        assert 0.0 < val < 1.0

    def test_mu_values_bounded(self):
        for v in np.linspace(0, 1, 50):
            assert 0.0 <= mu_free(v) <= 1.0
            assert 0.0 <= mu_blocked(v) <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# Fuzzy Rule Evaluation (Fig. 9)
# ════════════════════════════════════════════════════════════════════════════

class TestRuleEvaluation:

    def test_clear_path_fires_ahead(self):
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        rules = evaluate_rules(ostatus)
        assert rules['move_ahead'] > 0.5
        assert rules['move_left'] < 0.1
        assert rules['move_right'] < 0.1

    def test_right_blocked_fires_left(self):
        """bot = [0.0, 0.36, 0.9] -> should move left (per Fig. 9)."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['bot_left'] = 0.0
        ostatus['bot_mid'] = 0.36
        ostatus['bot_right'] = 0.9

        rules = evaluate_rules(ostatus)
        assert rules['move_left'] > rules['move_right']

    def test_left_blocked_fires_right(self):
        """bot = [0.49, 0.29, 0.0] -> should move right (per Fig. 9)."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['bot_left'] = 0.49
        ostatus['bot_mid'] = 0.29
        ostatus['bot_right'] = 0.0

        rules = evaluate_rules(ostatus)
        assert rules['move_right'] > rules['move_left']

    def test_center_free_fires_ahead(self):
        """bot = [0.13, 0.0, 0.60] -> should move ahead (per Fig. 9)."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['bot_left'] = 0.13
        ostatus['bot_mid'] = 0.0
        ostatus['bot_right'] = 0.60

        rules = evaluate_rules(ostatus)
        assert rules['move_ahead'] > rules['move_left']
        assert rules['move_ahead'] > rules['move_right']


# ════════════════════════════════════════════════════════════════════════════
# Defuzzification & Action Classification
# ════════════════════════════════════════════════════════════════════════════

class TestDefuzzify:

    def test_pure_ahead(self):
        centroid = defuzzify({'move_ahead': 1.0, 'move_left': 0.0,
                              'move_right': 0.0})
        assert -0.20 <= centroid <= 0.20

    def test_pure_left(self):
        centroid = defuzzify({'move_ahead': 0.0, 'move_left': 1.0,
                              'move_right': 0.0})
        assert centroid < -0.20

    def test_pure_right(self):
        centroid = defuzzify({'move_ahead': 0.0, 'move_left': 0.0,
                              'move_right': 1.0})
        assert centroid > 0.20

    def test_all_zero_returns_zero(self):
        centroid = defuzzify({'move_ahead': 0.0, 'move_left': 0.0,
                              'move_right': 0.0})
        assert centroid == 0.0


class TestClassifyAction:

    def test_left(self):
        assert classify_action(-0.5) == 'MOVE LEFT'

    def test_right(self):
        assert classify_action(0.5) == 'MOVE RIGHT'

    def test_ahead(self):
        assert classify_action(0.0) == 'MOVE AHEAD'

    def test_boundary_left(self):
        assert classify_action(-0.20) == 'MOVE AHEAD'

    def test_boundary_right(self):
        assert classify_action(0.20) == 'MOVE AHEAD'


# ════════════════════════════════════════════════════════════════════════════
# Integration: plan_path()
# ════════════════════════════════════════════════════════════════════════════

class TestPlanPath:

    def _make_inputs(self, h=100, w=300):
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)
        return mask, labels

    def test_clear_scene_move_ahead(self):
        mask, labels = self._make_inputs()
        instruction, details = plan_path(mask, labels)

        assert 'MOVE AHEAD' in instruction
        assert 'path clear' in instruction
        assert details['action'] == 'MOVE AHEAD'

    def test_obstacle_right_move_left(self):
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)

        # Fill bot_right with obstacles
        mask[50:100, 200:300] = True
        labels[50:100, 200:300] = 12  # person

        instruction, details = plan_path(mask, labels)
        assert details['action'] in ('MOVE LEFT', 'MOVE AHEAD')

    def test_obstacle_left_move_right(self):
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)

        # Fill bot_left with obstacles
        mask[50:100, 0:100] = True
        labels[50:100, 0:100] = 19  # chair

        instruction, details = plan_path(mask, labels)
        assert details['action'] in ('MOVE RIGHT', 'MOVE AHEAD')

    def test_all_blocked_stop(self):
        h, w = 100, 300
        mask = np.ones((h, w), dtype=bool)
        labels = np.full((h, w), 12, dtype=np.int16)  # person everywhere

        instruction, details = plan_path(mask, labels)
        assert details['action'] == 'STOP'
        assert 'STOP' in instruction

    def test_prominent_obstacle_in_instruction(self):
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)

        # Put a person in bot_mid (prominent mid-sector obstacle)
        mask[50:100, 100:200] = True
        labels[50:100, 100:200] = 12  # person

        instruction, details = plan_path(mask, labels)
        assert details['prominent_obstacle'] == 'person'
        assert 'person' in instruction

    def test_details_contains_expected_keys(self):
        mask, labels = self._make_inputs()
        _, details = plan_path(mask, labels)

        expected_keys = {'action', 'prominent_obstacle', 'position',
                         'ostatus', 'sector_labels', 'rule_strengths',
                         'centroid'}
        assert expected_keys == set(details.keys())

    def test_ostatus_has_six_sectors(self):
        mask, labels = self._make_inputs()
        _, details = plan_path(mask, labels)

        assert len(details['ostatus']) == 6

    def test_position_ahead_or_overhead(self):
        h, w = 100, 300
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)

        mask[50:100, 100:200] = True
        labels[50:100, 100:200] = 19

        _, details = plan_path(mask, labels)
        assert details['position'] in ('ahead', 'overhead')
