"""Tests for the fuzzy-logic Path Planner module (PPM)."""

import numpy as np
import pytest

from nav_assist.path_planner import (
    compute_alert_zone, compute_sector_bounds, compute_ostatus,
    find_prominent_obstacle, mu_free, mu_blocked,
    evaluate_rules, defuzzify, classify_action, plan_path,
    create_depth_gated_mask,
)
from nav_assist.config import ADE20K_CLASS_TO_IDX, PATH_CLASS_INDICES


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
# Depth-Gated Obstacle Masking
# ════════════════════════════════════════════════════════════════════════════

class TestDepthGatedMask:

    def test_floor_excluded(self):
        """Walkable surfaces (floor = class 3) must be zeroed."""
        h, w = 100, 200
        seg = np.full((h, w), 3, dtype=np.uint8)   # floor everywhere
        depth = np.ones((h, w), dtype=np.float32) * 100
        mask, labels = create_depth_gated_mask(seg, depth)
        assert mask.sum() == 0

    def test_near_object_included(self):
        """A near non-walkable object must appear in the mask."""
        h, w = 100, 200
        seg = np.full((h, w), 3, dtype=np.uint8)   # floor background
        seg[40:60, 80:120] = 12                      # person patch
        depth = np.ones((h, w), dtype=np.float32) * 100
        mask, labels = create_depth_gated_mask(seg, depth)
        assert mask[50, 100]            # centre of person
        assert labels[50, 100] == 12

    def test_far_object_excluded(self):
        """An object beyond the depth threshold must be filtered out."""
        h, w = 100, 200
        seg = np.full((h, w), 3, dtype=np.uint8)
        seg[40:60, 80:120] = 12                      # person
        depth = np.ones((h, w), dtype=np.float32) * 100
        depth[40:60, 80:120] = 1.0                   # very far away
        mask, _ = create_depth_gated_mask(seg, depth)
        assert not mask[50, 100]

    def test_depth_resolution_mismatch_handled(self):
        """Depth map at different resolution should be resized."""
        seg = np.full((480, 640), 12, dtype=np.uint8)   # person
        depth = np.ones((240, 320), dtype=np.float32) * 100
        mask, labels = create_depth_gated_mask(seg, depth)
        assert mask.shape == (480, 640)
        assert mask.any()

    def test_zero_depth_returns_empty(self):
        """Completely dark / zero depth → no obstacles."""
        seg = np.full((100, 200), 12, dtype=np.uint8)
        depth = np.zeros((100, 200), dtype=np.float32)
        mask, labels = create_depth_gated_mask(seg, depth)
        assert mask.sum() == 0


# ════════════════════════════════════════════════════════════════════════════
# 6-Sector Grid — Full Coverage
# ════════════════════════════════════════════════════════════════════════════

class TestSectorBounds:

    def test_six_sectors_returned(self):
        bounds = compute_sector_bounds(100, 300)
        assert len(bounds) == 6
        expected = {'top_left', 'top_mid', 'top_right',
                    'bot_left', 'bot_mid', 'bot_right'}
        assert set(bounds.keys()) == expected

    def test_overhead_region_top_half(self):
        """Overhead sectors span 0 → 0.5H (top half)."""
        bounds = compute_sector_bounds(480, 640)
        assert bounds['top_mid'][0] == 0               # y0
        assert bounds['top_mid'][1] == int(0.50 * 480) # y1 = 240

    def test_ground_region_bottom_half(self):
        """Ground sectors span 0.5H → H (bottom half)."""
        bounds = compute_sector_bounds(480, 640)
        assert bounds['bot_mid'][0] == int(0.50 * 480) # y0 = 240
        assert bounds['bot_mid'][1] == 480              # y1 = H

    def test_no_vertical_gap(self):
        """Overhead bottom == Ground top: full vertical coverage."""
        bounds = compute_sector_bounds(480, 640)
        assert bounds['top_mid'][1] == bounds['bot_mid'][0]

    def test_column_widths_30_40_30(self):
        """Columns are 30%/40%/30% of W."""
        bounds = compute_sector_bounds(480, 640)
        assert bounds['top_left'][2] == 0
        assert bounds['top_left'][3] == int(0.30 * 640)
        assert bounds['top_mid'][2] == int(0.30 * 640)
        assert bounds['top_mid'][3] == int(0.70 * 640)
        assert bounds['top_right'][2] == int(0.70 * 640)
        assert bounds['top_right'][3] == 640

    def test_full_image_coverage(self):
        """All sectors combined cover the entire image (no blind spots)."""
        h, w = 480, 640
        bounds = compute_sector_bounds(h, w)
        covered = np.zeros((h, w), dtype=bool)
        for y0, y1, x0, x1 in bounds.values():
            covered[y0:y1, x0:x1] = True
        assert covered.all()

    def test_sectors_do_not_overlap(self):
        h, w = 480, 640
        bounds = compute_sector_bounds(h, w)
        canvas = np.zeros((h, w), dtype=np.int32)
        for y0, y1, x0, x1 in bounds.values():
            assert (canvas[y0:y1, x0:x1] == 0).all(), "Sectors overlap"
            canvas[y0:y1, x0:x1] = 1


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
        h, w = 480, 640
        mask = np.zeros((h, w), dtype=bool)
        labels = np.full((h, w), -1, dtype=np.int16)
        bounds = compute_sector_bounds(h, w)

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
        sector_labels['bot_mid'] = {12: 500}
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
# Trapezoidal Membership Functions — Overlapping
# ════════════════════════════════════════════════════════════════════════════

class TestFuzzyMembership:

    def test_mu_free_at_zero(self):
        assert mu_free(0.0) == 1.0

    def test_mu_free_at_010(self):
        """Fully free at v = 0.10 (upper edge of plateau)."""
        assert mu_free(0.10) == 1.0

    def test_mu_free_at_040(self):
        """Drops to zero at v = 0.40."""
        assert mu_free(0.40) == 0.0

    def test_mu_blocked_at_zero(self):
        assert mu_blocked(0.0) == 0.0

    def test_mu_blocked_at_010(self):
        """Still zero at v = 0.10 (bottom of ramp)."""
        assert mu_blocked(0.10) == 0.0

    def test_mu_blocked_at_040(self):
        """Fully blocked at v = 0.40."""
        assert mu_blocked(0.40) == 1.0

    def test_crossover_at_025(self):
        """At v = 0.25 both MFs should return ~0.5."""
        assert mu_free(0.25) == pytest.approx(0.5)
        assert mu_blocked(0.25) == pytest.approx(0.5)

    def test_mu_values_bounded(self):
        for v in np.linspace(0, 1, 50):
            assert 0.0 <= mu_free(v) <= 1.0
            assert 0.0 <= mu_blocked(v) <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# Fuzzy Rule Evaluation
# ════════════════════════════════════════════════════════════════════════════

class TestRuleEvaluation:

    def test_clear_path_fires_ahead(self):
        """All sectors free → Move Ahead dominates."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        rules = evaluate_rules(ostatus)
        assert rules['move_ahead'] > 0.5
        assert rules['move_left'] < 0.1
        assert rules['move_right'] < 0.1

    def test_mid_and_right_blocked_fires_left(self):
        """Mid and right blocked, left free → Move Left."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['bot_mid'] = 0.7
        ostatus['bot_right'] = 0.9

        rules = evaluate_rules(ostatus)
        assert rules['move_left'] > rules['move_right']
        assert rules['move_left'] > rules['move_ahead']

    def test_mid_and_left_blocked_fires_right(self):
        """Mid and left blocked, right free → Move Right."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['bot_left'] = 0.8
        ostatus['bot_mid'] = 0.7

        rules = evaluate_rules(ostatus)
        assert rules['move_right'] > rules['move_left']
        assert rules['move_right'] > rules['move_ahead']

    def test_center_free_fires_ahead(self):
        """Both mid sectors free → Move Ahead regardless of sides."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['bot_left'] = 0.8
        ostatus['bot_right'] = 0.8

        rules = evaluate_rules(ostatus)
        assert rules['move_ahead'] > rules['move_left']
        assert rules['move_ahead'] > rules['move_right']

    def test_overhead_mid_blocked_triggers_avoidance(self):
        """Overhead obstacle in mid triggers avoidance."""
        ostatus = {s: 0.0 for s in [
            'top_left', 'top_mid', 'top_right',
            'bot_left', 'bot_mid', 'bot_right',
        ]}
        ostatus['top_mid'] = 0.8
        ostatus['top_left'] = 0.8  # block left to force right

        rules = evaluate_rules(ostatus)
        assert rules['move_right'] > rules['move_ahead']


# ════════════════════════════════════════════════════════════════════════════
# Defuzzification & Action Classification (5-level)
# ════════════════════════════════════════════════════════════════════════════

class TestDefuzzify:

    def test_pure_ahead(self):
        centroid = defuzzify({'move_ahead': 1.0, 'move_left': 0.0,
                              'move_right': 0.0})
        assert -0.15 <= centroid <= 0.15

    def test_pure_left(self):
        centroid = defuzzify({'move_ahead': 0.0, 'move_left': 1.0,
                              'move_right': 0.0})
        assert centroid < -0.15

    def test_pure_right(self):
        centroid = defuzzify({'move_ahead': 0.0, 'move_left': 0.0,
                              'move_right': 1.0})
        assert centroid > 0.15

    def test_all_zero_returns_zero(self):
        centroid = defuzzify({'move_ahead': 0.0, 'move_left': 0.0,
                              'move_right': 0.0})
        assert centroid == 0.0


class TestClassifyAction:

    def test_strong_left(self):
        assert classify_action(-0.5) == 'MOVE LEFT'

    def test_slight_left(self):
        assert classify_action(-0.25) == 'MOVE SLIGHT LEFT'

    def test_ahead(self):
        assert classify_action(0.0) == 'MOVE AHEAD'

    def test_slight_right(self):
        assert classify_action(0.25) == 'MOVE SLIGHT RIGHT'

    def test_strong_right(self):
        assert classify_action(0.5) == 'MOVE RIGHT'

    def test_boundary_ahead_left(self):
        """At exactly -0.15, should be Move Ahead (inclusive)."""
        assert classify_action(-0.15) == 'MOVE AHEAD'

    def test_boundary_ahead_right(self):
        """At exactly 0.15, should be Move Ahead (inclusive)."""
        assert classify_action(0.15) == 'MOVE AHEAD'

    def test_boundary_slight_left(self):
        """At exactly -0.40, should be Move Slight Left."""
        assert classify_action(-0.40) == 'MOVE SLIGHT LEFT'

    def test_boundary_slight_right(self):
        """At exactly 0.40, should be Move Slight Right."""
        assert classify_action(0.40) == 'MOVE SLIGHT RIGHT'


# ════════════════════════════════════════════════════════════════════════════
# Integration: plan_path(seg_mask, depth_map)
# ════════════════════════════════════════════════════════════════════════════

class TestPlanPath:
    """Integration tests using the full pipeline: seg_mask + depth_map."""

    def _clear_scene(self, h=480, w=640):
        """Floor everywhere, all near → no obstacles."""
        seg = np.full((h, w), 3, dtype=np.uint8)   # floor (path class)
        depth = np.ones((h, w), dtype=np.float32) * 100.0
        return seg, depth

    def _scene_with_obstacle(self, sectors, cls_id=12, h=480, w=640):
        """Floor background with a non-walkable object in named sectors."""
        seg, depth = self._clear_scene(h, w)
        bounds = compute_sector_bounds(h, w)
        for name in sectors:
            y0, y1, x0, x1 = bounds[name]
            seg[y0:y1, x0:x1] = cls_id
        return seg, depth

    # ── Basic navigation ───────────────────────────────────────────────

    def test_clear_scene_move_ahead(self):
        seg, depth = self._clear_scene()
        instruction, details = plan_path(seg, depth)
        assert details['action'] == 'MOVE AHEAD'
        assert instruction == 'Move Ahead'

    def test_obstacle_mid_and_right_move_left(self):
        seg, depth = self._scene_with_obstacle(('bot_mid', 'bot_right'))
        instruction, details = plan_path(seg, depth)
        assert 'LEFT' in details['action']

    def test_obstacle_mid_and_left_move_right(self):
        seg, depth = self._scene_with_obstacle(
            ('bot_mid', 'bot_left'), cls_id=19)  # chair
        instruction, details = plan_path(seg, depth)
        assert 'RIGHT' in details['action']

    def test_all_blocked_stop(self):
        h, w = 480, 640
        seg = np.full((h, w), 12, dtype=np.uint8)      # person everywhere
        depth = np.ones((h, w), dtype=np.float32) * 100
        instruction, details = plan_path(seg, depth)
        assert details['action'] == 'STOP'
        assert instruction == 'Stop'

    # ── Prominent obstacle & instruction format ────────────────────────

    def test_prominent_obstacle_identified(self):
        seg, depth = self._scene_with_obstacle(('bot_mid',), cls_id=12)
        instruction, details = plan_path(seg, depth)
        assert details['prominent_obstacle'] == 'person'
        assert 'Person' in instruction

    def test_instruction_format_with_obstacle(self):
        """Format: '{Obstacle} {position}. {Action}'."""
        seg, depth = self._scene_with_obstacle(
            ('bot_mid', 'bot_right'), cls_id=19)
        instruction, details = plan_path(seg, depth)
        assert 'Chair' in instruction
        assert 'Left' in instruction   # Move Left or Move Slight Left

    def test_overhead_obstacle_format(self):
        """Overhead obstacle shows 'overhead' in instruction."""
        seg, depth = self._scene_with_obstacle(
            ('top_mid', 'top_left'), cls_id=5)  # ceiling
        instruction, details = plan_path(seg, depth)
        assert 'overhead' in instruction
        assert 'Right' in instruction  # Move Right or Move Slight Right

    # ── Depth gating integration ───────────────────────────────────────

    def test_far_obstacles_ignored(self):
        """Objects beyond the depth gate should not affect navigation."""
        h, w = 480, 640
        seg = np.full((h, w), 12, dtype=np.uint8)      # person everywhere
        depth = np.ones((h, w), dtype=np.float32) * 1.0 # very far
        # Max disparity is 1.0, threshold = 0.4*1.0 = 0.4
        # All pixels have depth 1.0 ≥ 0.4 → passes depth gate!
        # This is correct: uniform disparity means everything is at same distance.
        # To truly test "far", set low disparity relative to max:
        depth[:] = 100.0
        bounds = compute_sector_bounds(h, w)
        for name in ('bot_mid', 'bot_right'):
            y0, y1, x0, x1 = bounds[name]
            depth[y0:y1, x0:x1] = 5.0   # far relative to max=100
        instruction, details = plan_path(seg, depth)
        # bot_mid and bot_right obstacles are far → filtered out
        # Remaining near obstacles in other sectors shouldn't trigger left/right
        assert details['ostatus']['bot_mid'] < 1.0

    # ── Details dict ───────────────────────────────────────────────────

    def test_details_contains_expected_keys(self):
        seg, depth = self._clear_scene()
        _, details = plan_path(seg, depth)
        expected_keys = {'action', 'prominent_obstacle', 'position',
                         'ostatus', 'sector_bounds', 'sector_labels',
                         'rule_strengths', 'centroid', 'valid_mask'}
        assert expected_keys == set(details.keys())

    def test_ostatus_has_six_sectors(self):
        seg, depth = self._clear_scene()
        _, details = plan_path(seg, depth)
        assert len(details['ostatus']) == 6

    def test_sector_bounds_in_details(self):
        seg, depth = self._clear_scene()
        _, details = plan_path(seg, depth)
        assert len(details['sector_bounds']) == 6

    def test_position_ahead_or_overhead(self):
        seg, depth = self._scene_with_obstacle(('bot_mid',), cls_id=19)
        _, details = plan_path(seg, depth)
        assert details['position'] in ('ahead', 'overhead')
