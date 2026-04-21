"""Tests for the Obstacle Detection Module (ODM)."""

import numpy as np
import pytest

from nav_assist.obstacle import detect_obstacles, get_zone_obstacles
from nav_assist.config import PATH_CLASS_INDICES, ADE20K_CLASS_TO_IDX


class TestDetectObstacles:
    """Tests for detect_obstacles() — Algorithm 1 from the paper."""

    def _make_inputs(self, h=100, w=150):
        """Create blank seg_mask and depth_map of given size."""
        seg = np.zeros((h, w), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        return seg, depth

    def test_empty_scene_returns_no_obstacles(self):
        """No objects, no depth -> no obstacles."""
        seg, depth = self._make_inputs()
        obs_bgr, obs_mask, obs_info = detect_obstacles(seg, depth)

        assert obs_bgr.shape == (100, 150, 3)
        assert obs_mask.shape == (100, 150)
        assert obs_mask.sum() == 0
        assert len(obs_info) == 0

    def test_nearby_object_detected(self):
        """A large object close to the camera should be detected."""
        seg, depth = self._make_inputs()

        # Place a 'chair' (class 19) in the center
        chair_id = ADE20K_CLASS_TO_IDX['chair']
        seg[30:70, 50:100] = chair_id

        # High disparity = close object
        depth[:] = 0.3    # background far
        depth[30:70, 50:100] = 1.0   # chair very close

        obs_bgr, obs_mask, obs_info = detect_obstacles(seg, depth)

        assert obs_mask.sum() > 0, "Nearby chair should be detected"
        assert len(obs_info) >= 1
        assert obs_info[0]['class_name'] == 'chair'

    def test_far_object_not_detected(self):
        """An object far from the camera (low disparity) should be discarded."""
        seg, depth = self._make_inputs()

        chair_id = ADE20K_CLASS_TO_IDX['chair']
        seg[30:70, 50:100] = chair_id

        # Low disparity everywhere — object is far away
        depth[:] = 0.5
        depth[30:70, 50:100] = 0.55  # barely above background

        obs_bgr, obs_mask, obs_info = detect_obstacles(
            seg, depth, threshold_ratio=0.60)

        # 0.55 / 0.55 * 0.60 = 0.33 threshold; 0.55 > 0.33 so it should be detected
        # Actually let's recalculate: max disparity = 0.55, threshold = 0.60 * 0.55 = 0.33
        # component max = 0.55 > 0.33, so it IS detected
        # Let me adjust: make the object disparity below threshold
        depth[:] = 1.0      # background close (max disp = 1.0)
        depth[30:70, 50:100] = 0.3  # object far

        obs_bgr, obs_mask, obs_info = detect_obstacles(
            seg, depth, threshold_ratio=0.60)

        # threshold = 0.60 * 1.0 = 0.6. Object disp = 0.3 < 0.6 => discarded
        assert obs_mask[30:70, 50:100].sum() == 0, "Far object should be discarded"

    def test_path_class_discarded(self):
        """Walkable surfaces (floor, road, etc.) should be excluded."""
        seg, depth = self._make_inputs()

        floor_id = ADE20K_CLASS_TO_IDX['floor']
        seg[50:100, :] = floor_id

        # Floor is close
        depth[:] = 0.2
        depth[50:100, :] = 1.0

        obs_bgr, obs_mask, obs_info = detect_obstacles(seg, depth)

        # Floor is a path class — should not appear as obstacle
        assert floor_id in PATH_CLASS_INDICES
        assert obs_mask[50:100, :].sum() == 0, "Floor should not be an obstacle"

    def test_small_components_filtered(self):
        """Tiny components below min_area should be ignored."""
        seg, depth = self._make_inputs()

        chair_id = ADE20K_CLASS_TO_IDX['chair']
        seg[10:13, 10:13] = chair_id  # 3x3 = 9 pixels
        depth[10:13, 10:13] = 1.0
        depth[:] = np.where(depth == 0, 0.1, depth)

        obs_bgr, obs_mask, obs_info = detect_obstacles(
            seg, depth, min_area=50)

        assert obs_mask.sum() == 0, "Tiny component should be filtered"

    def test_multiple_obstacles_detected(self):
        """Multiple separate objects should each be detected."""
        seg, depth = self._make_inputs(h=100, w=200)

        chair_id = ADE20K_CLASS_TO_IDX['chair']
        table_id = ADE20K_CLASS_TO_IDX['table']

        seg[20:40, 10:50] = chair_id
        seg[60:80, 120:170] = table_id

        depth[:] = 0.1
        depth[20:40, 10:50] = 1.0
        depth[60:80, 120:170] = 0.9

        obs_bgr, obs_mask, obs_info = detect_obstacles(seg, depth)

        assert len(obs_info) >= 2, "Should detect both chair and table"
        names = {o['class_name'] for o in obs_info}
        assert 'chair' in names
        assert 'table' in names

    def test_obstacle_info_sorted_by_disparity(self):
        """Obstacles should be sorted nearest-first (highest disparity)."""
        seg, depth = self._make_inputs(h=100, w=200)

        chair_id = ADE20K_CLASS_TO_IDX['chair']
        table_id = ADE20K_CLASS_TO_IDX['table']

        seg[20:40, 10:50] = chair_id
        seg[60:80, 120:170] = table_id

        depth[:] = 0.1
        depth[20:40, 10:50] = 0.8    # chair: closer
        depth[60:80, 120:170] = 1.0  # table: nearest

        obs_bgr, obs_mask, obs_info = detect_obstacles(seg, depth)

        assert len(obs_info) >= 2
        assert obs_info[0]['disparity'] >= obs_info[1]['disparity']

    def test_depth_seg_size_mismatch_handled(self):
        """ODM should handle depth and seg having different resolutions."""
        seg = np.zeros((100, 150), dtype=np.uint8)
        depth = np.zeros((200, 300), dtype=np.float32)  # different size

        chair_id = ADE20K_CLASS_TO_IDX['chair']
        seg[30:70, 50:100] = chair_id
        depth[:] = 0.1
        depth[60:140, 100:200] = 1.0  # scaled region

        obs_bgr, obs_mask, obs_info = detect_obstacles(seg, depth)

        # Should not crash; output matches seg resolution
        assert obs_bgr.shape == (100, 150, 3)
        assert obs_mask.shape == (100, 150)


class TestGetZoneObstacles:
    """Tests for get_zone_obstacles()."""

    def test_empty_scene(self):
        obstacle_mask = np.zeros((100, 150), dtype=bool)
        zones = get_zone_obstacles(obstacle_mask, [], 150)

        assert zones['left']['density'] == 0.0
        assert zones['center']['density'] == 0.0
        assert zones['right']['density'] == 0.0

    def test_obstacle_in_left_zone(self):
        obstacle_mask = np.zeros((100, 150), dtype=bool)
        obstacle_mask[20:80, 10:40] = True

        obstacle_info = [{
            'class_id': 19, 'class_name': 'chair',
            'disparity': 0.9,
            'bbox': (10, 20, 40, 80), 'area': 1800,
            'centroid': (25, 50),  # x=25, in left zone (0-50)
        }]

        zones = get_zone_obstacles(obstacle_mask, obstacle_info, 150)

        assert zones['left']['density'] > 0
        assert zones['center']['density'] == 0
        assert zones['right']['density'] == 0
        assert len(zones['left']['obstacles']) == 1

    def test_all_three_zones(self):
        obstacle_mask = np.zeros((100, 300), dtype=bool)
        obstacle_mask[10:30, 10:30] = True    # left
        obstacle_mask[40:60, 130:170] = True  # center
        obstacle_mask[70:90, 250:280] = True  # right

        obstacle_info = [
            {'class_id': 19, 'class_name': 'chair', 'disparity': 0.9,
             'bbox': (10, 10, 30, 30), 'area': 400, 'centroid': (20, 20)},
            {'class_id': 15, 'class_name': 'table', 'disparity': 0.8,
             'bbox': (130, 40, 170, 60), 'area': 800, 'centroid': (150, 50)},
            {'class_id': 23, 'class_name': 'sofa', 'disparity': 0.7,
             'bbox': (250, 70, 280, 90), 'area': 600, 'centroid': (265, 80)},
        ]

        zones = get_zone_obstacles(obstacle_mask, obstacle_info, 300)

        assert zones['left']['density'] > 0
        assert zones['center']['density'] > 0
        assert zones['right']['density'] > 0
