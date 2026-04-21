"""Tests for visualization helper functions."""

import numpy as np
import pytest

from nav_assist.visualization import colorize_depth, colorize_seg
from nav_assist.config import ADE20K_PALETTE, PANEL_W, PANEL_H


class TestColorizeDepth:
    """Tests for depth map colourisation."""

    def test_output_shape_matches_input(self):
        depth = np.random.rand(100, 150).astype(np.float32)
        coloured, norm = colorize_depth(depth)

        assert coloured.shape == (100, 150, 3)
        assert coloured.dtype == np.uint8
        assert norm.shape == (100, 150)

    def test_normalisation_range(self):
        depth = np.random.rand(50, 50).astype(np.float32) * 10 + 5
        _, norm = colorize_depth(depth)

        assert norm.min() >= 0.0 - 1e-6
        assert norm.max() <= 1.0 + 1e-6

    def test_uniform_depth_returns_zeros_norm(self):
        depth = np.ones((50, 50), dtype=np.float32) * 42.0
        _, norm = colorize_depth(depth)

        assert np.allclose(norm, 0.0), "Uniform depth should normalise to 0"

    def test_zero_depth_no_crash(self):
        depth = np.zeros((50, 50), dtype=np.float32)
        coloured, norm = colorize_depth(depth)

        assert coloured.shape == (50, 50, 3)


class TestColorizeSeg:
    """Tests for segmentation mask colourisation."""

    def test_output_shape(self):
        seg = np.zeros((64, 64), dtype=np.uint8)
        result = colorize_seg(seg, 100, 150)

        assert result.shape == (100, 150, 3)
        assert result.dtype == np.uint8

    def test_single_class_uniform_colour(self):
        seg = np.full((64, 64), 5, dtype=np.uint8)
        result = colorize_seg(seg, 64, 64)

        expected_rgb = ADE20K_PALETTE[5]
        expected_bgr = expected_rgb[::-1]
        # All pixels should have the same colour
        assert np.all(result[0, 0] == expected_bgr)

    def test_multiple_classes(self):
        seg = np.zeros((100, 100), dtype=np.uint8)
        seg[:50, :] = 0   # wall
        seg[50:, :] = 19  # chair

        result = colorize_seg(seg, 100, 100)

        # Top and bottom should have different colours
        assert not np.array_equal(result[10, 50], result[75, 50])

    def test_all_class_indices_valid(self):
        """Every valid class index (0-149) should produce a valid colour."""
        for class_id in range(150):
            seg = np.full((10, 10), class_id, dtype=np.uint8)
            result = colorize_seg(seg, 10, 10)
            assert result.shape == (10, 10, 3)


class TestColorizeConfig:
    """Verify config data integrity."""

    def test_palette_shape(self):
        assert ADE20K_PALETTE.shape == (150, 3)
        assert ADE20K_PALETTE.dtype == np.uint8

    def test_palette_values_valid(self):
        assert ADE20K_PALETTE.min() >= 0
        assert ADE20K_PALETTE.max() <= 255
