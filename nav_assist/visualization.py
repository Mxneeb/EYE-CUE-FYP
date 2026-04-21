"""
Visualization helpers: drawing utilities, panel builders, colourisation.
"""

import cv2
import numpy as np
from matplotlib import colormaps

from nav_assist.config import (
    ADE20K_CLASSES, ADE20K_PALETTE, PANEL_W, PANEL_H, STATUS_H, WIN_W,
)

_CMAP_SPECTRAL = colormaps.get_cmap('Spectral_r')


# ════════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ════════════════════════════════════════════════════════════════════════════

def put_text(img, text, pos, scale=0.55, color=(255, 255, 255),
             thickness=1, bg_color=(0, 0, 0)):
    """Draw text with a shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), font,
                scale, bg_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def put_label_box(img, text, x, y, color_bgr, font_scale=0.45):
    """Coloured legend box + label."""
    cv2.rectangle(img, (x, y), (x + 14, y + 14), color_bgr, -1)
    cv2.rectangle(img, (x, y), (x + 14, y + 14), (50, 50, 50), 1)
    put_text(img, text, (x + 18, y + 11), scale=font_scale,
             color=(230, 230, 230), bg_color=(20, 20, 20))


# ════════════════════════════════════════════════════════════════════════════
# Colourisation
# ════════════════════════════════════════════════════════════════════════════

def colorize_depth(depth_np):
    """
    Raw depth array -> coloured BGR uint8 image (Spectral_r).
    Returns (coloured_bgr, normalised_0_1_map).
    """
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_np)
    else:
        norm = (depth_np - d_min) / (d_max - d_min)

    coloured = (_CMAP_SPECTRAL(norm)[:, :, :3] * 255).astype(np.uint8)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
    return coloured, norm


def colorize_seg(seg_mask, h, w):
    """
    (H, W) class-index array -> coloured BGR uint8 image.
    seg_mask values are 0..149.
    """
    colour_map = ADE20K_PALETTE[seg_mask]                # (H, W, 3) RGB
    colour_bgr = colour_map[..., ::-1].astype(np.uint8)  # RGB -> BGR
    colour_bgr = cv2.resize(colour_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return colour_bgr


# ════════════════════════════════════════════════════════════════════════════
# Panel builders
# ════════════════════════════════════════════════════════════════════════════

def build_camera_panel(frame, cam_fps):
    """Original camera feed with FPS overlay."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H))
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'CAMERA FEED', (8, 18), scale=0.50,
             color=(200, 200, 200), bg_color=(0, 0, 0))
    put_text(panel, f'{cam_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.45, color=(100, 220, 100))
    return panel


def build_depth_panel(frame, depth_np, depth_fps):
    """
    Depth Anything V2 output with near-obstacle zone warnings.
    Red = NEAR, Blue = FAR.
    """
    coloured, norm = colorize_depth(depth_np)
    panel = cv2.resize(coloured, (PANEL_W, PANEL_H))
    norm_s = cv2.resize(norm.astype(np.float32), (PANEL_W, PANEL_H))

    # Header
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'DEPTH (red=NEAR blue=FAR)', (8, 18),
             scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{depth_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.45, color=(100, 220, 100))

    # Near-obstacle zone warnings (left / centre / right)
    NEAR_THRESH = 0.20
    near_mask = (norm_s < NEAR_THRESH).astype(np.uint8) * 255
    col_w = PANEL_W // 3
    labels = ['LEFT', 'CENTRE', 'RIGHT']
    positions = [0, col_w, 2 * col_w]
    for label, cx in zip(labels, positions):
        region = near_mask[:, cx: cx + col_w]
        frac = region.mean() / 255.0
        if frac > 0.08:
            danger_colour = (0, 0, 200) if frac > 0.25 else (0, 100, 220)
            overlay = panel.copy()
            cv2.rectangle(overlay, (cx + 2, PANEL_H - 50),
                          (cx + col_w - 2, PANEL_H - 2), danger_colour, 2)
            cv2.addWeighted(overlay, 0.4, panel, 0.6, 0, panel)
            put_text(panel, f'OBS {label}',
                     (cx + 5, PANEL_H - 10), scale=0.40,
                     color=(50, 50, 255), bg_color=(0, 0, 0))

    # Colour scale bar
    bar_h, bar_w = 12, 100
    bar_x, bar_y = PANEL_W - bar_w - 10, PANEL_H - 22
    bar = np.linspace(0, 1, bar_w)[None, :]
    bar_rgb = (_CMAP_SPECTRAL(bar)[0, :, :3] * 255).astype(np.uint8)
    bar_bgr = bar_rgb[:, ::-1]
    bar_bgr = np.repeat(bar_bgr[None], bar_h, axis=0)
    panel[bar_y: bar_y + bar_h, bar_x: bar_x + bar_w] = bar_bgr
    put_text(panel, 'NEAR', (bar_x - 38, bar_y + 9), scale=0.35,
             color=(50, 50, 255))
    put_text(panel, 'FAR', (bar_x + bar_w + 3, bar_y + 9), scale=0.35,
             color=(200, 100, 40))

    return panel


def build_seg_panel(frame, seg_mask, seg_fps):
    """
    TopFormer segmentation: ADE20K coloured mask blended with camera.
    Legend shows top-5 classes.
    """
    coloured = colorize_seg(seg_mask, PANEL_H, PANEL_W)
    cam_small = cv2.resize(frame, (PANEL_W, PANEL_H))
    panel = cv2.addWeighted(coloured, 0.65, cam_small, 0.35, 0)

    # Header
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'SEMANTIC SEG (ADE20K)', (8, 18),
             scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{seg_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.45, color=(100, 220, 100))

    # Legend: top-5 classes
    seg_small = cv2.resize(
        seg_mask.astype(np.uint8), (PANEL_W, PANEL_H),
        interpolation=cv2.INTER_NEAREST)
    unique, counts = np.unique(seg_small, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_k = min(6, len(unique))

    leg_x, leg_y = 8, 36
    cv2.rectangle(panel, (leg_x - 2, leg_y - 2),
                  (leg_x + 145, leg_y + top_k * 18 + 2),
                  (20, 20, 20), -1)

    for rank in range(top_k):
        cls_id = int(unique[order[rank]])
        if cls_id >= len(ADE20K_CLASSES):
            continue
        rgb = ADE20K_PALETTE[cls_id].tolist()
        bgr = (rgb[2], rgb[1], rgb[0])
        name = ADE20K_CLASSES[cls_id][:18]
        pct = 100.0 * counts[order[rank]] / seg_small.size
        put_label_box(panel, f'{name} {pct:.0f}%',
                      leg_x, leg_y + rank * 18, bgr)
    return panel


def build_obstacle_panel(obstacle_img, nav_instruction, obs_fps,
                         obstacle_info=None):
    """
    Obstacle detection output: fused depth+segmentation obstacle image.
    Shows identified obstacles coloured by their semantic class,
    nearest obstacle label, and navigation instruction overlay.
    """
    if obstacle_info is None:
        obstacle_info = []

    panel = cv2.resize(obstacle_img, (PANEL_W, PANEL_H),
                       interpolation=cv2.INTER_NEAREST)

    # Header
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'OBSTACLES (ODM)', (8, 18),
             scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{obs_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.45, color=(100, 220, 100))

    # Nearest obstacle label (immediately below header)
    if obstacle_info:
        nearest = obstacle_info[0]
        label = nearest['class_name'].upper()
        cv2.rectangle(panel, (0, 28), (PANEL_W, 52), (15, 15, 15), -1)
        put_text(panel, f'NEAREST: {label}', (8, 46),
                 scale=0.48, color=(0, 200, 255), bg_color=(0, 0, 0))
    else:
        cv2.rectangle(panel, (0, 28), (PANEL_W, 52), (15, 15, 15), -1)
        put_text(panel, 'NO OBSTACLES', (8, 46),
                 scale=0.48, color=(0, 255, 100), bg_color=(0, 0, 0))

    # Navigation instruction overlay at bottom
    if nav_instruction:
        cv2.rectangle(panel, (0, PANEL_H - 40), (PANEL_W, PANEL_H),
                      (0, 0, 0), -1)
        inst_lower = nav_instruction.lower()
        if 'stop' in inst_lower:
            color = (0, 0, 255)     # red
        elif 'left' in inst_lower or 'right' in inst_lower:
            color = (0, 200, 255)   # yellow
        else:
            color = (0, 255, 100)   # green
        put_text(panel, nav_instruction, (8, PANEL_H - 12),
                 scale=0.50, color=color, bg_color=(0, 0, 0))

    return panel


def build_status_bar(cam_fps, depth_fps, seg_fps, obs_fps):
    """Bottom status bar with FPS counters and key hints."""
    bar = np.zeros((STATUS_H, WIN_W, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.line(bar, (0, 0), (WIN_W, 0), (80, 80, 80), 1)

    put_text(bar, f'Cam: {cam_fps:4.1f}', (12, 22),
             scale=0.45, color=(130, 220, 130))
    put_text(bar, f'Depth: {depth_fps:4.1f}', (140, 22),
             scale=0.45, color=(130, 180, 255))
    put_text(bar, f'Seg: {seg_fps:4.1f}', (290, 22),
             scale=0.45, color=(255, 180, 100))
    put_text(bar, f'ODM: {obs_fps:4.1f}', (420, 22),
             scale=0.45, color=(255, 130, 130))
    put_text(bar, 'Q/Esc: Quit | S: Screenshot | M: Mute',
             (WIN_W - 320, 22), scale=0.42, color=(160, 160, 160))

    # Model labels row
    cv2.line(bar, (0, STATUS_H // 2), (WIN_W, STATUS_H // 2), (50, 50, 50), 1)
    put_text(bar, 'Depth Anything V2 (vitb)', (12, STATUS_H - 8),
             scale=0.38, color=(90, 140, 200))
    put_text(bar, 'TopFormer-Base ADE20K', (250, STATUS_H - 8),
             scale=0.38, color=(180, 130, 90))
    put_text(bar, 'ODM + Fuzzy Path Planner', (490, STATUS_H - 8),
             scale=0.38, color=(200, 90, 90))

    return bar
