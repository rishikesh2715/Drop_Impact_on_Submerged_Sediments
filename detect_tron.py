#!/usr/bin/env python3

from __future__ import annotations
import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import filedialog, Tk
import os
import csv
import argparse
from tqdm import tqdm
from collections import deque
import json
from typing import Dict, Any, Optional

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run YOLO segmentation on videos with ROI, trigger, and presets.")
parser.add_argument("--show_video", action="store_true", help="Render live windows (slower).")
parser.add_argument("--save_video", action="store_true", help="Write annotated video to disk.")
parser.add_argument("--model", type=str, default="runs/segment/train2/weights/best.pt", help="Path to YOLO model.")
parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if None).")
parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")

# ROI lines (absolute Y in pixels in the original frame coordinates)
parser.add_argument("--interface_y", type=int, default=None, help="Air–water interface y (px).")
parser.add_argument("--sediment_y", type=int, default=None, help="Water–sediment interface y (px).")
parser.add_argument("--roi_safety", type=int, default=0, help="Expand ROI vertically by this many px on both sides.")

# Droplet trigger controls
parser.add_argument("--no_droplet_trigger", action="store_true", help="Disable droplet-trigger gating.")
parser.add_argument("--drop_area_min", type=int, default=20, help="Min contour area (px) to treat as droplet candidate.")
parser.add_argument("--drop_area_max", type=int, default=3000, help="Max contour area (px) to treat as droplet candidate.")
parser.add_argument("--drop_margin", type=int, default=20, help="Band height (px) above interface used to watch for droplet.")
parser.add_argument("--trigger_dilate", type=int, default=2, help="Morphological dilation iters on FG mask before contours.")

# Plotting / display
parser.add_argument("--plot_width", type=int, default=500, help="Right panel width in px.")
parser.add_argument("--max_trace_points", type=int, default=1200, help="Visible history length for on-screen plots.")
parser.add_argument("--unit_depth", type=str, default="px", help="Unit label for depth axis.")
parser.add_argument("--unit_width", type=str, default="px", help="Unit label for width axis.")
parser.add_argument("--unit_area", type=str, default="px^2", help="Unit label for area axis.")

# Timebase control
parser.add_argument("--delay_ms", type=int, default=1, help="imshow waitKey delay (ms) when showing video.")
parser.add_argument("--save_fps", type=float, default=None, help="Output video FPS. Default: source FPS.")
parser.add_argument("--start_paused", action="store_true", help="Start playback in paused state; press 'p' to begin.")

# --- Droplet detector selection & Hough params ---
parser.add_argument("--droplet_detector", choices=["auto", "hough", "bg"], default="auto",
                    help="Droplet trigger method: 'hough' (circles), 'bg' (band-cross), or 'auto' (try hough then bg).")
parser.add_argument("--hough_dp", type=float, default=1.2, help="HoughCircles dp.")
parser.add_argument("--hough_min_dist", type=int, default=30, help="HoughCircles minDist.")
parser.add_argument("--hough_param1", type=int, default=50, help="HoughCircles param1.")
parser.add_argument("--hough_param2", type=int, default=30, help="HoughCircles param2 (higher=fewer).")
parser.add_argument("--hough_min_radius", type=int, default=5, help="Minimum droplet radius (px).")
parser.add_argument("--hough_max_radius", type=int, default=30, help="Maximum droplet radius (px).")

# --- Low-light enhancement ---
parser.add_argument("--enhance", choices=["none", "clahe", "hist", "gamma", "auto"], default="auto",
                    help="Brightness/contrast enhancement. 'auto'=CLAHE if frame is dark.")
parser.add_argument("--gamma", type=float, default=1.2, help="Gamma for --enhance gamma.")
parser.add_argument("--auto_brightness_thresh", type=float, default=90.0, help="Mean L* threshold for auto CLAHE (0-255).")

# Config presets
parser.add_argument("--config", type=str, default=None, help="Path to JSON config with per-folder presets.")
parser.add_argument("--save_config", type=str, default=None, help="If set, write back effective settings to this JSON.")

args = parser.parse_args()
SHOW_VIDEO: bool = args.show_video
SAVE_VIDEO: bool = args.save_video

# ── Config ───────────────────────────────────────────────────────────────────
PLOT_WIDTH = int(args.plot_width)
FONT = cv2.FONT_HERSHEY_SIMPLEX
HISTORY_KEEP = int(args.max_trace_points)
DELAY_MS = int(args.delay_ms)

# ── Config JSON helpers ──────────────────────────────────────────────────────
def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load config {path}: {e}")
        return {}


def merge_settings(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = v
    return out


def pick_videos() -> list[str]:
    Tk().withdraw()
    paths = filedialog.askopenfilenames(title="Pick video(s)", filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv"), ("All", "*.*")])
    return list(paths)


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def draw_text(img, text, org, color=(255,255,255), scale=0.6, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def make_panel(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def plot_trace(panel: np.ndarray, values: deque[float], title: str, unit: str, y: int, height: int, margin: int = 8):
    # Simple polyline plotter
    cv2.rectangle(panel, (0, y), (panel.shape[1]-1, y+height), (30, 30, 30), -1)
    if len(values) >= 2:
        v = np.array(values, dtype=np.float32)
        v = v - np.nanmin(v)
        vmax = max(np.nanmax(v), 1e-6)
        v = (v / vmax) * (height-2)
        xs = np.linspace(0, panel.shape[1]-2*margin, num=len(v)).astype(int)
        ys = height - v.astype(int) + y - 1
        pts = np.stack([xs+margin, ys], axis=1)
        cv2.polylines(panel, [pts], isClosed=False, color=(0,200,255), thickness=2)
    draw_text(panel, f"{title} ({unit})", (margin, y+18), color=(200,200,200))


def combine_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (right.shape[1], left.shape[0]))
    return np.hstack([left, right])


def resize_mask_to_frame(mask: np.ndarray, frame_shape) -> np.ndarray:
    H, W = frame_shape[:2]
    return cv2.resize((mask>0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)


def apply_roi(mask: np.ndarray, top: int, bottom: int) -> np.ndarray:
    out = np.zeros_like(mask)
    top = max(0, min(mask.shape[0]-1, top))
    bottom = max(0, min(mask.shape[0]-1, bottom))
    if bottom < top:
        top, bottom = bottom, top
    out[top:bottom+1] = mask[top:bottom+1]
    return out


def largest_component(binary: np.ndarray):
    cnts, _ = cv2.findContours((binary>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(binary), 0.0, (0,0,0,0)
    areas = [cv2.contourArea(c) for c in cnts]
    idx = int(np.argmax(areas))
    c = cnts[idx]
    area = float(areas[idx])
    x,y,w,h = cv2.boundingRect(c)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)
    return mask, area, (x,y,w,h)


def get_frame_timestamp(frame_idx: int, fps: float) -> float:
    return frame_idx / max(1e-6, fps)


def enhance_frame_bgr(frame_bgr: np.ndarray, mode: str, gamma: float = 1.2, auto_thresh: float = 90.0) -> np.ndarray:
    # Enhance brightness/contrast in BGR space.
    if mode == "none":
        return frame_bgr
    if mode == "auto":
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        if float(np.mean(L)) < auto_thresh:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            L2 = clahe.apply(L)
            lab2 = cv2.merge([L2, A, B])
            return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return frame_bgr
    if mode == "clahe":
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if mode == "hist":
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        Y2 = cv2.equalizeHist(Y)
        ycrcb2 = cv2.merge([Y2, Cr, Cb])
        return cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)
    if mode == "gamma":
        inv = 1.0 / max(gamma, 1e-6)
        table = (np.array([((i/255.0)**inv)*255 for i in range(256)])).astype("uint8")
        return cv2.LUT(frame_bgr, table)
    return frame_bgr


def detect_droplet_hough(gray: np.ndarray,
                         interface_y: Optional[int],
                         dp: float,
                         min_dist: int,
                         p1: int,
                         p2: int,
                         rmin: int,
                         rmax: int):
    # Returns best circle (x, y, r) chosen as the lowest center above interface_y (if provided); else overall lowest.
    if gray is None or gray.size == 0:
        return None
    H = gray.shape[0]
    search_end = interface_y if interface_y is not None else H
    if search_end <= 0:
        return None
    roi = gray[0:search_end, :]
    blurred = cv2.GaussianBlur(roi, (9,9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=p1, param2=p2,
                               minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    best, max_y = None, -1
    for (xc, y_rel, rc) in circles:
        y_abs = int(y_rel)
        if y_abs > max_y:
            best, max_y = (int(xc), y_abs, int(rc)), y_abs
    return best


def dir_settings_for_video(vpath: str) -> Dict[str, Any]:
    """Return settings from CONFIG for this video's directory; supports exact or prefix matches.
    JSON shape example:
    {
      "__default__": {"interface_y": 220, "sediment_y": 700, "conf": 0.25},
      "/data/runA": {"interface_y": 200, "sediment_y": 500},
      "/data/runB": {"interface_y": 220, "sediment_y": 450}
    }
    """
    vdir = os.path.abspath(os.path.dirname(vpath))
    best_match = None
    best_len = -1
    for key, cfg in CONFIG.items():
        if key == "__default__":
            continue
        key_abs = os.path.abspath(key)
        if vdir.startswith(key_abs) and len(key_abs) > best_len:
            best_match = cfg
            best_len = len(key_abs)
    # If none matched, fallback to __default__
    if best_match is None:
        best_match = CONFIG.get("__default__", {})
    return dict(best_match)

# ── MAIN ─────────────────────────────────────────────────────────────────────
CONFIG = load_config(args.config)

videos = pick_videos()
if not videos:
    print("[INFO] No videos selected; exiting.")
    raise SystemExit(0)

model = YOLO(args.model)

for vpath in videos:
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {vpath}")
        continue

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Folder-specific overrides
    folder_over = dir_settings_for_video(vpath)
    # Merge CLI (base) → folder overrides (take if provided)
    settings = merge_settings(vars(args), folder_over)

    # ROI lines: initialize
    interface_y = settings.get("interface_y") if settings.get("interface_y") is not None else int(0.35 * H)
    sediment_y  = settings.get("sediment_y")  if settings.get("sediment_y")  is not None else int(0.85 * H)
    interface_y = max(0, min(H-1, int(interface_y)))
    sediment_y  = max(0, min(H-1, int(sediment_y)))

    roi_top_abs = max(0, min(H-1, interface_y - int(settings.get("roi_safety", 0))))
    roi_bottom_abs = max(0, min(H-1, sediment_y + int(settings.get("roi_safety", 0))))
    if roi_bottom_abs < roi_top_abs:
        roi_top_abs, roi_bottom_abs = roi_bottom_abs, roi_top_abs

    # Output paths
    stem, ext = os.path.splitext(vpath)
    csv_path = f"{stem}_metrics.csv"
    out_path = f"{stem}_metrics.mp4"

    # Video writer
    out_writer = None
    out_fps = settings.get("save_fps") if settings.get("save_fps") is not None else src_fps
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, float(out_fps), (W + PLOT_WIDTH, H))
        if not out_writer.isOpened():
            print("[WARN] Could not open VideoWriter; disabling save_video.")
            out_writer = None

    # CSV
    ensure_dir(csv_path)
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["frame","time_s","triggered","depth_px","width_px","area_px2","roi_top_y","roi_bottom_y","cavity_conf"]) 

    # Histories (bounded for UI only)
    depth_hist: deque[float] = deque(maxlen=HISTORY_KEEP)
    width_hist: deque[float] = deque(maxlen=HISTORY_KEEP)
    area_hist:  deque[float] = deque(maxlen=HISTORY_KEEP)

    # Enhancement & detector params
    detector_mode = str(settings.get("droplet_detector", args.droplet_detector))
    hough_dp      = float(settings.get("hough_dp", args.hough_dp))
    hough_md      = int(settings.get("hough_min_dist", args.hough_min_dist))
    hough_p1      = int(settings.get("hough_param1", args.hough_param1))
    hough_p2      = int(settings.get("hough_param2", args.hough_param2))
    hough_rmin    = int(settings.get("hough_min_radius", args.hough_min_radius))
    hough_rmax    = int(settings.get("hough_max_radius", args.hough_max_radius))

    enh_mode   = str(settings.get("enhance", args.enhance))
    enh_gamma  = float(settings.get("gamma", args.gamma))
    auto_L_thr = float(settings.get("auto_brightness_thresh", args.auto_brightness_thresh))

    # Hough tracking state
    tracking_active = True
    prev_position   = None
    prev_time       = None
    trajectory      = []
    impact_frame    = None
    impact_x_coord  = None

    # Droplet trigger setup (legacy flag honored)
    is_triggered = True if args.no_droplet_trigger else False
    if not args.no_droplet_trigger:
        is_triggered = False

    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # Playback control
    paused = bool(settings.get("start_paused", args.start_paused)) and SHOW_VIDEO

    frame_idx = 0
    pbar = None if SHOW_VIDEO else tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc=os.path.basename(vpath), unit="f")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # === Enhancement (applied before trigger & inference) ===
        proc_frame = enhance_frame_bgr(frame, mode=enh_mode, gamma=enh_gamma, auto_thresh=auto_L_thr)
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1
        ts = get_frame_timestamp(frame_idx-1, src_fps)

        # ── Draw ROI lines on a copy for display/recording ───────────────────
        disp = frame.copy()
        # Lines
        cv2.line(disp, (0, roi_top_abs), (W-1, roi_top_abs), (0, 200, 255), 1)
        cv2.line(disp, (0, roi_bottom_abs), (W-1, roi_bottom_abs), (255, 180, 0), 1)
        draw_text(disp, f"ROI: {roi_top_abs}..{roi_bottom_abs}", (10, 20))

        # ── Droplet trigger logic ────────────────────────────────────────────
        trigger_disabled = bool(settings.get("no_droplet_trigger", args.no_droplet_trigger))
        if not trigger_disabled and not is_triggered:
            # timestamp for velocity
            timestamp = ts
            # True interface line for contact (without safety padding)
            interface_y_true = int(interface_y) if interface_y is not None else None

            # Try Hough-circles first (if selected/auto)
            if detector_mode in ("hough", "auto"):
                best = detect_droplet_hough(gray, interface_y_true, hough_dp, hough_md, hough_p1, hough_p2, hough_rmin, hough_rmax)
                if best is not None:
                    x, y, r = best
                    cv2.circle(disp, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(disp, (x, y), 2, (0, 0, 255), 3)
                    velocity = None
                    if prev_position is not None and prev_time is not None and timestamp > prev_time:
                        dy = y - prev_position[1]; dt = timestamp - prev_time
                        if dt > 1e-6: velocity = dy / dt
                    trajectory.append({"timestamp": timestamp, "x": x, "y": y, "radius": r, "velocity": velocity})
                    prev_position, prev_time = (x, y), timestamp
                    if interface_y_true is not None and (y + r) >= interface_y_true:
                        impact_frame = frame_idx
                        impact_x_coord = x
                        is_triggered = True
                        cv2.putText(disp, "CONTACT", (x + 10, y + r + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)

            # Fallback / alternative: original BG-sub band crossing
            if not is_triggered and (detector_mode in ("bg", "auto")):
                band_top = max(0, roi_top_abs - int(settings.get("drop_margin", args.drop_margin)))
                band = proc_frame[band_top:roi_top_abs, :]
                fg = bg.apply(band)
                iters = int(settings.get("trigger_dilate", args.trigger_dilate))
                if iters > 0:
                    fg = cv2.dilate(fg, np.ones((3,3), np.uint8), iterations=iters)
                cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                amin = int(settings.get("drop_area_min", args.drop_area_min))
                amax = int(settings.get("drop_area_max", args.drop_area_max))
                for c in cnts:
                    a = cv2.contourArea(c)
                    if amin <= a <= amax:
                        ys = c[:,0,1] + band_top
                        if ys.max() >= roi_top_abs:
                            is_triggered = True
                            break
        draw_text(disp, f"Triggered: {is_triggered}", (10, 40), color=(0,255,0) if is_triggered else (0,0,255))

        depth_px = 0.0; width_px = 0.0; area_px2 = 0.0; cav_conf = 0.0

        if is_triggered:
            # ── Inference ───────────────────────────────────────────────────
            results = model.predict(proc_frame, imgsz=int(settings.get("imgsz", args.imgsz)), conf=float(settings.get("conf", args.conf)), iou=float(settings.get("iou", args.iou)), verbose=False)
            r = results[0]
            Hf, Wf = frame.shape[:2]

            # Collect per-instance masks & confidences, then choose the largest cavity INSTANCE inside ROI
            best_area = 0.0
            best = {
                "mask": None,
                "area": 0.0,
                "bbox": (0,0,0,0),
                "conf": 0.0
            }
            sed_roi = np.zeros((Hf, Wf), dtype=np.uint8)

            if r.masks is not None and r.boxes is not None and len(r.masks) > 0:
                clses = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy().astype(float)
                for m_i, cls in enumerate(clses):
                    m = r.masks.data[m_i].cpu().numpy()
                    m_res = resize_mask_to_frame(m, frame.shape)
                    if cls == 0:  # cavity instance
                        cav_inst_roi = apply_roi(m_res, roi_top_abs, roi_bottom_abs)
                        mask_lrg, area, (x,y,w,h) = largest_component(cav_inst_roi)
                        if area > best_area:
                            best_area = area
                            best = {"mask": mask_lrg, "area": area, "bbox": (x,y,w,h), "conf": float(confs[m_i])}
                    elif cls == 1:  # sediment
                        sed_roi = cv2.bitwise_or(sed_roi, apply_roi(m_res, roi_top_abs, roi_bottom_abs))

            # Metrics from best cavity instance
            area_px2 = float(best["area"]) if best["area"] else 0.0
            x,y,w,h = best["bbox"]
            width_px = float(w)
            depth_px = float(h)
            cav_conf = float(best["conf"]) if best["area"] else 0.0

            # Overlays (pink for cavity, blue for sediment)
            if area_px2 > 0:
                cav_vis_mask = (best["mask"] > 0)
                cav_vis_bgr = np.zeros_like(disp)
                cav_vis_bgr[cav_vis_mask] = (255, 0, 255)  # pink in BGR only where mask==1
                disp = cv2.addWeighted(disp, 1.0, cav_vis_bgr, 0.35, 0)  # semi-transparent
                cv2.rectangle(disp, (x,y), (x+w, y+h), (255, 0, 255), 2)
                draw_text(disp, f"Cavity: w={w}px, h={h}px, A={int(area_px2)}px^2, conf={cav_conf:.2f}", (10, 60))
            if sed_roi.any():
                sed_vis_bgr = np.zeros_like(disp)
                sed_vis_bgr[sed_roi>0] = (255, 170, 0)  # blue-ish? keep consistent; earlier was blue
                disp = cv2.addWeighted(disp, 1.0, sed_vis_bgr, 0.25, 0)

            # Append histories for live plots
            depth_hist.append(depth_px)
            width_hist.append(width_px)
            area_hist.append(area_px2)

        # ── Right panel with three traces ────────────────────────────────────
        panel = make_panel(H, PLOT_WIDTH)
        plot_trace(panel, depth_hist, "Depth", settings.get("unit_depth", args.unit_depth), 0, H//3)
        plot_trace(panel, width_hist, "Width", settings.get("unit_width", args.unit_width), H//3, H//3)
        plot_trace(panel, area_hist,  "Area",  settings.get("unit_area", args.unit_area), 2*H//3, H//3)

        # ── IO and UI ───────────────────────────────────────────────────────
        combo = combine_side_by_side(disp, panel)
        if out_writer is not None:
            out_writer.write(combo)

        if SHOW_VIDEO:
            if paused:
                cv2.imshow("Cavity ROI Analyzer", combine_side_by_side(disp, panel))
                while True:
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('p'):
                        paused = False
                        break
                    elif key == ord('q'):
                        cap.release()
                        if out_writer: out_writer.release()
                        csv_f.close()
                        cv2.destroyAllWindows()
                        raise SystemExit(0)
                    # Allow adjusting ROI while paused
                    elif key == ord('i'):
                        interface_y = max(0, interface_y-1)
                    elif key == ord('k'):
                        interface_y = min(H-1, interface_y+1)
                    elif key == ord('o'):
                        sediment_y = max(0, sediment_y-1)
                    elif key == ord('l'):
                        sediment_y = min(H-1, sediment_y+1)
                    # Recompute ROI with safety when changed
                    roi_top_abs = max(0, min(H-1, interface_y - int(settings.get("roi_safety", 0))))
                    roi_bottom_abs = max(0, min(H-1, sediment_y + int(settings.get("roi_safety", 0))))
                    if roi_bottom_abs < roi_top_abs:
                        roi_top_abs, roi_bottom_abs = roi_bottom_abs, roi_top_abs
                    disp2 = frame.copy()
                    cv2.line(disp2, (0, roi_top_abs), (W-1, roi_top_abs), (0, 200, 255), 1)
                    cv2.line(disp2, (0, roi_bottom_abs), (W-1, roi_bottom_abs), (255, 180, 0), 1)
                    draw_text(disp2, f"ROI: {roi_top_abs}..{roi_bottom_abs}", (10, 20))
                    cv2.imshow("Cavity ROI Analyzer", combine_side_by_side(disp2, panel))
            else:
                key = cv2.waitKey(DELAY_MS) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = True
                elif key == ord('i'):
                    interface_y = max(0, interface_y-1)
                elif key == ord('k'):
                    interface_y = min(H-1, interface_y+1)
                elif key == ord('o'):
                    sediment_y = max(0, sediment_y-1)
                elif key == ord('l'):
                    sediment_y = min(H-1, sediment_y+1)
                # update ROI
                roi_top_abs = max(0, min(H-1, interface_y - int(settings.get("roi_safety", 0))))
                roi_bottom_abs = max(0, min(H-1, sediment_y + int(settings.get("roi_safety", 0))))
                if roi_bottom_abs < roi_top_abs:
                    roi_top_abs, roi_bottom_abs = roi_bottom_abs, roi_top_abs

        if not SHOW_VIDEO and pbar is not None:
            pbar.update(1)

        # CSV row per frame
        writer.writerow([frame_idx-1, f"{ts:.6f}", int(is_triggered), f"{depth_px:.2f}", f"{width_px:.2f}", f"{area_px2:.2f}", roi_top_abs, roi_bottom_abs, f"{cav_conf:.3f}"])

    # Cleanup this video
    cap.release()
    if out_writer: out_writer.release()
    csv_f.close()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()

# Save updated config if requested
if args.save_config:
    existing = load_config(args.save_config)
    # Preserve other folders; update the current folder with last-used settings
    # (Here we simply write back __default__ with current args; you can expand to per-folder if desired.)
    existing["__default__"] = {
        "interface_y": args.interface_y,
        "sediment_y": args.sediment_y,
        "roi_safety": args.roi_safety,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save_fps": args.save_fps,
        "unit_depth": args.unit_depth,
        "unit_width": args.unit_width,
        "unit_area": args.unit_area,
        "droplet_detector": args.droplet_detector,
        "hough_dp": args.hough_dp,
        "hough_min_dist": args.hough_min_dist,
        "hough_param1": args.hough_param1,
        "hough_param2": args.hough_param2,
        "hough_min_radius": args.hough_min_radius,
        "hough_max_radius": args.hough_max_radius,
        "enhance": args.enhance,
        "gamma": args.gamma,
        "auto_brightness_thresh": args.auto_brightness_thresh
    }
    try:
        with open(args.save_config, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"[OK] Wrote settings to {args.save_config}")
    except Exception as e:
        print(f"[WARN] Could not write {args.save_config}: {e}")
