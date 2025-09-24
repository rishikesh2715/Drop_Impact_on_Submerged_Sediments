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
parser.add_argument("--model", type=str, default="runs/segment/train3/weights/best.pt", help="Path to YOLO model.")
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

# --- Class IDs ---------------------------------------------------------------
CAVITY, CLUSTER, SEDIMENT = 0, 1, 2

# -----------------------------------------------------------------------------

# ── Load config ──────────────────────────────────────────────────────────────
def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    if not os.path.exists(path):
        print(f"[WARN] Config not found at {path}; using CLI defaults.")
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse config: {e}")
        return {}

def merge_settings(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update({k:v for k,v in override.items() if v is not None})
    return out

CONFIG = load_config(args.config)

# ── Utils ────────────────────────────────────────────────────────────────────
def pick_videos() -> list[str]:
    root = Tk(); root.withdraw()
    paths = filedialog.askopenfilenames(title="Select video(s)", filetypes=[("Video files", "*.mp4;*.mov;*.avi;*.mkv"), ("All files", "*.*")])
    root.update(); root.destroy()
    return list(paths)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def draw_text(img, text, org, color=(255,255,255), scale=0.5, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def make_panel(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def plot_trace(panel, values, y_max=None, label=""):
    h, w = panel.shape[:2]
    if len(values) < 2:
        draw_text(panel, label, (5, 15))
        return panel
    vals = np.array(values[-min(len(values), w):], dtype=float)
    if y_max is None:
        vmax = vals.max() if vals.size else 1.0
        y_max = max(1.0, vmax * 1.1)
    x_coords = np.linspace(0, w-1, len(vals)).astype(int)
    y_coords = (h - 1 - (vals / max(1e-6, y_max) * (h - 1))).astype(int)
    for i in range(1, len(vals)):
        cv2.line(panel, (x_coords[i-1], y_coords[i-1]), (x_coords[i], y_coords[i]), (255,255,255), 1, cv2.LINE_AA)
    draw_text(panel, f"{label} curr={values[-1]:.1f} max={np.max(values):.1f}", (5, 15))
    return panel


def combine_side_by_side(frame, right_panel):
    return np.hstack([frame, right_panel])


def resize_mask_to_frame(mask: np.ndarray, frame_shape) -> np.ndarray:
    H, W = frame_shape[:2]
    return cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)


def apply_roi(mask: np.ndarray, top_y: int, bottom_y: int) -> np.ndarray:
    m = np.zeros_like(mask, dtype=np.uint8)
    top = max(0, min(top_y, mask.shape[0]-1))
    bot = max(0, min(bottom_y, mask.shape[0]-1))
    if bot < top:
        top, bot = bot, top
    m[top:bot+1, :] = 1
    return (mask.astype(np.uint8) & m)


def largest_component(binary: np.ndarray) -> tuple[np.ndarray, float, tuple[int,int,int,int]]:
    """Return mask of largest connected component, its area, and its bounding rect."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    base = CONFIG.get("__default__", {})
    return merge_settings(base, best_match or {})


# ── Load model ───────────────────────────────────────────────────────────────
model = YOLO(args.model)

# ── Select videos ────────────────────────────────────────────────────────────
video_paths = pick_videos()
if not video_paths:
    raise SystemExit("No video selected.")

# Prepare to optionally accumulate effective settings to save
effective_cfg: Dict[str, Any] = json.loads(json.dumps(CONFIG)) if CONFIG else {}

for vpath in video_paths:
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {vpath}")
        continue

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # Droplet trigger setup
    triggered = settings.get("no_droplet_trigger", False)  # if disabled, start as True
    triggered = bool(triggered)  # value may be str/bool from JSON
    if triggered:
        # If JSON says no_droplet_trigger true, that means 'start as triggered'
        pass
    else:
        # If JSON/CLI says no_droplet_trigger False -> actual gating
        triggered = args.no_droplet_trigger  # respect CLI toggle when not overridden
        if triggered:  # When CLI --no_droplet_trigger set, we're triggered from start
            pass
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
        frame_idx += 1
        ts = get_frame_timestamp(frame_idx-1, src_fps)

        # ── Draw ROI lines on a copy for display/recording ───────────────────
        disp = frame.copy()
        # Lines
        cv2.line(disp, (0, roi_top_abs), (W-1, roi_top_abs), (0, 200, 255), 2)
        cv2.line(disp, (0, roi_bottom_abs), (W-1, roi_bottom_abs), (255, 180, 0), 2)
        draw_text(disp, f"ROI: {roi_top_abs}..{roi_bottom_abs}", (10, 20))

        # ── Droplet trigger logic ────────────────────────────────────────────
        if not args.no_droplet_trigger and not is_triggered:
            band_top = max(0, roi_top_abs - int(settings.get("drop_margin", args.drop_margin)))
            band = frame[band_top:roi_top_abs, :]
            fg = bg.apply(band)
            iters = int(settings.get("trigger_dilate", args.trigger_dilate))
            if iters > 0:
                fg = cv2.dilate(fg, np.ones((3,3), np.uint8), iterations=iters)
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            crossed = False
            amin = int(settings.get("drop_area_min", args.drop_area_min))
            amax = int(settings.get("drop_area_max", args.drop_area_max))
            for c in cnts:
                a = cv2.contourArea(c)
                if amin <= a <= amax:
                    ys = c[:,0,1] + band_top
                    if ys.max() >= roi_top_abs:
                        crossed = True
                        break
            if crossed:
                is_triggered = True
        draw_text(disp, f"Triggered: {is_triggered}", (10, 40), color=(0,255,0) if is_triggered else (0,0,255))

        depth_px = 0.0; width_px = 0.0; area_px2 = 0.0; cav_conf = 0.0

        if is_triggered:
            # ── Inference ───────────────────────────────────────────────────
            results = model.predict(frame, imgsz=int(settings.get("imgsz", args.imgsz)), device=settings.get("device", args.device), conf=float(settings.get("conf", args.conf)), iou=float(settings.get("iou", args.iou)), verbose=False)
            r = results[0]
            Hf, Wf = frame.shape[:2]

            best_area = 0.0
            best = {
                "mask": None,
                "area": 0.0,
                "bbox": (0,0,0,0),
                "conf": 0.0
            }
            sed_roi  = np.zeros((Hf, Wf), dtype=np.uint8)
            clus_roi = np.zeros((Hf, Wf), dtype=np.uint8)

            if r.masks is not None and r.boxes is not None and len(r.masks) > 0:
                clses = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy().astype(float)
                for m_i, cls in enumerate(clses):
                    m = r.masks.data[m_i].cpu().numpy()
                    m_res = resize_mask_to_frame(m, frame.shape)

                    if cls == CAVITY:  # cavity instance
                        cav_inst_roi = apply_roi(m_res, roi_top_abs, roi_bottom_abs)
                        mask_lrg, area, (x,y,w,h) = largest_component(cav_inst_roi)
                        if area > best_area:
                            best_area = area
                            best = {"mask": mask_lrg, "area": area, "bbox": (x,y,w,h), "conf": float(confs[m_i])}

                    elif cls == CLUSTER:  # cluster instances → union in ROI
                        clus_roi = cv2.bitwise_or(clus_roi, apply_roi(m_res, roi_top_abs, roi_bottom_abs))

                    elif cls == SEDIMENT:  # sediment instances → union in ROI
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
                sed_vis_bgr[sed_roi.astype(bool)] = (255, 0, 0)  # blue only on sediment mask
                disp = cv2.addWeighted(disp, 1.0, sed_vis_bgr, 0.18, 0)

            # Cluster overlay (orange) — visualization only
            if clus_roi.any():
                clus_vis_bgr = np.zeros_like(disp)
                clus_vis_bgr[clus_roi.astype(bool)] = (0, 165, 255)  # orange in BGR
                disp = cv2.addWeighted(disp, 1.0, clus_vis_bgr, 0.25, 0)
                # tiny legend (non-intrusive)
                draw_text(disp, "Cluster overlay", (10, 80), color=(0,165,255))


        # Update histories (UI only)
        depth_hist.append(depth_px)
        width_hist.append(width_px)
        area_hist.append(area_px2)

        # Right panel (three stacked charts) + unit labels
        panel = make_panel(H, PLOT_WIDTH)
        h_each = H // 3
        p1 = panel[0:h_each, :];   plot_trace(p1, list(depth_hist), y_max=H, label=f"depth ({args.unit_depth})")
        p2 = panel[h_each:2*h_each, :]; plot_trace(p2, list(width_hist), y_max=W, label=f"width ({args.unit_width})")
        p3 = panel[2*h_each:H, :]; plot_trace(p3, list(area_hist), y_max=None, label=f"area ({args.unit_area})")

        composite = combine_side_by_side(disp, panel)

        # CSV row
        writer.writerow([frame_idx-1, f"{ts:.6f}", int(is_triggered), f"{depth_px:.3f}", f"{width_px:.3f}", f"{area_px2:.3f}", roi_top_abs, roi_bottom_abs, f"{cav_conf:.3f}"])

        # I/O & controls
        if SHOW_VIDEO:
            cv2.imshow("Cavity ROI Analyzer", composite)
            # If starting paused, wait for 'p' to begin; still allow ROI adjustments
            if paused:
                while True:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('p'):
                        paused = False
                        break
                    if k == ord('q'):
                        paused = False
                        cap.release()
                        if pbar is not None: pbar.close()
                        if out_writer is not None: out_writer.release()
                        csv_f.close()
                        cv2.destroyAllWindows()
                        raise SystemExit(0)
                    # ROI adjust while paused
                    if k == ord('i'): interface_y -= 2
                    if k == ord('k'): interface_y += 2
                    if k == ord('o'): sediment_y  -= 2
                    if k == ord('l'): sediment_y  += 2
                    interface_y = max(0, min(H-1, interface_y))
                    sediment_y  = max(0, min(H-1, sediment_y))
                    roi_top_abs = max(0, min(H-1, interface_y - int(settings.get("roi_safety", 0))))
                    roi_bottom_abs = max(0, min(H-1, sediment_y + int(settings.get("roi_safety", 0))))
                    if roi_bottom_abs < roi_top_abs:
                        roi_top_abs, roi_bottom_abs = roi_bottom_abs, roi_top_abs
                    disp2 = frame.copy()
                    cv2.line(disp2, (0, roi_top_abs), (W-1, roi_top_abs), (0, 200, 255), 2)
                    cv2.line(disp2, (0, roi_bottom_abs), (W-1, roi_bottom_abs), (255, 180, 0), 2)
                    draw_text(disp2, f"ROI: {roi_top_abs}..{roi_bottom_abs}", (10, 20))
                    cv2.imshow("Cavity ROI Analyzer", combine_side_by_side(disp2, panel))
            else:
                key = cv2.waitKey(DELAY_MS) & 0xFF
                if key == ord('q'):
                    break
                # pause toggle
                if key == ord('p'):
                    paused = True
                # live ROI nudge
                if key == ord('i'): interface_y -= 2
                if key == ord('k'): interface_y += 2
                if key == ord('o'): sediment_y  -= 2
                if key == ord('l'): sediment_y  += 2
                interface_y = max(0, min(H-1, interface_y))
                sediment_y  = max(0, min(H-1, sediment_y))
                roi_top_abs = max(0, min(H-1, interface_y - int(settings.get("roi_safety", 0))))
                roi_bottom_abs = max(0, min(H-1, sediment_y + int(settings.get("roi_safety", 0))))
                if roi_bottom_abs < roi_top_abs:
                    roi_top_abs, roi_bottom_abs = roi_bottom_abs, roi_top_abs

        if SAVE_VIDEO and out_writer is not None:
            out_writer.write(composite)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Save effective settings for this folder if requested
    if args.save_config:
        vdir = os.path.abspath(os.path.dirname(vpath))
        if vdir not in effective_cfg:
            effective_cfg[vdir] = {}
        # Record ROI & selected timing/threshold knobs (extend as needed)
        effective_cfg[vdir].update({
            "interface_y": int(interface_y),
            "sediment_y": int(sediment_y),
            "conf": float(settings.get("conf", args.conf)),
            "delay_ms": int(settings.get("delay_ms", args.delay_ms)),
            "save_fps": settings.get("save_fps", None),
            "start_paused": bool(settings.get("start_paused", args.start_paused)),
            "roi_safety": int(settings.get("roi_safety", 0))
        })

    cap.release()
    if out_writer is not None:
        out_writer.release()
    csv_f.close()

# After processing all videos, write effective config if requested
if args.save_config:
    # Merge with any existing on-disk JSON to preserve past presets
    merged = {}
    if os.path.exists(args.save_config):
        try:
            with open(args.save_config, "r") as f:
                merged = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read existing save_config file: {e}. Creating anew.")
    # Deep-merge: keep existing keys, update/insert folders we processed
    for k, v in (merged or {}).items():
        if k not in effective_cfg:
            effective_cfg[k] = v
    try:
        with open(args.save_config, "w") as f:
            json.dump(effective_cfg, f, indent=2)
        print(f"[INFO] Saved config to {args.save_config}")
    except Exception as e:
        print(f"[WARN] Failed to save config: {e}")

cv2.destroyAllWindows()