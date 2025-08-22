#!/usr/bin/env python3
"""
YOLO cavity-sediment analysis with optional live display for multiple videos.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import filedialog, Tk
import os
import csv
import argparse
from tqdm import tqdm

# ── Command-line switches ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run YOLO segmentation on videos and log cavity metrics."
)
parser.add_argument(
    "--show_video",
    action="store_true",
    help="Render live windows (slower). Omit for head-less processing.",
)
parser.add_argument(
    "--save_video",
    action="store_true",
    help="Write annotated video to disk (default: off).",
)
args = parser.parse_args()
SHOW_VIDEO: bool = args.show_video
SAVE_VIDEO: bool = args.save_video

# ── Plotting config ─────────────────────────────────────────────
PLOT_WIDTH = 500
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
COLORS = {
    "depth": (255, 100, 100),
    "width": (100, 255, 100),
    "area": (100, 100, 255),
}

def draw_plot(canvas, data, label, color, current_value, pos_y, max_val_display=None):
    h, w = canvas.shape[:2]
    plot_h = h // 3
    start_y = pos_y * plot_h
    cv2.putText(canvas, f"{label}: {current_value:.2f}", (10, start_y + 20), FONT, FONT_SCALE, color, FONT_THICKNESS, lineType=cv2.LINE_AA)
    if len(data) < 2:
        return
    max_val = max_val_display or max(max(data), 1)
    pts = [(int(10 + (i / (len(data) - 1)) * (w - 20)), int(start_y + plot_h - 20 - ((val / max_val) * (plot_h - 40)))) for i, val in enumerate(data)]
    cv2.polylines(canvas, [np.array(pts)], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

# ── Select multiple files ─────────────────────────────────────────────
root = Tk()
root.withdraw()
video_paths = filedialog.askopenfilenames(title="Select one or more videos", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
if not video_paths:
    print("No videos selected. Exiting.")
    exit()

# ── Load model ─────────────────────────────────────────────
model = YOLO("runs/segment/train2/weights/best.pt")

# ── Process each video ─────────────────────────────────────────────
for video_path in video_paths:
    print(f"\n🎬 Processing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # delay_ms = int(1000 / fps)
    delay_ms = int(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    combined_width = frame_width + PLOT_WIDTH
    progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame") if not SHOW_VIDEO else None

    if SAVE_VIDEO:
        output_path = os.path.splitext(video_path)[0] + "_metrics.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, frame_height))
    else:
        out = None

    csv_path = os.path.splitext(video_path)[0] + "_metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_s", "depth_px", "width_px", "area_px2"])

    print(f"Loaded: {video_path} | FPS: {fps:.2f}")
    if SAVE_VIDEO:
        print(f"Saving annotated video to: {output_path}")
    print(f"Saving metrics to: {csv_path}")

    # ── Runtime state ─
    depth_history, width_history, area_history = [], [], []
    frame_idx = 0
    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if progress_bar:
                progress_bar.update(1)

        results = model.predict(frame, conf=0.3, iou=0.5, verbose=False)

        segmented_sed_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if results[0].masks is not None:
            cls_ids = results[0].boxes.cls.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy().astype("uint8")
            for idx, cid in enumerate(cls_ids):
                if cid == 1:
                    resized = cv2.resize(masks[idx], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    segmented_sed_mask = cv2.bitwise_or(segmented_sed_mask, resized * 255)

        if SHOW_VIDEO:
            cv2.imshow("Segmented Sediment (Binary)", segmented_sed_mask)

        cavity_depth = cavity_width = cavity_area = 0
        if results[0].masks is not None:
            cls_ids = results[0].boxes.cls.cpu().numpy()
            cav_idx = np.where(cls_ids == 0)[0]
            if cav_idx.size:
                mask_data = results[0].masks.data[cav_idx[0]].cpu().numpy().astype("uint8")
                cavity_area = int(mask_data.sum())
                contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cavity_width, cavity_depth = w, h

        depth_history.append(cavity_depth)
        width_history.append(cavity_width)
        area_history.append(cavity_area)

        timestamp_s = frame_idx / fps
        csv_writer.writerow([frame_idx, f"{timestamp_s:.4f}", cavity_depth, cavity_width, cavity_area])

        annotated = results[0].plot(boxes=False, conf=False, labels=False)
        plot_canvas = np.zeros((frame_height, PLOT_WIDTH, 3), dtype=np.uint8)
        draw_plot(plot_canvas, depth_history, "Depth (px)", COLORS["depth"], cavity_depth, 0, frame_height)
        draw_plot(plot_canvas, width_history, "Width (px)", COLORS["width"], cavity_width, 1, frame_width)
        draw_plot(plot_canvas, area_history, "Area (px²)", COLORS["area"], cavity_area, 2)
        combined = cv2.hconcat([annotated, plot_canvas])

        if SHOW_VIDEO:
            cv2.imshow("YOLO Segmentation with Real-time Metrics", combined)
            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print("[INFO] Paused" if paused else "[INFO] Resumed")
        else:
            cv2.waitKey(1)

        if SAVE_VIDEO and not paused:
            out.write(combined)

    # ── Cleanup per video ─
    cap.release()
    if SAVE_VIDEO:
        out.release()
    csv_file.close()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
    if progress_bar:
        progress_bar.close()

    print("✅ Metrics saved to:", csv_path)
    if SAVE_VIDEO:
        print("✅ Video saved to  :", output_path)

print("\n🎉 All videos processed.")
