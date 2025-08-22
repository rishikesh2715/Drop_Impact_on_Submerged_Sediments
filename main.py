import cv2
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk

# --- Config ---
root = Tk()
root.withdraw()
VIDEO_PATH = filedialog.askopenfilename(filetypes=[("Video files", "*.avi")])
print(VIDEO_PATH)
# VIDEO_PATH = 'D:/Documents/Drop_Impact_on_Submerged_Sediments/video_data/L1-34.2cm-H1-4.2cm-d50-600-um-Blue-nozzle.avi'
SHOW_LIVE = True

MANUAL_SEDIMENT_ANALYSIS_START_FRAME = 198

# --- MODIFIED THRESHOLD PARAMETERS ---
# This threshold separates brighter water from darker sediment/cavity.
# Pixels <= this value will initially be considered for sediment/cavity.
WATER_DARKS_SEPARATOR_THRESH = 190 # TUNE THIS (e.g., 90-110, must be < water intensity)

# This threshold separates the very dark cavity from the (less dark) sediment.
# Pixels <= this value are considered cavity.
CAVITY_DARKNESS_THRESH = 45     # TUNE THIS (e.g., 30-50, must be < sediment intensity but > cavity)
# --- END MODIFICATION ---


# Load the video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {frame_width}x{frame_height}")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
delay = int(1000 / fps) if fps > 0 else 1
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

interface_y = None
# air_water_interface_y = int(frame_height * 0.15)
# sediment_interface_y = int(frame_height * 0.7)

# if video path contains "Regime_II" then change the interface y to 0.15
if "Regime_II" in VIDEO_PATH:
    air_water_interface_y = int(frame_height * 0.5)
    sediment_interface_y = int(frame_height * 0.63)
else:
    air_water_interface_y = int(frame_height * 0.18)
    sediment_interface_y = int(frame_height * 0.65)

tracking_active = True
cavity_active = False
cavity_contact_frame = None
prev_position = None
prev_time = None
impact_frame = None
impact_x_coord = None
prev_cavity_stats = None
trajectory = []
sediment_analysis_active = False
sediment_data_list = []
SAVE_VIDEO = False

# pause the video but keep the main process running
paused = False
stored_frame = None

# --- KALMAN FILTER for smoothing the cavity metrics---
class CavityKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)  # 8 states (x,y,d,w and their velocities), 4 measurements
        dt = 1.0  # time step between frames (you can tune later)

        # State: [x, y, d, w, vx, vy, vd, vw]
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)  # observe x, y, d, w only
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    def update(self, measurement):  # measurement = (x, y, depth, width)
        if not self.initialized:
            self.kf.statePost[:4, 0] = np.array(measurement, dtype=np.float32)
            self.kf.statePost[4:, 0] = 0
            self.initialized = True
        else:
            self.kf.correct(np.array(measurement, dtype=np.float32).reshape(4,1))

    def predict(self):
        pred = self.kf.predict()
        x, y, d, w = pred[:4].flatten()
        return int(x), int(y), float(d), float(w)

# --- END KALMAN FILTER ---

# Initialize the Kalman filter
kalman_filter = CavityKalman()


# --- DETECT CAVITY CONTOUR ---
def detect_cavity_contour(gray_frame, interface_y_val, sediment_y_val,
                          expected_center_x=None, prev_metrics=None, frame_bgr=None,
                          canny_thresh1=20, canny_thresh2=100,
                          blur_ksize=9, min_ctr_w=12, min_ctr_h=8, y_cutoff_ratio=0.4):

    # ... (YOUR detect_cavity_contour function - UNCHANGED) ...
    if "Regime_II" in VIDEO_PATH:
        roi_top_offset = 28
    else:
        roi_top_offset = 50

    roi_top_abs = interface_y_val + roi_top_offset
    roi_bottom_abs = min(interface_y_val + 250, sediment_y_val, frame_height -1)

    # visulize this roi_top_abs and roi_bottom_abs
    cv2.rectangle(frame_bgr, (0, roi_top_abs), (frame_bgr.shape[1], roi_bottom_abs), (0, 0, 255), 1)

    if roi_top_abs >= roi_bottom_abs:
        return None, np.zeros_like(gray_frame, dtype=np.uint8), None, frame_bgr

    roi_gray = gray_frame[roi_top_abs:roi_bottom_abs, :]
    if roi_gray.size == 0 or roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0 :
         return None, np.zeros_like(gray_frame, dtype=np.uint8), None, frame_bgr

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1  # ensure odd
    blurred_roi = cv2.GaussianBlur(roi_gray, (ksize, ksize), 2)

    edges_roi = cv2.Canny(blurred_roi, threshold1=canny_thresh1, threshold2=canny_thresh2)


    contours_roi, _ = cv2.findContours(edges_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    empty_mask = np.zeros_like(gray_frame, dtype=np.uint8)
    if not contours_roi:
        return None, empty_mask, None, frame_bgr

    candidate_contours = []
    for c_roi in contours_roi:
        x_r, y_r, w_r, h_r = cv2.boundingRect(c_roi)

        if h_r < min_ctr_h or w_r < min_ctr_w:
            continue

        if y_r > roi_gray.shape[0] * y_cutoff_ratio and y_r > 5:
            continue
    
        candidate_contours.append(c_roi)

    if not candidate_contours:
        return None, empty_mask, None, frame_bgr

    best_contour_roi = None
    max_score = -float('inf')

    w_area = 1.0
    w_proximity_x = 0.8
    w_proximity_y_apex = 0.5

    for c_roi in candidate_contours:
        area_roi = cv2.contourArea(c_roi)
        if area_roi < 10: continue

        current_score = w_area * area_roi

        if expected_center_x is not None:
            x_r_bbox, _, w_r_bbox, _ = cv2.boundingRect(c_roi)
            center_x_roi = x_r_bbox + w_r_bbox / 2.0
            current_score -= w_proximity_x * abs(center_x_roi - expected_center_x)

        if prev_metrics and 'apex' in prev_metrics and prev_metrics['apex'] is not None:
            if c_roi.shape[0] > 0 :
                temp_c_full_y = c_roi[:, 0, 1] + roi_top_abs
                if temp_c_full_y.size > 0:
                    current_tentative_apex_y = np.max(temp_c_full_y)
                    prev_apex_y = prev_metrics['apex'][1]
                    current_score -= w_proximity_y_apex * abs(current_tentative_apex_y - prev_apex_y)
        
        if current_score > max_score:
            max_score = current_score
            best_contour_roi = c_roi

    if best_contour_roi is None:
        return None, empty_mask, None, frame_bgr

    contour_ff = best_contour_roi.copy()
    contour_ff[:, 0, 1] += roi_top_abs

    closing_line_y_abs = roi_top_abs

    ys_ff = contour_ff[:, 0, 1]
    xs_ff = contour_ff[:, 0, 0]

    if len(xs_ff) < 2 :
        return None, empty_mask, None, frame_bgr
        
    min_x_contour_ff = np.min(xs_ff)
    max_x_contour_ff = np.max(xs_ff)

    base_points_ff = contour_ff[:, 0, :]
    sorted_base_points_ff = base_points_ff[np.argsort(base_points_ff[:, 0])]

    closed_poly_vertices = np.vstack([
        sorted_base_points_ff,
        [[max_x_contour_ff, closing_line_y_abs]],
        [[min_x_contour_ff, closing_line_y_abs]],
        [sorted_base_points_ff[0]]
    ]).astype(np.int32)

    closed_contour_cv_format = closed_poly_vertices.reshape((-1, 1, 2))

    cavity_mask_out = np.zeros_like(gray_frame, dtype=np.uint8)
    cv2.drawContours(cavity_mask_out, [closed_contour_cv_format], -1, 255, thickness=cv2.FILLED)

    actual_ys_in_poly = sorted_base_points_ff[:, 1]
    if len(actual_ys_in_poly) == 0:
        return None, cavity_mask_out, None, frame_bgr

    apex_y_abs = np.max(actual_ys_in_poly)
    apex_x_candidates = sorted_base_points_ff[actual_ys_in_poly == apex_y_abs][:, 0]
    apex_x_abs = np.mean(apex_x_candidates) if len(apex_x_candidates) > 0 else min_x_contour_ff

    depth = apex_y_abs - closing_line_y_abs
    width = max_x_contour_ff - min_x_contour_ff
    area = cv2.contourArea(closed_contour_cv_format)
    
    if depth < 0: depth = 0

    cavity_metrics_out = {
        'depth': float(depth),
        'width': float(width),
        'area': float(area),
        'apex': (int(apex_x_abs), int(apex_y_abs))
    }

    # # --- OPTIONAL DEPTH SMOOTHING FILTER ---
    # max_depth_delta = 30  # or 25; tune this
    # if prev_metrics and 'depth' in prev_metrics:
    #     prev_depth = prev_metrics['depth']
    #     if abs(cavity_metrics_out['depth'] - prev_depth) > max_depth_delta:
    #         print("[WARN] Depth jump too large; ignoring cavity this frame.")
    #         return None, empty_mask, prev_metrics, frame_bgr
    # # --- END FILTER ---


    if frame_bgr is not None:
        overlay = frame_bgr.copy()
        cavity_color_bgr = (148, 95, 234)
        
        cv2.drawContours(overlay, [closed_contour_cv_format], -1, cavity_color_bgr, thickness=cv2.FILLED)
        alpha = 0.4
        frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
        cv2.drawContours(frame_bgr, [closed_contour_cv_format], -1, cavity_color_bgr, thickness=2)

    return closed_contour_cv_format, cavity_mask_out, cavity_metrics_out, frame_bgr


fourcc = cv2.VideoWriter_fourcc(*'FFV1')
output_path = 'output_video_cavity_sediment_detection.avi'
out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (frame_width, frame_height))

# if SHOW_LIVE:
    # cv2.namedWindow('Sediment Binary Mask ROI', cv2.WINDOW_NORMAL)
    # # --- MODIFICATION: Add a window for the intermediate cavity mask for tuning ---
    # cv2.namedWindow('Intermediate Cavity Mask', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Intermediate Sediment+Cavity Mask', cv2.WINDOW_NORMAL)
    # --- END MODIFICATION ---

if SHOW_LIVE:
    cv2.namedWindow('Cavity Tuning')
    cv2.createTrackbar('Canny Thresh1', 'Cavity Tuning', 25, 255, lambda x: None)
    cv2.createTrackbar('Canny Thresh2', 'Cavity Tuning', 90, 255, lambda x: None)
    cv2.createTrackbar('Blur KSize', 'Cavity Tuning', 9, 31, lambda x: None)
    cv2.createTrackbar('Min Ctr W', 'Cavity Tuning', 12, 100, lambda x: None)
    cv2.createTrackbar('Min Ctr H', 'Cavity Tuning', 8, 100, lambda x: None)
    cv2.createTrackbar('Y Cutoff %', 'Cavity Tuning', 40, 100, lambda x: None)  # e.g., 40 = 40% down

    


while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        stored_frame = frame.copy()
    else:
        if stored_frame is None: continue
        frame = stored_frame.copy()

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_full_frame = cv2.GaussianBlur(gray, (9, 9), 2)

    # # === Step 1: Detect horizontal air–water interface ===
    # if interface_y is None:
    #     roi = blurred_full_frame[int(frame_height * 0.08):, :]
    #     edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    #     horizontal_lines = []
    #     if lines is not None:
    #         for line in lines:
    #             x1, y1, x2, y2 = line[0]
    #             angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
    #             if angle < 5:
    #                 horizontal_lines.append((x1, y1, x2, y2))
    #         if horizontal_lines:
    #             y_vals = [y1 for (_, y1, _, y2) in horizontal_lines] + [y2 for (_, y1, _, y2) in horizontal_lines]
    #             interface_y = int(np.median(y_vals))
    
    # if interface_y is not None:
    #     cv2.line(frame, (0, interface_y), (frame.shape[1], interface_y), (148, 95, 234), 2)
    #     cv2.putText(frame, "Air-Water", (10, interface_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 95, 234), 1, cv2.LINE_AA)

    # put air - water interface line manually
    interface_y = air_water_interface_y
    cv2.line(frame, (0, interface_y), (frame.shape[1], interface_y), (148, 95, 234), 2)
    cv2.putText(frame, "Air-Water", (10, interface_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 95, 234), 1, cv2.LINE_AA)

    cv2.line(frame, (0, sediment_interface_y), (frame.shape[1], sediment_interface_y), (255, 0, 0), 2)
    cv2.putText(frame, "Water-Sediment", (10, sediment_interface_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    # === Step 2: Detect Droplet (YOUR ORIGINAL LOGIC) ===
    if tracking_active:
        search_region_droplet_end_y = interface_y if interface_y is not None else frame_height // 2
        if search_region_droplet_end_y > 0:
            search_region_droplet = gray[0:search_region_droplet_end_y, :]
            blurred_droplet_search = cv2.GaussianBlur(search_region_droplet, (9,9), 2)
            circles = cv2.HoughCircles(blurred_droplet_search, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=30, param1=50, param2=30, minRadius=5, maxRadius=30)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                best_droplet = None; max_y_droplet = -1
                for (x_c, y_c_rel, r_c) in circles:
                    y_c_abs = y_c_rel
                    if y_c_abs > max_y_droplet : max_y_droplet = y_c_abs; best_droplet = (x_c, y_c_abs, r_c)
                if best_droplet:
                    (x, y, r) = best_droplet
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2); cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                    velocity = None
                    if prev_position is not None and prev_time is not None and timestamp > prev_time:
                        dy = y - prev_position[1]; dt = timestamp - prev_time
                        if dt > 1e-6: velocity = dy / dt
                        # if velocity is not None: cv2.putText(frame, f"Vel: {velocity:.1f} px/s", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    trajectory.append({'timestamp': timestamp, 'x': x, 'y': y, 'radius':r, 'velocity': velocity})
                    current_time_for_prev = timestamp; prev_position = (x, y); prev_time = current_time_for_prev

                    if interface_y is not None and (y + r) >= interface_y:
                        tracking_active = False
                        impact_frame = frame_idx
                        impact_x_coord = x
                        prev_cavity_stats = None
                        cavity_active = True

                        # Show contact label
                        cv2.putText(frame, "CONTACT", (x + 10, y + r + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)


    # === Step 3: Cavity detection post-impact (YOUR ORIGINAL LOGIC) ===
    if cavity_active and impact_frame is not None and interface_y is not None and frame_idx > impact_frame:
        current_expected_cavity_x = impact_x_coord if impact_x_coord is not None else (prev_cavity_stats['apex'][0] if prev_cavity_stats and 'apex' in prev_cavity_stats else None)
        
        cavity_contour_pts, _, cavity_metrics_data, frame_with_cavity = detect_cavity_contour(
            gray, interface_y, sediment_interface_y,
            expected_center_x=current_expected_cavity_x, prev_metrics=prev_cavity_stats, frame_bgr=frame,
            canny_thresh1=cv2.getTrackbarPos('Canny Thresh1', 'Cavity Tuning'),
            canny_thresh2=cv2.getTrackbarPos('Canny Thresh2', 'Cavity Tuning'),
            blur_ksize=cv2.getTrackbarPos('Blur KSize', 'Cavity Tuning'),
            min_ctr_w=cv2.getTrackbarPos('Min Ctr W', 'Cavity Tuning'),
            min_ctr_h=cv2.getTrackbarPos('Min Ctr H', 'Cavity Tuning'),
            y_cutoff_ratio=cv2.getTrackbarPos('Y Cutoff %', 'Cavity Tuning') / 100.0
        )

        
        if frame_with_cavity is not None: frame = frame_with_cavity

        # --- MODIFICATION: Use Kalman filter for smoother metrics ---
        if cavity_contour_pts is not None and cavity_metrics_data:
            prev_cavity_stats = cavity_metrics_data.copy()
            apex  = cavity_metrics_data['apex']; depth = cavity_metrics_data['depth']
            width = cavity_metrics_data['width']; area  = cavity_metrics_data['area']
            cv2.putText(frame, "Cavity", (apex[0] + 5, apex[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            cv2.line(frame, (apex[0], interface_y), apex, (0,255,255), 1)
            cv2.putText(frame, f"Depth: {depth:.1f}px", (apex[0] + 5, apex[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Width: {width:.1f}px", (apex[0] + 5, apex[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Area: {area:.1f}px^2",  (apex[0] + 5, apex[1] + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
            if apex[1] >= sediment_interface_y - 5: # tolerance
                cavity_active = False; cavity_contact_frame = frame_idx
                cv2.putText(frame, "CONTACT", (apex[0] + 5, apex[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)
    
    if frame_idx >= MANUAL_SEDIMENT_ANALYSIS_START_FRAME and not sediment_analysis_active:
        sediment_analysis_active = True
        print(f"--- Starting Sediment Analysis from frame {frame_idx} ---")

    full_frame_sediment_binary_mask = np.zeros_like(gray, dtype=np.uint8)
    # --- MODIFICATION: Prepare canvases for intermediate masks ---
    full_frame_intermediate_cavity_mask = np.zeros_like(gray, dtype=np.uint8)
    full_frame_intermediate_sed_plus_cav_mask = np.zeros_like(gray, dtype=np.uint8)
    # --- END MODIFICATION ---


    if sediment_analysis_active and interface_y is not None and sediment_interface_y is not None:
        roi_sed_top_factor = 0.5
        roi_sed_top = interface_y + int((sediment_interface_y - interface_y) * roi_sed_top_factor)
        roi_sed_bottom = sediment_interface_y - 3
        roi_sed_top = max(0, min(roi_sed_top, frame_height - 2))
        roi_sed_bottom = max(roi_sed_top + 1, min(roi_sed_bottom, frame_height - 1))

        if roi_sed_top < roi_sed_bottom:
            current_analysis_center_x = impact_x_coord if impact_x_coord is not None else frame_width // 2
            effective_roi_width = frame_width * 0.5
            if prev_cavity_stats and 'width' in prev_cavity_stats and prev_cavity_stats['width'] > 0:
                effective_roi_width = min(max(prev_cavity_stats['width'] * 1.5, frame_width * 0.2), frame_width * 0.8)
            roi_sed_left = int(max(0, current_analysis_center_x - effective_roi_width / 2))
            roi_sed_right = int(min(frame_width - 1, current_analysis_center_x + effective_roi_width / 2))

            if roi_sed_left < roi_sed_right:
                sediment_roi_gray = gray[roi_sed_top:roi_sed_bottom, roi_sed_left:roi_sed_right]
                
                if sediment_roi_gray.size > 0:
                    blurred_sed_roi = cv2.GaussianBlur(sediment_roi_gray, (5, 5), 0) # Blur once

                    # --- MODIFIED THRESHOLDING LOGIC ---
                    # Step 1: Get mask for (Sediment + Cavity) by thresholding out brighter water
                    # Pixels <= WATER_DARKS_SEPARATOR_THRESH become white
                    _, mask_sed_plus_cavity_roi = cv2.threshold(blurred_sed_roi, 
                                                              WATER_DARKS_SEPARATOR_THRESH, 
                                                              255, cv2.THRESH_BINARY_INV)
                    if SHOW_LIVE and mask_sed_plus_cavity_roi is not None : # For debug window
                         full_frame_intermediate_sed_plus_cav_mask[roi_sed_top:roi_sed_bottom, roi_sed_left:roi_sed_right] = mask_sed_plus_cavity_roi


                    # Step 2: Get mask for Cavity_Only (very dark regions)
                    # Pixels <= CAVITY_DARKNESS_THRESH become white
                    _, mask_cavity_only_roi = cv2.threshold(blurred_sed_roi, 
                                                          CAVITY_DARKNESS_THRESH, 
                                                          255, cv2.THRESH_BINARY_INV)
                    if SHOW_LIVE and mask_cavity_only_roi is not None: # For debug window
                        full_frame_intermediate_cavity_mask[roi_sed_top:roi_sed_bottom, roi_sed_left:roi_sed_right] = mask_cavity_only_roi
                    
                    # Step 3: Subtract cavity mask from (sediment + cavity) mask
                    # This assumes sediment is darker than WATER_DARKS_SEPARATOR_THRESH but brighter than CAVITY_DARKNESS_THRESH
                    # A pixel is sediment if it's in mask_sed_plus_cavity_roi AND NOT in mask_cavity_only_roi
                    sediment_mask_roi = cv2.bitwise_and(mask_sed_plus_cavity_roi, cv2.bitwise_not(mask_cavity_only_roi))
                    # --- END MODIFIED THRESHOLDING LOGIC ---

                    kernel_sed = np.ones((3,3), np.uint8) # Keep kernel relatively small initially
                    sediment_mask_roi = cv2.morphologyEx(sediment_mask_roi, cv2.MORPH_OPEN, kernel_sed, iterations=1)
                    sediment_mask_roi = cv2.morphologyEx(sediment_mask_roi, cv2.MORPH_CLOSE, kernel_sed, iterations=2) # Close can help fill bead centers if edges are detected

                    if sediment_mask_roi is not None and sediment_mask_roi.shape[0] > 0 and sediment_mask_roi.shape[1] > 0:
                        full_frame_sediment_binary_mask[roi_sed_top:roi_sed_bottom, roi_sed_left:roi_sed_right] = sediment_mask_roi
                    
                    disturbed_sediment_area_pixels = cv2.countNonZero(sediment_mask_roi)
                    max_disturbance_height_pixels = 0.0
                    contours_sed, _ = cv2.findContours(sediment_mask_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_sed:
                        min_y_contour_roi_overall = float('inf')
                        for c_sed in contours_sed:
                            if c_sed.shape[0] > 0:
                                y_c_roi = c_sed[:,0,1].min() 
                                if y_c_roi < min_y_contour_roi_overall: min_y_contour_roi_overall = y_c_roi
                        if min_y_contour_roi_overall != float('inf'):
                            highest_sediment_point_ff = roi_sed_top + min_y_contour_roi_overall
                            max_disturbance_height_pixels = float(sediment_interface_y - highest_sediment_point_ff)
                            if max_disturbance_height_pixels < 0: max_disturbance_height_pixels = 0.0
                    
                    sediment_data_list.append({
                        'frame': frame_idx, 'timestamp': timestamp,
                        'disturbed_area_px': disturbed_sediment_area_pixels,
                        'max_disturbance_height_px': max_disturbance_height_pixels
                    })

                    overlay_color = [0, 165, 255]; frame_roi_slice = frame[roi_sed_top:roi_sed_bottom, roi_sed_left:roi_sed_right]
                    condition = sediment_mask_roi > 0
                    frame_roi_slice[condition] = (frame_roi_slice[condition] * 0.5 + np.array(overlay_color, dtype=np.uint8) * 0.5).astype(np.uint8)
                    
                    text_x_sed = roi_sed_left + 5 if roi_sed_left + 150 < frame_width else 10
                    text_y_sed_start = roi_sed_top - 20 if roi_sed_top > 50 else sediment_interface_y + 20
                    if text_y_sed_start < 15 : text_y_sed_start = 15
                    if text_y_sed_start + 45 > frame_height: text_y_sed_start = frame_height - 45
                    cv2.putText(frame, "Sed. Analysis:", (text_x_sed, text_y_sed_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f" Area: {disturbed_sediment_area_pixels}", (text_x_sed, text_y_sed_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f" Max H: {max_disturbance_height_pixels:.1f}px", (text_x_sed, text_y_sed_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (roi_sed_left, roi_sed_top), (roi_sed_right, roi_sed_bottom), (0,255,0), 1)

    # if SHOW_LIVE :
    #     if sediment_analysis_active:
    #         cv2.imshow('Sediment Binary Mask ROI', full_frame_sediment_binary_mask)
    #         # --- MODIFICATION: Show intermediate masks ---
    #         cv2.imshow('Intermediate Cavity Mask', full_frame_intermediate_cavity_mask)
    #         cv2.imshow('Intermediate Sediment+Cavity Mask', full_frame_intermediate_sed_plus_cav_mask)
    #         # --- END MODIFICATION ---
    #     else: # Show empty masks if not active yet but windows are open
    #         empty_canvas = np.zeros_like(gray, dtype=np.uint8)
    #         cv2.imshow('Sediment Binary Mask ROI', empty_canvas)
    #         cv2.imshow('Intermediate Cavity Mask', empty_canvas)
    #         cv2.imshow('Intermediate Sediment+Cavity Mask', empty_canvas)


    cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 144, 28), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 144, 28), 1, cv2.LINE_AA)
    # bar_x, bar_y, bar_width, bar_height = 50, frame.shape[0] - 30, frame.shape[1] - 100, 20
    # if total_frames > 0:
    #     progress = int((frame_idx / total_frames) * bar_width)
    #     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    #     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), (0, 255, 0), -1)

    if SAVE_VIDEO:
        out.write(frame)

    if SHOW_LIVE:
        cv2.imshow('Droplet and Sediment Analysis', frame)
        key = cv2.waitKey(delay if delay > 0 else 1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused

cap.release()
out.release()
cv2.destroyAllWindows()

if trajectory: pd.DataFrame(trajectory).to_csv('trajectory_cavity_analysis.csv', index=False); print("Trajectory saved.")
if sediment_data_list: pd.DataFrame(sediment_data_list).to_csv('sediment_analysis_data.csv', index=False); print("Sediment analysis data saved.")