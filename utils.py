import cv2
import os
from tkinter import filedialog, Tk

def extract_frames(video_path):
    # Create a folder name based on video file name (no extension)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = f"{video_name}_frames"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break  # No more frames to read

        # Save every 3rd frame as PNG
        if frame_count % 3 == 0:
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Done! Saved {saved_count} frames (every 3rd) to '{output_folder}'")

# --- Run via file picker ---
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.avi")])
    if video_file:
        extract_frames(video_file)
