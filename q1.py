# ==============================================================
# Q1: ROI-Based Motion Estimation using Lucas–Kanade Optical Flow
# ==============================================================
# Author: Sajjan Singh
# Objective:
#   Detect and analyze motion (speed and direction) of an object
#   within a fixed region of interest (ROI) in a video sequence.
#   Compute pixel-wise displacement using Lucas–Kanade optical flow
#   and estimate mean/median/max speed trends across time.
# --------------------------------------------------------------
#   Output:
#     1. Flow visualization frames (arrows showing motion)
#     2. Speed graph (pixels/second vs. time)
# ==============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# -----------------------------
# STEP 1: Load video & get properties
# -----------------------------
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Extract FPS, width, height, total frames
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if FPS unavailable

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded: {width}x{height} at {fps} fps")
print(f"Total frames: {total_frames}")

# Create output directory
os.makedirs('./q1_output', exist_ok=True)

# -----------------------------
# STEP 2: Gradient computation
# -----------------------------
def get_gradients(img1, img2):
    """
    Compute spatial (Ix, Iy) and temporal (It) gradients.
    These gradients are the building blocks of optical flow.
    """
    kernel_x = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernel_y = np.array([[-1, -1], [1, 1]]) * 0.25
    kernel_t = np.array([[1, 1], [1, 1]]) * 0.25

    Ix = cv2.filter2D(img1, -1, kernel_x) + cv2.filter2D(img2, -1, kernel_x)
    Iy = cv2.filter2D(img1, -1, kernel_y) + cv2.filter2D(img2, -1, kernel_y)
    It = cv2.filter2D(img2, -1, kernel_t) - cv2.filter2D(img1, -1, kernel_t)
    return Ix, Iy, It

# -----------------------------
# STEP 3: Lucas–Kanade Optical Flow
# -----------------------------
def lk_optical_flow(img1, img2, window_size=15, threshold=0.01):
    """
    Compute optical flow vectors (u, v) between two consecutive frames
    using the Lucas–Kanade method (small window least-squares).
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Compute gradients
    Ix, Iy, It = get_gradients(img1, img2)

    # Create sampling grid of points
    step = 20
    y_coords, x_coords = np.mgrid[step:img1.shape[0]-step:step,
                                  step:img1.shape[1]-step:step]
    points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    # Initialize arrays for flow and status
    u = np.zeros(len(points))
    v = np.zeros(len(points))
    status = np.zeros(len(points), dtype=bool)

    half_win = window_size // 2

    # Process each point
    for i, (x, y) in enumerate(points):
        y1, y2 = y - half_win, y + half_win + 1
        x1, x2 = x - half_win, x + half_win + 1

        # Skip if window goes out of image
        if y1 < 0 or y2 >= img1.shape[0] or x1 < 0 or x2 >= img1.shape[1]:
            continue

        # Extract local patch
        Ix_win = Ix[y1:y2, x1:x2].ravel()
        Iy_win = Iy[y1:y2, x1:x2].ravel()
        It_win = It[y1:y2, x1:x2].ravel()

        # Build least-squares system A·[u,v] = b
        A = np.column_stack([Ix_win, Iy_win])
        b = -It_win
        ATA = A.T @ A

        # Check reliability based on eigenvalues (good corners)
        eigenvalues = np.linalg.eigvals(ATA)
        if np.min(eigenvalues) > threshold:
            try:
                flow = np.linalg.solve(ATA, A.T @ b)
                u[i], v[i] = flow
                status[i] = True
            except:
                pass
    return points, u, v, status

# -----------------------------
# STEP 4: Draw flow visualization
# -----------------------------
def draw_flow_arrows(frame, points, u, v, status, scale=1.0):
    """
    Overlay arrows showing motion direction and magnitude.
    """
    vis = frame.copy()
    for pt, flow_x, flow_y, is_good in zip(points, u, v, status):
        if not is_good:
            continue
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)  # start point
        end_x = int(x + flow_x * scale)
        end_y = int(y + flow_y * scale)
        cv2.arrowedLine(vis, (x, y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
    return vis

# -----------------------------
# STEP 5: Define Region of Interest (ROI)
# -----------------------------
roi_x1 = int(width * 0.35)
roi_y1 = int(height * 0.65)
roi_x2 = int(width * 0.50)
roi_y2 = int(height * 0.75)
print(f"ROI set to: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")

# -----------------------------
# STEP 6: Process each frame
# -----------------------------
frame_numbers, mean_speeds, median_speeds, max_speeds = [], [], [], []
sample_frames, sample_indices = [], []
prev_frame = None
frame_count = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Processing frames...")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Convert to grayscale & extract ROI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    # Compute optical flow with previous frame
    if prev_frame is not None:
        points, u, v, status = lk_optical_flow(prev_frame, roi_gray)
        good_points = points[status]
        good_u, good_v = u[status], v[status]

        if len(good_u) > 0:
            # Compute magnitude (speed) of motion vectors
            speeds = np.sqrt(good_u**2 + good_v**2)
            mean_speeds.append(np.mean(speeds))
            median_speeds.append(np.median(speeds))
            max_speeds.append(np.max(speeds))
            frame_numbers.append(frame_count)

            # Save visualization every 30 frames
            if frame_count % 30 == 0:
                roi_points = good_points + np.array([roi_x1, roi_y1])
                vis_frame = draw_flow_arrows(frame, roi_points, good_u, good_v,
                                             np.ones(len(good_u), dtype=bool), scale=3.0)
                sample_frames.append(vis_frame)
                sample_indices.append(frame_count)

    prev_frame = roi_gray.copy()

    if frame_count % 100 == 0:
        print(f"Processed {frame_count}/{total_frames} frames")

cap.release()
elapsed = time.time() - start_time
print(f"Processing finished in {elapsed:.1f} seconds")

# -----------------------------
# STEP 7: Plot and Save Speed Graph
# -----------------------------
if len(frame_numbers) > 0:
    time_seconds = np.array(frame_numbers) / fps
    speed_pixels_per_sec = np.array(mean_speeds) * fps

    plt.figure(figsize=(14, 6))
    plt.plot(time_seconds, speed_pixels_per_sec, label='Mean Speed', linewidth=2, color='blue')
    plt.fill_between(time_seconds,
                     np.array(median_speeds) * fps,
                     np.array(max_speeds) * fps,
                     alpha=0.3, label='Speed Range (Median–Max)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speed (pixels/second)')
    plt.title('Object Speed Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./q1_output/speed_graph.png', dpi=150)
    plt.close()
    print("Speed graph saved → ./q1_output/speed_graph.png")

# -----------------------------
# STEP 8: Save Sample Flow Frames
# -----------------------------
for i, (vis_frame, frame_idx) in enumerate(zip(sample_frames[:3], sample_indices[:3])):
    filename = f'./q1_output/flow_frame_{i+1}.jpg'
    cv2.imwrite(filename, vis_frame)
    print(f"Saved optical flow frame {i+1}")

# -----------------------------
# STEP 9: Display Summary Statistics
# -----------------------------
if len(frame_numbers) > 0:
    print("\n--- Motion Statistics ---")
    print(f"Average speed: {np.mean(speed_pixels_per_sec):.2f} pixels/sec")
    print(f"Max speed: {np.max(speed_pixels_per_sec):.2f} pixels/sec")
    print(f"Min speed: {np.min(speed_pixels_per_sec):.2f} pixels/sec")
    print(f"Total frames analyzed: {len(frame_numbers)}")
