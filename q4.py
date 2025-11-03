# ==============================================================
# Q4: Vanishing Point Detection in Video Frames
# ==============================================================
#  Objective:
#   Detect vanishing points in video frames by identifying
#   dominant line directions using Hough Transform and computing
#   their intersection. The script visualizes detected lines,
#   left/right lane approximations, and smoothed vanishing point.
# --------------------------------------------------------------
#   Outputs:
#     1. Annotated video with detected vanishing point.
#     2. Snapshot frame with marked lines and vanishing point.
#     3. Plot of vanishing point coordinates over time.
# ==============================================================

import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Parse Command Line Arguments
# -----------------------------
def parse_args():
    """
    Parses command line arguments for flexibility.
    You can change input video, output folder, and key parameters.
    """
    ap = argparse.ArgumentParser(description="Vanishing-point detector for video02.mp4")
    ap.add_argument("--video", type=str, default="video02.mp4", help="input video file")
    ap.add_argument("--outdir", type=str, default="vp_out", help="output folder path")
    ap.add_argument("--canny", nargs=2, type=int, default=[80, 160], metavar=("LOW","HIGH"))
    ap.add_argument("--min_len", type=int, default=70, help="minimum line length for Hough")
    ap.add_argument("--max_gap", type=int, default=25, help="max gap between line segments")
    ap.add_argument("--angle_min_deg", type=float, default=15.0, help="min line angle in degrees")
    ap.add_argument("--angle_max_deg", type=float, default=85.0, help="max line angle in degrees")
    ap.add_argument("--ema", type=float, default=0.15, help="Exponential Moving Average smoothing factor")
    ap.add_argument("--snapshot", type=int, default=60, help="frame index to save snapshot image")
    return ap.parse_args()

# -----------------------------
# STEP 2: Line Equation Helpers
# -----------------------------
def line_params(x1, y1, x2, y2):
    """
    Compute normalized line parameters (a, b, c)
    for the line ax + by + c = 0.
    """
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    n = np.hypot(a, b) + 1e-9
    return a/n, b/n, c/n

# -----------------------------
# STEP 3: Least Squares VP Estimation
# -----------------------------
def least_squares_vp(lines_abc):
    """
    Solve for vanishing point (x, y) using least squares
    from multiple line equations.
    """
    A = np.array([[a,b] for (a,b,_) in lines_abc], dtype=np.float64)
    c = np.array([c for (_,_,c) in lines_abc], dtype=np.float64)
    AtA = A.T @ A
    Atc = A.T @ c
    if np.linalg.cond(AtA) > 1e6:  # avoid unstable inversion
        return None
    return -np.linalg.solve(AtA, Atc)

# -----------------------------
# STEP 4: Robust RANSAC VP Estimation (backup)
# -----------------------------
def ransac_vp(lines_abc, trials=600, tol=4.0):
    """
    Use random sampling consensus (RANSAC) to find
    a robust vanishing point resistant to outlier lines.
    """
    n = len(lines_abc)
    if n < 2: return None
    best, best_in = None, -1
    for _ in range(trials):
        i, j = np.random.randint(0, n, 2)
        if i == j: continue
        a1,b1,c1 = lines_abc[i]; a2,b2,c2 = lines_abc[j]
        d = a1*b2 - a2*b1
        if abs(d) < 1e-9: continue
        x = (b1*c2 - b2*c1)/d
        y = (c1*a2 - c2*a1)/d
        dists = np.abs(np.array([a*x + b*y + c for (a,b,c) in lines_abc]))
        cnt = int((dists < tol).sum())
        if cnt > best_in:
            best_in, best = cnt, np.array([x,y], np.float64)
    return best

# -----------------------------
# STEP 5: Linear Fit Helpers
# -----------------------------
def fit_line_ls(pts):
    """
    Fit line y = m*x + b through a set of points using least squares.
    Used for estimating left/right lane approximations.
    """
    if len(pts) < 2: return None
    P = np.asarray(pts, np.float64)
    X, Y = P[:,0], P[:,1]
    A = np.c_[X, np.ones_like(X)]
    m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m), float(b)

def extend_and_draw(img, m, b, color, thickness=2):
    """
    Extend a fitted line to full image boundaries and draw it.
    """
    H, W = img.shape[:2]
    pts = []

    # Compute intersection with image borders
    y_left = m*0 + b
    y_right = m*(W-1) + b
    if 0 <= y_left < H: pts.append((0, int(round(y_left))))
    if 0 <= y_right < H: pts.append((W-1, int(round(y_right))))
    if abs(m) > 1e-9:
        x_top = (0 - b) / m
        x_bot = ((H-1) - b) / m
        if 0 <= x_top < W: pts.append((int(round(x_top)), 0))
        if 0 <= x_bot < W: pts.append((int(round(x_bot)), H-1))
    if len(pts) < 2: return

    P = np.array(pts, dtype=np.int32)
    d2 = np.sum((P[None,:,:] - P[:,None,:])**2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    if i == j: return
    cv2.line(img, tuple(P[i]), tuple(P[j]), color, thickness, cv2.LINE_AA)

# -----------------------------
# STEP 6: Main Function
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = os.path.join(args.outdir, "vp_demo.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Define angle limits to keep near-vertical lines
    theta_min = np.deg2rad(args.angle_min_deg)
    theta_max = np.deg2rad(args.angle_max_deg)

    ema = None  # for smoothed vanishing point
    saved_snap = False
    frame_idx = 0
    vp_history = []  # store vanishing points for plotting

    # -----------------------------
    # STEP 7: Frame Processing Loop
    # -----------------------------
    while True:
        ok, frame = cap.read()
        if not ok: break

        # Preprocess frame: grayscale + contrast enhancement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Detect edges using Canny
        edges = cv2.Canny(gray, args.canny[0], args.canny[1], L2gradient=True)

        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=args.min_len, maxLineGap=args.max_gap)

        kept, left_pts, right_pts = [], [], []
        if lines is not None:
            # Filter lines based on angle and length
            for x1,y1,x2,y2 in lines[:,0,:]:
                dx, dy = x2 - x1, y2 - y1
                theta = np.pi/2 if dx == 0 else abs(np.arctan2(dy, dx))
                if theta < theta_min or theta > theta_max: continue
                if np.hypot(dx, dy) < args.min_len: continue
                kept.append((x1,y1,x2,y2))
                m = (dy/dx) if dx != 0 else 1e6
                (left_pts if m < 0 else right_pts).extend([(x1,y1),(x2,y2)])

        # Convert lines to normal form ax + by + c = 0
        lines_abc = [line_params(*seg) for seg in kept]

        # Compute vanishing point (VP)
        vp = None
        if len(lines_abc) >= 2:
            vp = least_squares_vp(lines_abc)
            if vp is None or not np.isfinite(vp).all():
                vp = ransac_vp(lines_abc, trials=700, tol=5.0)

        # -----------------------------
        # STEP 8: Visualization
        # -----------------------------
        vis = frame.copy()
        for (x1,y1,x2,y2) in kept[:200]:
            cv2.line(vis, (x1,y1), (x2,y2), (80,255,80), 1, cv2.LINE_AA)

        L = fit_line_ls(left_pts)
        R = fit_line_ls(right_pts)
        if L and np.all(np.isfinite(L)): extend_and_draw(vis, *L, (0,200,0), 3)
        if R and np.all(np.isfinite(R)): extend_and_draw(vis, *R, (0,200,0), 3)

        if vp is not None and np.isfinite(vp).all():
            ema = vp if ema is None else (1.0 - args.ema) * ema + args.ema * vp
            cx, cy = int(np.clip(ema[0], 0, W-1)), int(np.clip(ema[1], 0, H-1))
            cv2.circle(vis, (cx, cy), 8, (0,0,255), -1)
            cv2.putText(vis, "Vanishing Point", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
            vp_history.append(ema.copy())

        # Write frame to video
        writer.write(vis)

        # Save snapshot at selected frame
        if not saved_snap and frame_idx == args.snapshot:
            cv2.imwrite(os.path.join(args.outdir, f"vp_frame_{frame_idx:05d}.png"), vis)
            saved_snap = True

        frame_idx += 1

    # -----------------------------
    # STEP 9: Save Outputs
    # -----------------------------
    cap.release()
    writer.release()
    print(f"[OK] Saved annotated video -> {out_path}")
    if saved_snap:
        print(f"[OK] Snapshot saved at frame {args.snapshot}")

    # -----------------------------
    # STEP 10: Plot VP Coordinates Over Time
    # -----------------------------
    if len(vp_history) > 0:
        vp_history = np.array(vp_history)
        plt.figure(figsize=(6,3))
        plt.plot(vp_history[:,0], label="VP X", color="blue")
        plt.plot(vp_history[:,1], label="VP Y", color="orange")
        plt.xlabel("Frame Index")
        plt.ylabel("Pixel Position")
        plt.title("Vanishing Point Coordinates Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "vp_over_time.png"), dpi=150)
        print("[OK] Saved plot -> vp_out/vp_over_time.png")

# -----------------------------
# Run the Script
# -----------------------------
if __name__ == "__main__":
    main()
