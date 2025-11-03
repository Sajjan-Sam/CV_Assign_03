# ==============================================================
# Q3: Video Stabilization using Sparse Optical Flow
# ==============================================================
#  Objective:
#   Reduce camera shake from a handheld or moving video by
#   estimating inter-frame translations using sparse Lucas–Kanade
#   optical flow and compensating for those with smoothing.
# --------------------------------------------------------------
#   Steps:
#     1. Compute sparse optical flow between consecutive frames.
#     2. Estimate per-frame median translation (dx, dy).
#     3. Accumulate these into a camera motion path.
#     4. Smooth the path with a moving average filter.
#     5. Apply inverse correction to stabilize the video.
# ==============================================================

import os, cv2, numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Parameters and Setup
# -----------------------------
VIDEO = "video02.mp4"     # Input shaky video
ASSUME_FPS = 30.0          # Fallback FPS if not found
GRID_STRIDE = 24           # Distance between sampling points (sparser = faster)
WIN = 9                    # Window size for optical flow neighborhood
SMOOTH = 15                # Moving-average smoothing window (must be odd)
OUTDIR = "q3_output"       # Output directory
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# STEP 2: Helper Functions
# -----------------------------
def to_gray_f32(bgr):
    """Convert BGR image to normalized grayscale (float32)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

def conv2(img, k):
    """2D convolution with border replication."""
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)

def grads(prev):
    """Compute spatial gradients Ix, Iy using Sobel-like filters."""
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/8.0
    ky = kx.T
    return conv2(prev, kx), conv2(prev, ky)

def window_sums(img, n):
    """Compute local sums over an n×n window (for LK matrix terms)."""
    return cv2.boxFilter(img, -1, (n,n), normalize=False)

# -----------------------------
# STEP 3: Sparse Lucas–Kanade Optical Flow
# -----------------------------
def lk_sparse(prev, curr, stride=24, win=9, eig_t=1e-3):
    """
    Compute sparse optical flow on a regular grid.
    Returns feature points and corresponding (u,v) flow vectors.
    """
    Ix, Iy = grads(prev)
    It = curr - prev

    # Precompute second-moment matrices for each pixel
    Ix2, Iy2, Ixy = Ix*Ix, Iy*Iy, Ix*Iy
    Ixt, Iyt = Ix*It, Iy*It
    Sxx, Syy, Sxy = window_sums(Ix2, win), window_sums(Iy2, win), window_sums(Ixy, win)
    Sxt, Syt = window_sums(Ixt, win), window_sums(Iyt, win)

    H, W = prev.shape
    pts, flow = [], []

    # Sample grid points and compute local optical flow
    for y in range(win//2, H-win//2, stride):
        for x in range(win//2, W-win//2, stride):
            A11, A22, A12 = Sxx[y,x], Syy[y,x], Sxy[y,x]
            b1, b2 = -Sxt[y,x], -Syt[y,x]
            det = A11*A22 - A12*A12
            if det <= 0: continue
            tr = A11 + A22
            lmin = tr/2 - np.sqrt(max(0.0, (tr/2)**2 - det))
            if lmin < eig_t: continue

            # Solve 2x2 system for flow (u,v)
            u = (-b1*A22 + A12*b2)/det
            v = (-b2*A11 + A12*b1)/det
            pts.append((x,y)); flow.append((u,v))

    if not pts:
        return np.zeros((0,2)), np.zeros((0,2))
    return np.array(pts, np.float32), np.array(flow, np.float32)

# -----------------------------
# STEP 4: Moving Average Smoothing
# -----------------------------
def moving_average(x, k):
    """Smooth a 1D path using a moving average of length k."""
    pad = k//2
    xx = np.pad(x, (pad,pad), mode="edge")
    ker = np.ones(k, dtype=np.float32)/k
    return np.convolve(xx, ker, mode="valid")

# -----------------------------
# STEP 5: Main Function
# -----------------------------
def main():
    cap = cv2.VideoCapture(VIDEO)
    ok, f0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read video file")

    H, W = f0.shape[:2]
    frames = [f0]

    # Read all frames into memory
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    # Convert frames to grayscale for flow computation
    gray = [to_gray_f32(f) for f in frames]

    # -------------------------------------------------
    # STEP 6: Estimate Per-Frame Median Translation
    # -------------------------------------------------
    dx, dy = [], []
    example_vis = None

    for i in range(len(gray)-1):
        pts, flow = lk_sparse(gray[i], gray[i+1],
                              stride=GRID_STRIDE, win=WIN)
        if pts.shape[0] == 0:
            dx.append(0.0); dy.append(0.0)
            continue

        # Median flow gives camera translation between frames
        med = np.median(flow, axis=0)
        dx.append(med[0]); dy.append(med[1])

        # Save one example visualization (first pair)
        if example_vis is None:
            vis = frames[i].copy()
            for (x,y),(u,v) in zip(pts, flow):
                p0 = (int(x), int(y))
                p1 = (int(x+u*4), int(y+v*4))
                cv2.arrowedLine(vis, p0, p1, (0,255,0), 1, tipLength=0.3)
            cv2.imwrite(os.path.join(OUTDIR, "flow_example.jpg"), vis)
            example_vis = True

    # -------------------------------------------------
    # STEP 7: Compute and Smooth Camera Path
    # -------------------------------------------------
    path_x = np.cumsum([0.0]+dx)
    path_y = np.cumsum([0.0]+dy)
    path_x, path_y = np.array(path_x), np.array(path_y)

    # Smooth the camera path to remove high-frequency shake
    sx = moving_average(path_x, SMOOTH)
    sy = moving_average(path_y, SMOOTH)

    # -------------------------------------------------
    # STEP 8: Plot Raw vs. Smoothed Paths
    # -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # Headless (no GUI)
    plt.figure(figsize=(8,4))
    plt.plot(path_x, label="Raw X")
    plt.plot(sx, label="Smoothed X")
    plt.plot(path_y, label="Raw Y")
    plt.plot(sy, label="Smoothed Y")
    plt.legend()
    plt.title("Camera Translation (Raw vs Smoothed)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "path_raw_vs_smooth.png"), dpi=200)

    # -------------------------------------------------
    # STEP 9: Apply Inverse Correction (Stabilization)
    # -------------------------------------------------
    # The correction = difference between smoothed and raw paths.
    corr_x = sx - path_x
    corr_y = sy - path_y

    # Side-by-side output (original + stabilized)
    side = cv2.VideoWriter(os.path.join(OUTDIR, "stabilized_side_by_side.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W*2, H))
    stabilized = []

    for i, fr in enumerate(frames):
        tx, ty = corr_x[i], corr_y[i]
        M = np.float32([[1,0,tx],[0,1,ty]])  # translation matrix
        stab = cv2.warpAffine(fr, M, (W,H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

        both = np.hstack([fr, stab])
        side.write(both)
        stabilized.append(stab)

    side.release()
    print("Stabilized video saved in:", OUTDIR)

# -----------------------------
# Run the main program
# -----------------------------
if __name__ == "__main__":
    main()
