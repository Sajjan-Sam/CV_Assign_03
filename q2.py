# ============================================================
# Q2: Slow-Motion Frame Interpolation using Dense Optical Flow
# ============================================================
# Generate intermediate "slow-motion" frames between real frames
#       using dense (pyramid-based) Lucas–Kanade optical flow.
# ------------------------------------------------------------
# Concept:
#   1. Compute forward and backward optical flow between two frames.
#   2. Warp both frames halfway along their respective flows.
#   3. Average the two warped results → creates a new middle (interpolated) frame.
#   4. Repeat for all frame pairs to produce a smooth slow-motion video.
# ------------------------------------------------------------

import os, cv2, numpy as np

# Input video and algorithm parameters
VIDEO = "video1.mp4"     # Input video file
ASSUME_FPS = 30.0         # Default FPS if not detected
LEVELS = 3                # Number of pyramid levels (coarse-to-fine flow)
WIN = 7                   # Local window size for LK (Lucas–Kanade)
EIG_T = 1e-4              # Eigenvalue threshold for reliable corners
NUM_PAIRS = 12            # Number of frame pairs to process
START_AT = 0              # Start frame index
OUTDIR = "q2_output"      # Folder to store results
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def to_gray_f32(img):
    """Convert BGR image to grayscale float32 (range 0–1)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

def conv2(img, k):
    """2D convolution helper using border replication."""
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE)

def grads(img):
    """Compute spatial gradients Ix and Iy using Sobel-like filters."""
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/8.0
    ky = kx.T
    Ix = conv2(img, kx)
    Iy = conv2(img, ky)
    return Ix, Iy

def gaussian_pyr(img, L):
    """Build Gaussian pyramid (for multi-scale optical flow)."""
    pyr = [img]
    for _ in range(1, L):
        blur = cv2.GaussianBlur(pyr[-1], (5,5), 1.0, borderType=cv2.BORDER_REPLICATE)
        small = cv2.resize(blur, (blur.shape[1]//2, blur.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        pyr.append(small)
    return pyr

def mask_pyr(mask, L):
    """Generate a mask pyramid matching the image pyramid resolution."""
    mp = [mask.astype(np.float32)]
    for _ in range(1, L):
        m = cv2.resize(mp[-1], (mp[-1].shape[1]//2, mp[-1].shape[0]//2), interpolation=cv2.INTER_NEAREST)
        mp.append(m)
    return mp

def up2(flow, shape):
    """Upsample optical flow from a coarser level to the next finer level."""
    up = cv2.resize(flow, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
    return up

def warp(img, flow):
    """Warp an image using optical flow (x,y displacement fields)."""
    H, W = img.shape
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (gx + flow[...,0]).astype(np.float32)
    map_y = (gy + flow[...,1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# ------------------------------------------------------------
# Dense LK at Single Level
# ------------------------------------------------------------
def lk_dense_one_level(I0, I1, init_flow, win, eig_t, mask=None):
    """
    Compute dense optical flow for one pyramid level.
    Iteratively updates flow vectors (u,v) using the Lucas–Kanade formulation.
    """
    u = init_flow[...,0].copy()
    v = init_flow[...,1].copy()
    Ix, Iy = grads(I0)

    # Iterative refinement (3 passes)
    for _ in range(3):
        I1w = warp(I1, np.dstack([u, v]))  # warp I1 towards I0
        It = I1w - I0                      # temporal derivative

        # Pre-compute second moment matrices
        Ix2, Iy2, Ixy = Ix*Ix, Iy*Iy, Ix*Iy
        Ixt, Iyt       = Ix*It, Iy*It

        Sxx = cv2.boxFilter(Ix2, -1, (win,win))
        Syy = cv2.boxFilter(Iy2, -1, (win,win))
        Sxy = cv2.boxFilter(Ixy, -1, (win,win))
        Sxt = cv2.boxFilter(Ixt, -1, (win,win))
        Syt = cv2.boxFilter(Iyt, -1, (win,win))

        # Solve for flow increments using the 2×2 normal equations
        det = Sxx*Syy - Sxy*Sxy
        tr  = Sxx + Syy
        lmin = tr/2 - np.sqrt(np.maximum(0.0,(tr/2)**2 - det))
        good = (det > 1e-9) & (lmin > eig_t)

        du = np.zeros_like(u)
        dv = np.zeros_like(v)
        du[good] = (-Sxt[good]*Syy[good] + Sxy[good]*Syt[good]) / det[good]
        dv[good] = (-Syt[good]*Sxx[good] + Sxy[good]*Sxt[good]) / det[good]

        # Apply mask to ignore irrelevant regions (outside ROI)
        if mask is not None:
            du *= mask
            dv *= mask

        # Update flow
        u += du
        v += dv

    return np.dstack([u, v])

# ------------------------------------------------------------
# Multi-level Dense LK (Pyramidal)
# ------------------------------------------------------------
def lk_dense(I0, I1, levels, win, eig_t, mask_fullres=None):
    """Compute optical flow using coarse-to-fine Lucas–Kanade."""
    pyr0 = gaussian_pyr(I0, levels)
    pyr1 = gaussian_pyr(I1, levels)

    # Build mask pyramid
    mp = None
    if mask_fullres is not None:
        mp = mask_pyr(mask_fullres, levels)

    flow = np.zeros((pyr0[-1].shape[0], pyr0[-1].shape[1], 2), np.float32)
    for lvl in reversed(range(levels)):  # process from coarse → fine
        I0l = pyr0[lvl]; I1l = pyr1[lvl]
        flow = up2(flow, I0l.shape)
        ml = None if mp is None else mp[lvl]
        flow = lk_dense_one_level(I0l, I1l, flow, win, eig_t, mask=ml)
    return flow

# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: fps = ASSUME_FPS  # fallback if FPS not found

    # Read frames
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_AT)
    for _ in range(NUM_PAIRS+1):
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("Not enough frames in video")

    H, W = frames[0].shape[:2]

    # -----------------------------
    # Define Fixed ROI (Region of Interest)
    # -----------------------------
    roi_x1 = int(W * 0.35); roi_y1 = int(H * 0.65)
    roi_x2 = int(W * 0.50); roi_y2 = int(H * 0.75)
    mask = np.zeros((H, W), np.float32)
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1.0  # focus only on moving person area

    # Convert to grayscale for processing
    gray = [to_gray_f32(f) for f in frames]

    # Create output writer
    writer = cv2.VideoWriter(os.path.join(OUTDIR, "slowmo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    # Process consecutive frame pairs
    for i in range(len(gray)-1):
        I0, I1 = gray[i], gray[i+1]

        # Forward and backward dense flow
        fwd = lk_dense(I0, I1, LEVELS, WIN, EIG_T, mask_fullres=mask)
        bwd = lk_dense(I1, I0, LEVELS, WIN, EIG_T, mask_fullres=mask)

        # Interpolate middle frame using half warps
        half_fwd = warp(I0, 0.5*fwd)
        half_bwd = warp(I1, 0.5*bwd)
        middle = 0.5*(half_fwd + half_bwd)

        # Visual diagnostics (difference maps)
        before = np.abs(I1 - I0)
        after  = np.abs(middle - 0.5*(I0 + I1))

        # Convert middle frame for saving
        mm = (middle*255.0).clip(0,255).astype(np.uint8)
        mid_bgr = cv2.cvtColor(mm, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mid_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,255,0), 2)

        # Save results
        cv2.imwrite(os.path.join(OUTDIR, f"pair_{i:04d}_middle.png"), mid_bgr)
        cv2.imwrite(os.path.join(OUTDIR, f"pair_{i:04d}_before_absdiff.png"),
                    (before/np.max(before+1e-6)*255).astype(np.uint8))
        cv2.imwrite(os.path.join(OUTDIR, f"pair_{i:04d}_after_absdiff.png"),
                    (after/np.max(after+1e-6)*255).astype(np.uint8))

        # Write frames to video output (original + interpolated)
        writer.write(frames[i])
        writer.write(mid_bgr)

    # Write last original frame
    writer.write(frames[-1])
    writer.release()

if __name__ == "__main__":
    main()
