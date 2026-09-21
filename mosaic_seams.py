"""Sequence seam experiment. Alignment copied from mosaic.py unchanged.
Assumes monotonically upward motion in the reference canvas.
Search bands are disjoint to keep neighboring seams from crossing.
Narrow side borders retain the center-based baseline where a full path
through shared valid coverage is unavailable. No blending is applied.
"""
from pathlib import Path

import numpy as np
import cv2

# Zero-padded filenames sort into chronological order.
import argparse

parser = argparse.ArgumentParser(description="Ordered horizontal seams for this vertical-flight video")
parser.add_argument("--frames", type=Path, default=Path("frames"))
parser.add_argument("--output", type=Path, default=Path("sequence_seam_results"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
frame_paths = sorted(args.frames.glob("frame_*.jpg"))

if len(frame_paths) < 2:
    raise RuntimeError("Need at least two images in the frames folder.")

orb = cv2.ORB_create(nfeatures=2000)
frames = []

for path in frame_paths:
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Could not read {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 4:
        raise RuntimeError(f"Not enough features in {path.name}")

    frames.append({
        "name": path.name,
        "image": image,
        "keypoints": keypoints,
        "descriptors": descriptors,
    })

    print(f"{path.name}: {len(keypoints)} features")

print(f"Ready to match {len(frames)} frames.")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
pair_transforms = []

for i in range(1, len(frames)):
    previous = frames[i - 1]
    current = frames[i]

    # Match FROM the current frame TO the previous frame.
    matches = matcher.match(
        current["descriptors"],
        previous["descriptors"],
    )

    if len(matches) < 4:
        raise RuntimeError(f"Too few matches for {current['name']}")

    current_points = np.float32([
        current["keypoints"][m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    previous_points = np.float32([
        previous["keypoints"][m.trainIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    A, mask = cv2.estimateAffinePartial2D(
        current_points,
        previous_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99,
    )

    if A is None:
        raise RuntimeError(f"Alignment failed for {current['name']}")

    # Convert the 2×3 matrix into the 3×3 format our pipeline uses.
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A

    if H is None or mask is None or not np.isfinite(H).all():
        raise RuntimeError(f"Alignment failed for {current['name']}")

    inliers = int(mask.sum())
    ratio = inliers / len(matches)

    pair_transforms.append(H)

    print(
        f"{current['name']} -> {previous['name']}: "
        f"{inliers}/{len(matches)} inliers ({ratio:.1%})"
    )

print(f"Estimated {len(pair_transforms)} neighboring transformations.")

# Frame 0000 is our reference, so its transformation is identity.
global_transforms = [np.eye(3, dtype=np.float64)]

for H in pair_transforms:
    combined = global_transforms[-1] @ H

    if not np.isfinite(combined).all():
        raise RuntimeError("Invalid combined transformation.")

    if abs(combined[2, 2]) < 1e-10:
        raise RuntimeError("Cannot normalize combined transformation.")

    combined = combined / combined[2, 2]
    global_transforms.append(combined)

# Find every frame's corners in the reference coordinate system.
transformed_corners = []

for frame, G in zip(frames, global_transforms):
    height, width = frame["image"].shape[:2]

    corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ]).reshape(-1, 1, 2)

    # Check that the projective mapping stays finite over the image.
    xy = corners.reshape(-1, 2)
    denominators = (
        G[2, 0] * xy[:, 0] + G[2, 1] * xy[:, 1] + G[2, 2]
    )

    if not (
        np.all(denominators > 1e-6)
        or np.all(denominators < -1e-6)
    ):
        raise RuntimeError(f"Unstable transformation: {frame['name']}")

    warped_corners = cv2.perspectiveTransform(corners, G)

    if not np.isfinite(warped_corners).all():
        raise RuntimeError(f"Invalid corners: {frame['name']}")

    transformed_corners.append(warped_corners)

all_corners = np.concatenate(transformed_corners, axis=0)

xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

canvas_width = int(xmax - xmin)
canvas_height = int(ymax - ymin)

print(f"Full canvas: {canvas_width} wide × {canvas_height} tall")
print(f"Canvas origin in frame 0000 coordinates: ({xmin}, {ymin})")

# Keep accidental oversized allocations from a bad transformation in check.
if (
    canvas_width <= 0 or canvas_height <= 0
    or canvas_width * canvas_height > 30_000_000
):
    raise RuntimeError("Unexpected canvas size. Inspect the transformations.")

# Shift reference coordinates into the output canvas.
T = np.float64([
    [1, 0, -xmin],
    [0, 1, -ymin],
    [0, 0, 1],
])



def minimum_seam(cost):
    """Find a left-to-right path; each step changes row by at most one."""
    height, width = cost.shape
    total = cost[:, 0].copy()
    parents = np.zeros((height, width), dtype=np.int8)
    for x in range(1, width):
        choices = np.stack((
            np.r_[np.inf, total[:-1]],
            total,
            np.r_[total[1:], np.inf],
        ))
        best = np.argmin(choices, axis=0)
        parents[:, x] = best - 1
        total = cost[:, x] + choices[best, np.arange(height)]
    y = int(np.argmin(total))
    if not np.isfinite(total[y]):
        raise RuntimeError('No valid seam across the overlap.')
    seam = np.empty(width, dtype=np.int32)
    seam[-1] = y
    for x in range(width - 1, 0, -1):
        y += int(parents[y, x])
        seam[x - 1] = y
    return seam


# Warp each original only once; retain the center-based baseline for comparison.
size = (canvas_width, canvas_height)
shape = (canvas_height, canvas_width)
baseline = np.zeros((*shape, 3), np.uint8)
best_score = np.zeros(shape, np.float32)
warps, masks, centers = [], [], []

for frame, G in zip(frames, global_transforms):
    image = frame['image']
    h, w = image.shape[:2]
    M = T @ G
    warped = cv2.warpPerspective(image, M, size)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size) == 255
    rows = (np.arange(h, dtype=np.float32) + .5) / h
    scores = np.repeat((1 - np.abs(2 * rows - 1))[:, None], w, axis=1)
    scores = cv2.warpPerspective(scores, M, size)
    # Nearest-neighbor mask matches the existing baseline implementation.
    base_mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size,
                                   flags=cv2.INTER_NEAREST) > 0
    use = base_mask & (scores > best_score)
    baseline[use] = warped[use]
    best_score[use] = scores[use]
    center = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), M)[0, 0]
    centers.append(float(center[1]))
    warps.append(warped)
    masks.append(mask)

centers = np.array(centers)
if not np.all(np.diff(centers) < -4):
    raise RuntimeError('This experiment requires steadily upward motion; inspect frame centers.')

# Nominal joins lie midway between adjacent frame centers.
nominal = (centers[:-1] + centers[1:]) / 2
# Each seam gets its own non-overlapping row band, preventing crossings.
bands = []
for i, middle in enumerate(nominal):
    gaps = []
    if i > 0:
        gaps.append(nominal[i - 1] - middle)
    if i + 1 < len(nominal):
        gaps.append(middle - nominal[i + 1])
    half = min(100, .45 * min(gaps)) if gaps else 100
    lo, hi = int(np.ceil(middle - half)), int(np.floor(middle + half)) + 1
    if lo < 0 or hi > canvas_height or hi <= lo:
        raise RuntimeError('Invalid seam search band.')
    bands.append((lo, hi))

# Use one common x interval for every seam. Preserve baseline at side borders.
columns_valid = np.ones(canvas_width, dtype=bool)
for i, (lo, hi) in enumerate(bands):
    columns_valid &= (masks[i][lo:hi] & masks[i + 1][lo:hi]).all(axis=0)
# Select the longest contiguous valid interval.
edges = np.diff(np.r_[False, columns_valid, False].astype(np.int8))
starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
if len(starts) == 0:
    raise RuntimeError('No shared valid seam interval.')
k = int(np.argmax(stops - starts))
x0, x1 = int(starts[k]), int(stops[k])
if x1 - x0 < .9 * canvas_width:
    raise RuntimeError('Too little shared width for a horizontal seam experiment.')

seams, straight_scores, seam_scores = [], [], []
for i, (lo, hi) in enumerate(bands):
    # Include two rows of padding for the small Gaussian neighborhood.
    a, b = max(0, lo - 2), min(canvas_height, hi + 2)
    diff = np.abs(warps[i][a:b].astype(np.float32)
                  - warps[i + 1][a:b].astype(np.float32)).mean(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    cost = diff[lo - a:hi - a, x0:x1]
    seam = minimum_seam(cost) + lo
    if seams and not np.all(seam < seams[-1]):
        raise RuntimeError('Seams crossed.')
    seams.append(seam)
    xs = np.arange(x1 - x0)
    straight_score = float(cost[int(round(nominal[i])) - lo].mean())
    seam_score = float(cost[seam - lo, xs].mean())
    straight_scores.append(straight_score)
    seam_scores.append(seam_score)
    print(f'Seam {i + 1:02d}: straight {straight_score:.2f}, optimized {seam_score:.2f}')

# Start with baseline, then assign ordered strips in the common interval.
result = baseline.copy()
ygrid = np.arange(canvas_height)[:, None]
source_map = np.full(shape, -1, np.int16)
for i in range(len(frames)):
    selected = np.ones((canvas_height, x1 - x0), bool)
    if i < len(seams):
        selected &= ygrid > seams[i][None, :]
    if i > 0:
        selected &= ygrid <= seams[i - 1][None, :]
    selected &= masks[i][:, x0:x1]
    result[:, x0:x1][selected] = warps[i][:, x0:x1][selected]
    source_map[:, x0:x1][selected] = i

# Every baseline-covered interior pixel must have an explicitly assigned source.
covered = np.logical_or.reduce(masks)
missing = covered[:, x0:x1] & (source_map[:, x0:x1] < 0)
# Subpixel differences at outer support edges can leave tiny coverage slivers.
# Fill these from any valid covering frame, never from an empty warp border.
fallback_count = int(missing.sum())
for i in range(len(frames)):
    fill = missing & masks[i][:, x0:x1]
    result[:, x0:x1][fill] = warps[i][:, x0:x1][fill]
    source_map[:, x0:x1][fill] = i
    missing[fill] = False
if missing.any():
    raise RuntimeError('Unfilled coverage remains.')
print(f'Coverage-edge fallback pixels: {fallback_count}')

marked = result.copy()
for seam in seams:
    pts = np.column_stack((np.arange(x0, x1), seam)).astype(np.int32)
    cv2.polylines(marked, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 1)

for name, img in [('full_mosaic_seams.jpg', result),
                  ('seam_paths_full.jpg', marked),
                  ('center_baseline.jpg', baseline)]:
    if not cv2.imwrite(str(args.output / name), img):
        raise RuntimeError(f'Could not save {name}')
np.save(args.output / 'seam_rows.npy', np.array(seams))
print(f'All {len(seams)} seams ordered; all covered interior pixels assigned.')
print(f'Seam width: {x1 - x0}/{canvas_width}; baseline retained at side borders.')
print(f'Mean straight seam cost: {np.mean(straight_scores):.2f}/255')
print(f'Mean optimized seam cost: {np.mean(seam_scores):.2f}/255')
print('Seam cost is not geometric accuracy or an independent quality score.')
print(f'Saved results in {args.output.resolve()}')
