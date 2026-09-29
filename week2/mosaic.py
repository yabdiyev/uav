from pathlib import Path

import numpy as np
import cv2

# Zero-padded filenames sort into chronological order.
frame_paths = sorted(Path("frames").glob("frame_*.jpg"))

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

mosaic = np.zeros(
    (canvas_height, canvas_width, 3),
    dtype=np.uint8,
)

# Track which frame offers the best central coverage at each pixel.
best_score = np.zeros(
    (canvas_height, canvas_width),
    dtype=np.float32,
)

for frame, G in zip(frames, global_transforms):
    image = frame["image"]
    height, width = image.shape[:2]
    transform = T @ G

    warped = cv2.warpPerspective(
        image, transform, (canvas_width, canvas_height)
    )

    # High scores near the middle row; low scores near top and bottom.
    row_positions = (np.arange(height, dtype=np.float32) + 0.5) / height
    row_scores = 1.0 - np.abs(2.0 * row_positions - 1.0)

    source_scores = np.repeat(
        row_scores[:, None], width, axis=1
    )

    warped_scores = cv2.warpPerspective(
        source_scores,
        transform,
        (canvas_width, canvas_height),
    )

    source_mask = np.full((height, width), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        source_mask,
        transform,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST,
    )

    use_pixels = (
        (warped_mask > 0)
        & (warped_scores > best_score)
    )

    mosaic[use_pixels] = warped[use_pixels]
    best_score[use_pixels] = warped_scores[use_pixels]

    print(f"Placed {frame['name']}")

if not cv2.imwrite("full_mosaic_center.jpg", mosaic):
    raise RuntimeError("Could not save full_mosaic_center.jpg.")

print("Saved full_mosaic_center.jpg")