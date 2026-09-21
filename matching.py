import cv2
import numpy as np

image1 = cv2.imread("frames/frame_0285.jpg")
image2 = cv2.imread("frames/frame_0270.jpg")

if image1 is None or image2 is None:
    raise RuntimeError("Could not load the images. Check their paths.")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

def detect_grid_features(gray):
    rows, cols = 4, 4
    height, width = gray.shape
    detector = cv2.ORB_create(nfeatures=125)

    all_keypoints = []
    descriptor_blocks = []

    for row in range(rows):
        for col in range(cols):
            y0 = row * height // rows
            y1 = (row + 1) * height // rows
            x0 = col * width // cols
            x1 = (col + 1) * width // cols

            cell = gray[y0:y1, x0:x1]
            keypoints, descriptors = detector.detectAndCompute(
                cell, None
            )

            if descriptors is None:
                continue

            # Convert cell coordinates into full-image coordinates.
            for point in keypoints:
                x, y = point.pt
                point.pt = (x + x0, y + y0)

            all_keypoints.extend(keypoints)
            descriptor_blocks.append(descriptors)

    if not descriptor_blocks:
        raise RuntimeError("No features found.")

    return all_keypoints, np.vstack(descriptor_blocks)


keypoints1, descriptors1 = detect_grid_features(gray1)
keypoints2, descriptors2 = detect_grid_features(gray2)

print("Features in first image:", len(keypoints1))
print("Features in second image:", len(keypoints2))

preview = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

if not cv2.imwrite("features_grid.jpg", preview):
    raise RuntimeError("Could not save features_grid.jpg.")

if descriptors1 is None or descriptors2 is None:
    raise RuntimeError("No descriptors found in one of the images.")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda match: match.distance)
print("Candidate matches:", len(matches))

match_preview = cv2.drawMatches(
    image1, keypoints1,
    image2, keypoints2,
    matches[:30],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

if not cv2.imwrite("matches.jpg", match_preview):
    raise RuntimeError("Could not save matches.jpg.")

if len(matches) < 4:
    raise RuntimeError("Not enough matches to estimate a homography.")

# Extract the pixel coordinates for each matched pair.
points1 = np.float32([
    keypoints1[m.queryIdx].pt for m in matches
]).reshape(-1, 1, 2)

points2 = np.float32([
    keypoints2[m.trainIdx].pt for m in matches
]).reshape(-1, 1, 2)

# Estimate a transformation from image 1 to image 2.
A, mask = cv2.estimateAffinePartial2D(
    points1,
    points2,
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0,
    maxIters=5000,
    confidence=0.99,
)

if A is None:
    raise RuntimeError("Could not estimate an alignment.")

H = np.eye(3, dtype=np.float64)
H[:2, :] = A

if H is None or mask is None:
    raise RuntimeError("Could not estimate an alignment.")

inlier_count = int(mask.sum())

print(f"Inliers: {inlier_count} / {len(matches)}")
print(f"Inlier percentage: {100 * inlier_count / len(matches):.1f}%")
print("Homography matrix:")
print(H)

height, width = image2.shape[:2]

# Map image 1 into image 2's coordinate system.
aligned = cv2.warpPerspective(image1, H, (width, height))

# Show both images together with equal weight.
overlay = cv2.addWeighted(aligned, 0.5, image2, 0.5, 0)

if not cv2.imwrite("alignment_overlay_grid.jpg", overlay):
    raise RuntimeError("Could not save alignment_overlay_grid.jpg.")

before = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

if not cv2.imwrite("before_alignment.jpg", before):
    raise RuntimeError("Could not save before_alignment.jpg.")

# 1. Get the corners of each image.
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]

corners1 = np.float32([
    [0, 0], [w1, 0], [w1, h1], [0, h1]
]).reshape(-1, 1, 2)

corners2 = np.float32([
    [0, 0], [w2, 0], [w2, h2], [0, h2]
]).reshape(-1, 1, 2)

# 2. Find where image 1's corners land after alignment.
warped_corners1 = cv2.perspectiveTransform(corners1, H)

# 3. Find the boundaries needed to contain both images.
all_corners = np.concatenate((warped_corners1, corners2), axis=0)

xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

canvas_width = int(xmax - xmin)
canvas_height = int(ymax - ymin)

# Guard against an unexpectedly large canvas from a bad transform.
if (
    canvas_width <= 0 or canvas_height <= 0
    or canvas_width * canvas_height > 4 * (w1 * h1 + w2 * h2)
):
    raise RuntimeError("Unexpected canvas size. Check the homography.")

# 4. Shift both images so all coordinates fit inside the canvas.
offset_x = int(-xmin)
offset_y = int(-ymin)

T = np.float64([
    [1, 0, offset_x],
    [0, 1, offset_y],
    [0, 0, 1],
])

# 5. Place the transformed first image on the larger canvas.
mosaic = cv2.warpPerspective(
    image1, T @ H, (canvas_width, canvas_height)
)

# 6. Place image 2, using its pixels wherever the images overlap.
mosaic[
    offset_y:offset_y + h2,
    offset_x:offset_x + w2
] = image2

if not cv2.imwrite("two_frame_mosaic.jpg", mosaic):
    raise RuntimeError("Could not save the mosaic.")

print(f"Mosaic size: {canvas_width} wide × {canvas_height} tall")