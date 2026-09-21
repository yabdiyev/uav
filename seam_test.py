"""Compare a straight join with a minimum-cost horizontal seam.
Uses ordinary ORB and one similarity alignment for BOTH outputs.
This experiment is for vertically displaced frames, not arbitrary panoramas.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, default=Path('frames'))
    parser.add_argument('--output', type=Path, default=Path('seam_results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    later = cv2.imread(str(args.frames / 'frame_0285.jpg'))
    earlier = cv2.imread(str(args.frames / 'frame_0270.jpg'))
    if later is None or earlier is None:
        raise RuntimeError('Could not load frames 0270 and 0285.')

    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(later, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(earlier, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        raise RuntimeError('Not enough features.')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)
    if len(matches) < 4:
        raise RuntimeError('Not enough matches.')
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(0)
    A, inliers = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.99,
    )
    if A is None or not np.isfinite(A).all():
        raise RuntimeError('Alignment failed.')
    print(f'Alignment inliers: {int(inliers.sum())}/{len(matches)}')

    # Use the earlier frame's canvas for this seam-only experiment.
    height, width = earlier.shape[:2]
    warped = cv2.warpAffine(later, A, (width, height))
    coverage = cv2.warpAffine(
        np.full(later.shape[:2], 255, np.uint8), A, (width, height)
    ) == 255

    # Restrict the seam to a central band, away from uncovered borders.
    y0, y1 = height // 2 - 180, height // 2 + 181
    valid_columns = np.flatnonzero(coverage[y0:y1].all(axis=0))
    if len(valid_columns) < width * 0.9:
        raise RuntimeError('Insufficient central overlap for this experiment.')
    x0, x1 = int(valid_columns[0]), int(valid_columns[-1]) + 1
    if not coverage[y0:y1, x0:x1].all():
        raise RuntimeError('Central overlap is not rectangular.')

    # Low cost means the two images agree at that location.
    difference = np.abs(warped.astype(np.float32) - earlier.astype(np.float32)).mean(axis=2)
    # Average over a small neighborhood rather than trusting one matching pixel.
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    cost = difference[y0:y1, x0:x1]
    local_seam = minimum_seam(cost)
    seam = np.empty(width, dtype=np.int32)
    seam[x0:x1] = local_seam + y0
    seam[:x0] = seam[x0]
    seam[x1:] = seam[x1 - 1]
    straight = np.full(width, height // 2, dtype=np.int32)

    def compose(boundary):
        result = earlier.copy()
        # Later frame supplies the upper region; earlier supplies the lower.
        use_later = coverage & (np.arange(height)[:, None] <= boundary[None, :])
        result[use_later] = warped[use_later]
        return result

    baseline, optimized = compose(straight), compose(seam)
    paths = optimized.copy()
    cv2.line(paths, (x0, height // 2), (x1 - 1, height // 2), (0, 0, 255), 2)
    pts = np.column_stack((np.arange(x0, x1), seam[x0:x1])).astype(np.int32)
    cv2.polylines(paths, [pts.reshape(-1, 1, 2)], False, (0, 255, 0), 2)

    outputs = {'straight_join.jpg': baseline, 'optimized_join.jpg': optimized,
               'seam_paths.jpg': paths}
    # Full-width close-up of the search band makes the join easier to compare.
    crops = []
    for name, result in [('Straight join', baseline), ('Optimized seam', optimized)]:
        crop = result[y0:y1].copy()
        cv2.rectangle(crop, (0, 0), (310, 38), (0, 0, 0), -1)
        cv2.putText(crop, name, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        crops.append(crop)
    outputs['comparison.jpg'] = np.vstack(crops)
    for name, result in outputs.items():
        if not cv2.imwrite(str(args.output / name), result):
            raise RuntimeError(f'Could not save {name}')
    xs = np.arange(x0, x1)
    baseline_cost = float(difference[straight[xs], xs].mean())
    optimized_cost = float(difference[seam[xs], xs].mean())
    print(f'Mean local color disagreement along straight join: {baseline_cost:.2f}/255')
    print(f'Mean local color disagreement along optimized seam: {optimized_cost:.2f}/255')
    print('This measures seam appearance, not geometric accuracy.')
    print(f'Output folder: {args.output.resolve()}')


if __name__ == '__main__':
    main()