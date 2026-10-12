"""
Generates a small SYNTHETIC demo dataset of procedurally-drawn "industrial
surface" images with and without defects, purely so the application has
something to run against out of the box.

These images are NOT real industrial photographs and are NOT a substitute
for a real labeled dataset. Every image produced here is written into
backend/storage/demo_dataset/ and every filename is prefixed to make its
synthetic origin obvious. The frontend/backend both label results computed
from these images as coming from demo data.

Usage:
    python scripts/generate_demo_dataset.py --count 40
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "storage" / "demo_dataset"


def base_surface(rng: np.random.Generator, size=(400, 400), base_gray=175, noise_sigma=6) -> np.ndarray:
    img = np.full((*size, 3), base_gray, dtype=np.uint8)
    noise = rng.normal(0, noise_sigma, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # faint brushed-metal style horizontal streaks
    for _ in range(30):
        y = rng.integers(0, size[0])
        shade = int(rng.integers(-6, 6))
        cv2.line(img, (0, y), (size[1], y), (base_gray + shade,) * 3, 1)
    return img


def add_scratch(img, rng):
    h, w = img.shape[:2]
    p1 = (rng.integers(0, w), rng.integers(0, h))
    p2 = (rng.integers(0, w), rng.integers(0, h))
    shade = int(rng.integers(40, 90))
    cv2.line(img, p1, p2, (shade, shade, shade), rng.integers(1, 3))
    return img, "scratch"


def add_crack(img, rng):
    h, w = img.shape[:2]
    x, y = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4)
    shade = int(rng.integers(30, 70))
    pts = [(x, y)]
    for _ in range(6):
        x += rng.integers(-25, 25)
        y += rng.integers(-25, 25)
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (shade, shade, shade), 2)
    return img, "crack"


def add_dent(img, rng):
    h, w = img.shape[:2]
    center = (rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4))
    radius = int(rng.integers(15, 35))
    shade = int(rng.integers(90, 130))
    overlay = img.copy()
    cv2.circle(overlay, center, radius, (shade, shade, shade), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    return img, "dent"


def add_hole(img, rng):
    h, w = img.shape[:2]
    center = (rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4))
    radius = int(rng.integers(8, 18))
    cv2.circle(img, center, radius, (20, 20, 20), -1)
    return img, "hole"


def add_contamination(img, rng):
    h, w = img.shape[:2]
    center = (rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4))
    for _ in range(rng.integers(20, 40)):
        dx, dy = rng.integers(-20, 20), rng.integers(-20, 20)
        pt = (np.clip(center[0] + dx, 0, w - 1), np.clip(center[1] + dy, 0, h - 1))
        shade = int(rng.integers(20, 60))
        cv2.circle(img, pt, rng.integers(1, 4), (shade, shade, shade), -1)
    return img, "contamination"


DEFECT_FNS = [add_scratch, add_crack, add_dent, add_hole, add_contamination]


def main(count: int, defect_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    normal_dir = OUT_DIR / "normal"
    defect_dir = OUT_DIR / "defective"
    normal_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img = base_surface(rng)
        if random.Random(seed + i).random() < defect_ratio:
            fn = DEFECT_FNS[i % len(DEFECT_FNS)]
            img, label = fn(img, rng)
            cv2.imwrite(str(defect_dir / f"demo_{label}_{i:03d}.png"), img)
        else:
            cv2.imwrite(str(normal_dir / f"demo_normal_{i:03d}.png"), img)

    print(f"Synthetic demo dataset written to: {OUT_DIR}")
    print("NOTE: These are procedurally generated images for demonstration only,")
    print("not real industrial photographs. Do not present results on this data")
    print("as a validated accuracy benchmark.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--defect-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.count, args.defect_ratio, args.seed)
