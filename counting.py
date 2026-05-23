import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count cars in an aerial parking-lot image using classical OpenCV."
    )
    parser.add_argument(
        "--image",
        default="input/parking_ori.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for visualization outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows after processing.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def clamp_box(box, shape):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def save_color_space_exploration(image_bgr, output_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    items = [
        ("RGB", image_rgb, None),
        ("Grayscale", gray, "gray"),
        ("HSV - Hue", hsv[:, :, 0], "gray"),
        ("HSV - Saturation", hsv[:, :, 1], "gray"),
        ("HSV - Value", hsv[:, :, 2], "gray"),
        ("LAB - L", lab[:, :, 0], "gray"),
        ("LAB - A", lab[:, :, 1], "gray"),
        ("LAB - B", lab[:, :, 2], "gray"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (title, image, cmap) in zip(axes.ravel(), items):
        axis.set_title(title)
        axis.axis("off")
        axis.imshow(image, cmap=cmap)

    fig.tight_layout()
    fig.savefig(output_path / "01_color_space_exploration.png", dpi=130)
    plt.close(fig)


def build_parking_line_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 150), (179, 80, 255))
    white_edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(
        white_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=160,
        maxLineGap=30,
    )

    line_mask = np.zeros_like(gray)
    if lines is None:
        return line_mask

    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)
        length = float(np.hypot(dx, dy))

        if length > 150 and (angle < 8 or abs(angle - 90) < 8):
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 13)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(line_mask, kernel, iterations=1)


def build_evidence_mask(image_bgr, line_mask):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    _, saturation, value = cv2.split(hsv)
    lightness = lab[:, :, 0]

    # Large blur approximates the local asphalt background.
    background = cv2.GaussianBlur(gray, (0, 0), 55)
    local_difference = cv2.absdiff(gray, background)

    contrast_mask = local_difference > 42
    color_mask = (saturation > 45) & (value > 45)
    dark_glass_mask = (lightness < 75) & ((local_difference > 25) | (saturation > 35))

    evidence = (contrast_mask | color_mask | dark_glass_mask).astype(np.uint8) * 255
    evidence = cv2.bitwise_and(evidence, cv2.bitwise_not(line_mask))

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_OPEN, open_kernel, iterations=1)
    evidence = cv2.dilate(evidence, dilate_kernel, iterations=1)
    return evidence


def extract_local_maxima(score_map, threshold, suppress_width, suppress_height):
    working = score_map.copy()
    map_height, map_width = working.shape
    peaks = []

    while True:
        _, max_value, _, max_location = cv2.minMaxLoc(working)
        if max_value < threshold:
            break

        x, y = max_location
        peaks.append((x, y, float(max_value)))

        x1 = max(0, x - suppress_width // 2)
        x2 = min(map_width, x + suppress_width // 2)
        y1 = max(0, y - suppress_height // 2)
        y2 = min(map_height, y + suppress_height // 2)
        working[y1:y2, x1:x2] = 0

    return peaks


def detect_horizontal_cars(image_bgr, evidence_mask):
    height, width = image_bgr.shape[:2]

    # The window is estimated from image size, not from per-car hand mapping.
    # It represents the approximate footprint of a horizontal car in this view.
    window_width = max(80, int(round(width * 0.136)))
    window_height = max(45, int(round(height * 0.096)))
    suppress_width = int(round(window_width * 1.43))
    suppress_height = int(round(window_height * 1.43))

    evidence_float = evidence_mask.astype(np.float32) / 255.0
    score_map = cv2.boxFilter(
        evidence_float,
        ddepth=-1,
        ksize=(window_width, window_height),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )

    peaks = extract_local_maxima(
        score_map,
        threshold=0.70,
        suppress_width=suppress_width,
        suppress_height=suppress_height,
    )

    detections = []
    for x, y, score in peaks:
        box = clamp_box(
            (
                x - window_width // 2,
                y - window_height // 2,
                x + window_width // 2,
                y + window_height // 2,
            ),
            image_bgr.shape,
        )
        detections.append(
            {
                "label": "car",
                "box": box,
                "score": score,
                "source": "sliding-window",
            }
        )

    return detections, score_map


def detect_red_cars(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 70, 80), (12, 255, 255))
    red_high = cv2.inRange(hsv, (170, 70, 80), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
        iterations=2,
    )

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    image_height, image_width = image_bgr.shape[:2]

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = h / max(1, w)

        if contour_area < image_width * image_height * 0.001:
            continue
        if aspect < 1.5:
            continue

        pad_x = int(w * 0.08)
        pad_y = int(h * 0.04)
        box = clamp_box((x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y), image_bgr.shape)
        detections.append(
            {
                "label": "red car",
                "box": box,
                "score": float(contour_area),
                "source": "red-contour",
            }
        )

    return detections, red_mask


def save_score_map(score_map, output_path):
    normalized = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path / "04_sliding_window_score_map.jpg"), heatmap)


def save_final_visualization(image_bgr, detections, output_path):
    visual = image_bgr.copy()

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["box"]
        color = (0, 255, 0) if detection["source"] == "sliding-window" else (0, 128, 255)
        cv2.rectangle(visual, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            visual,
            str(index),
            (x1 + 8, y1 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.putText(
        visual,
        f"Count: {len(detections)}",
        (1260, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (0, 0, 255),
        5,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path / "05_detected_cars.jpg"), visual)


def save_masks(line_mask, evidence_mask, red_mask, output_path):
    cv2.imwrite(str(output_path / "02_parking_line_mask.jpg"), line_mask)
    cv2.imwrite(str(output_path / "03_car_evidence_mask.jpg"), evidence_mask)
    cv2.imwrite(str(output_path / "03_red_car_mask.jpg"), red_mask)


def main():
    args = parse_args()
    output_path = ensure_output_dir(args.output)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    save_color_space_exploration(image_bgr, output_path)

    line_mask = build_parking_line_mask(image_bgr)
    evidence_mask = build_evidence_mask(image_bgr, line_mask)

    horizontal_detections, score_map = detect_horizontal_cars(image_bgr, evidence_mask)
    red_detections, red_mask = detect_red_cars(image_bgr)

    detections = horizontal_detections + red_detections
    detections.sort(key=lambda item: (item["box"][1], item["box"][0]))

    save_masks(line_mask, evidence_mask, red_mask, output_path)
    save_score_map(score_map, output_path)
    save_final_visualization(image_bgr, detections, output_path)

    print(f"Input image: {args.image}")
    print(f"Detected horizontal cars: {len(horizontal_detections)}")
    print(f"Detected red cars: {len(red_detections)}")
    print(f"Total cars: {len(detections)}")
    print(f"Outputs saved to: {output_path}")

    if args.show:
        final_image = cv2.cvtColor(
            cv2.imread(str(output_path / "05_detected_cars.jpg")),
            cv2.COLOR_BGR2RGB,
        )
        plt.figure(figsize=(14, 8))
        plt.imshow(final_image)
        plt.axis("off")
        plt.title(f"Detected cars: {len(detections)}")
        plt.show()


if __name__ == "__main__":
    main()
