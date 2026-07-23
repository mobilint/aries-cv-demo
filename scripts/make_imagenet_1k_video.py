#!/usr/bin/env python3
"""Create a 1,000-class ImageNet validation video without downloading the full dataset.

By default this script uses Hugging Face `datasets` streaming with image decoding
disabled, so rows are consumed sequentially and selected image bytes are written
directly to the video. The previous Dataset Server API backend is still available
for debugging/comparison via `--backend api`.

Default output:
  output/imagenet_1k_random_10frames_30fps_528x320.mp4
  output/imagenet_1k_random_10frames_30fps_528x320_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from huggingface_hub import get_token
from PIL import Image, ImageOps
from tqdm import tqdm


DATASET_SERVER = "https://datasets-server.huggingface.co"


@dataclass
class Sample:
    row_idx: int
    label: int
    label_name: str
    image_url: str = ""
    src_width: int | None = None
    src_height: int | None = None
    image_bytes: bytes | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one ImageNet-1K validation image per class and make a 30 FPS video."
    )
    parser.add_argument("--dataset", default="ILSVRC/imagenet-1k")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames-per-image", type=int, default=10)
    parser.add_argument("--width", type=int, default=528)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--backend",
        choices=("streaming", "api"),
        default="streaming",
        help="ImageNet access backend. 'streaming' avoids slow Dataset Server row pagination.",
    )
    parser.add_argument(
        "--output",
        default="output/imagenet_1k_random_10frames_30fps_528x320.mp4",
    )
    parser.add_argument(
        "--manifest",
        default="output/imagenet_1k_random_10frames_30fps_528x320_manifest.csv",
    )
    parser.add_argument(
        "--metadata-cache",
        default="output/imagenet_1k_random_10frames_30fps_528x320_selected_rows.csv",
        help="CSV cache for the selected 1,000 rows. Reused if present unless --refresh-selection is set.",
    )
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None, help="Debug only: stop metadata scan early.")
    parser.add_argument(
        "--full-reservoir-scan",
        action="store_true",
        help="Scan all metadata rows for unbiased reservoir sampling. By default stop once all classes are covered.",
    )
    parser.add_argument(
        "--keep-cached-urls",
        action="store_true",
        help="Use signed image URLs stored in the selection cache. By default URLs are refreshed by row_idx.",
    )
    return parser.parse_args()


def request_json(session: requests.Session, url: str, timeout: float, max_retries: int, retry_sleep: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retries need broad handling
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}")


def request_bytes(session: requests.Session, url: str, timeout: float, max_retries: int, retry_sleep: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"Failed to download image from {url[:120]}...: {last_error}")


def make_session() -> requests.Session:
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face token not found. Run `huggingface-cli login` first.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def load_label_names(session: requests.Session, args: argparse.Namespace) -> list[str]:
    url = (
        f"{DATASET_SERVER}/first-rows?dataset={args.dataset}"
        f"&config={args.config}&split={args.split}"
    )
    data = request_json(session, url, args.timeout, args.max_retries, args.retry_sleep)
    for feature in data.get("features", []):
        if feature.get("name") == "label":
            names = feature.get("type", {}).get("names")
            if isinstance(names, list) and names:
                return [str(x) for x in names]
    raise RuntimeError("Could not find label names in Dataset Server response.")


def rows_url(args: argparse.Namespace, offset: int, length: int) -> str:
    return (
        f"{DATASET_SERVER}/rows?dataset={args.dataset}"
        f"&config={args.config}&split={args.split}&offset={offset}&length={length}"
    )


def first_rows_url(args: argparse.Namespace) -> str:
    return (
        f"{DATASET_SERVER}/first-rows?dataset={args.dataset}"
        f"&config={args.config}&split={args.split}"
    )


def row_to_sample(row_obj: dict[str, Any], label_names: list[str]) -> Sample | None:
    row_idx = int(row_obj["row_idx"])
    row = row_obj.get("row", {})
    label = int(row["label"])
    image = row.get("image", {})
    image_url = image.get("src")
    if not image_url:
        return None
    label_name = label_names[label] if 0 <= label < len(label_names) else f"class_{label}"
    return Sample(
        row_idx=row_idx,
        label=label,
        label_name=label_name,
        image_url=image_url,
        src_width=image.get("width"),
        src_height=image.get("height"),
    )


def select_one_per_class(session: requests.Session, args: argparse.Namespace, label_names: list[str]) -> list[Sample]:
    """Reservoir-sample one row per class while streaming metadata pages."""
    rng = random.Random(args.seed)
    selected: dict[int, Sample] = {}
    counts: dict[int, int] = {}
    offset = 0
    pbar = tqdm(total=args.num_classes, desc="classes covered", unit="class")

    while True:
        if args.limit_rows is not None and offset >= args.limit_rows:
            break

        if offset == 0 and args.page_size <= 100:
            data = request_json(session, first_rows_url(args), args.timeout, args.max_retries, args.retry_sleep)
            rows = data.get("rows", [])[: args.page_size]
        else:
            data = request_json(session, rows_url(args, offset, args.page_size), args.timeout, args.max_retries, args.retry_sleep)
            rows = data.get("rows", [])

        if not rows:
            break

        for row_obj in rows:
            sample = row_to_sample(row_obj, label_names)
            if sample is None:
                continue
            label = sample.label
            counts[label] = counts.get(label, 0) + 1
            # Reservoir sampling: after seeing n examples of this label, replace with probability 1/n.
            if label not in selected:
                selected[label] = sample
                pbar.n = len(selected)
                pbar.refresh()
            elif rng.randrange(counts[label]) == 0:
                selected[label] = sample

        if not args.full_reservoir_scan and len(selected) >= args.num_classes:
            break

        offset += len(rows)
        if args.limit_rows is not None and offset >= args.limit_rows:
            break

        # Keep scanning to the end for unbiased per-class reservoir sampling.
        if len(rows) < args.page_size:
            break

    pbar.close()

    missing = [i for i in range(args.num_classes) if i not in selected]
    if missing:
        raise RuntimeError(
            f"Only found {len(selected)}/{args.num_classes} classes. Missing examples: {missing[:20]}"
        )
    return [selected[i] for i in range(args.num_classes)]


def select_one_per_class_streaming(args: argparse.Namespace, label_names: list[str]) -> list[Sample]:
    """Reservoir-sample one row per class from Hugging Face streaming rows.

    `Image(decode=False)` returns image payloads as dictionaries containing raw
    bytes, which lets us keep only the selected 1,000 images in memory and avoid
    materializing the full dataset on disk.
    """
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The streaming backend requires the `datasets` package. Install it with `pip install datasets`."
        ) from exc

    token = get_token()
    load_kwargs: dict[str, Any] = {
        "split": args.split,
        "streaming": True,
    }
    if token:
        load_kwargs["token"] = token

    dataset = load_dataset(args.dataset, args.config, **load_kwargs)
    dataset = dataset.cast_column("image", HFImage(decode=False))

    rng = random.Random(args.seed)
    selected: dict[int, Sample] = {}
    counts: dict[int, int] = {}
    row_idx = -1
    pbar = tqdm(total=args.num_classes, desc="classes covered", unit="class")

    for row_idx, row in enumerate(dataset):
        if args.limit_rows is not None and row_idx >= args.limit_rows:
            break

        label = int(row["label"])
        if not 0 <= label < args.num_classes:
            continue

        image = row.get("image")
        if not isinstance(image, dict) or image.get("bytes") is None:
            raise RuntimeError(
                "Streaming row did not contain image bytes. Ensure the image column is cast with Image(decode=False)."
            )

        counts[label] = counts.get(label, 0) + 1
        label_name = label_names[label] if 0 <= label < len(label_names) else f"class_{label}"
        sample = Sample(
            row_idx=row_idx,
            label=label,
            label_name=label_name,
            image_url=str(image.get("path") or ""),
            image_bytes=image["bytes"],
        )

        # Reservoir sampling: after seeing n examples of this label, replace with probability 1/n.
        if label not in selected:
            selected[label] = sample
            pbar.n = len(selected)
            pbar.refresh()
        elif rng.randrange(counts[label]) == 0:
            selected[label] = sample

        if not args.full_reservoir_scan and len(selected) >= args.num_classes:
            break

    pbar.close()
    print(f"Scanned {row_idx + 1 if row_idx >= 0 else 0} streaming rows.")

    missing = [i for i in range(args.num_classes) if i not in selected]
    if missing:
        raise RuntimeError(
            f"Only found {len(selected)}/{args.num_classes} classes. Missing examples: {missing[:20]}"
        )
    return [selected[i] for i in range(args.num_classes)]


def save_selection(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_idx", "label", "label_name", "src_width", "src_height", "image_url"],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "row_idx": s.row_idx,
                    "label": s.label,
                    "label_name": s.label_name,
                    "src_width": s.src_width,
                    "src_height": s.src_height,
                    "image_url": s.image_url,
                }
            )


def load_selection(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append(
                Sample(
                    row_idx=int(row["row_idx"]),
                    label=int(row["label"]),
                    label_name=row["label_name"],
                    image_url=row["image_url"],
                    src_width=int(row["src_width"]) if row["src_width"] else None,
                    src_height=int(row["src_height"]) if row["src_height"] else None,
                )
            )
    return samples


def refresh_sample_urls(session: requests.Session, args: argparse.Namespace, samples: list[Sample]) -> list[Sample]:
    """Refresh expiring Dataset Server signed image URLs for cached row indices."""
    refreshed: list[Sample] = []
    by_row_idx = {s.row_idx: s for s in samples}
    page_offsets = sorted({(s.row_idx // args.page_size) * args.page_size for s in samples})
    pbar = tqdm(page_offsets, desc="refreshing signed URLs", unit="page")
    for offset in pbar:
        data = request_json(session, rows_url(args, offset, args.page_size), args.timeout, args.max_retries, args.retry_sleep)
        for row_obj in data.get("rows", []):
            row_idx = int(row_obj["row_idx"])
            old = by_row_idx.get(row_idx)
            if old is None:
                continue
            row = row_obj.get("row", {})
            image = row.get("image", {})
            image_url = image.get("src")
            if not image_url:
                raise RuntimeError(f"Missing refreshed image URL for row_idx={row_idx}")
            refreshed.append(
                Sample(
                    row_idx=row_idx,
                    label=old.label,
                    label_name=old.label_name,
                    image_url=image_url,
                    src_width=image.get("width", old.src_width),
                    src_height=image.get("height", old.src_height),
                )
            )
    if len(refreshed) != len(samples):
        missing = sorted(set(by_row_idx) - {s.row_idx for s in refreshed})
        raise RuntimeError(f"Failed to refresh {len(missing)} cached rows: {missing[:20]}")
    refreshed.sort(key=lambda s: s.label)
    return refreshed


def decode_image_rgb(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def letterbox_rgb_to_bgr_frame(image: Image.Image, width: int, height: int) -> np.ndarray:
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid image size")
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas.paste(resized, (x, y))
    rgb = np.asarray(canvas)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_video(session: requests.Session | None, args: argparse.Namespace, samples: list[Sample]) -> None:
    output = Path(args.output)
    manifest = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + 1)
    ordered = list(samples)
    rng.shuffle(ordered)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output}")

    manifest_rows: list[dict[str, Any]] = []
    try:
        for order, sample in enumerate(tqdm(ordered, desc="writing video", unit="image")):
            if sample.image_bytes is not None:
                image_bytes = sample.image_bytes
            else:
                if session is None:
                    raise RuntimeError("A requests session is required when samples do not contain image bytes.")
                image_bytes = request_bytes(session, sample.image_url, args.timeout, args.max_retries, args.retry_sleep)
            image = decode_image_rgb(image_bytes)
            if sample.src_width is None or sample.src_height is None:
                sample.src_width, sample.src_height = image.size
            frame = letterbox_rgb_to_bgr_frame(image, args.width, args.height)
            start_frame = order * args.frames_per_image
            end_frame = start_frame + args.frames_per_image - 1
            for _ in range(args.frames_per_image):
                writer.write(frame)
            manifest_rows.append(
                {
                    "order": order,
                    "row_idx": sample.row_idx,
                    "label": sample.label,
                    "label_name": sample.label_name,
                    "source_width": sample.src_width,
                    "source_height": sample.src_height,
                    "video_start_frame": start_frame,
                    "video_end_frame": end_frame,
                }
            )
    finally:
        writer.release()

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(manifest_rows)


def verify_outputs(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.output))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open generated video: {args.output}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

    expected_frames = args.num_classes * args.frames_per_image
    print("\nVerification")
    print(f"  video: {args.output}")
    print(f"  size: {width}x{height}")
    print(f"  fps: {fps}")
    print(f"  frames: {frames}")
    print(f"  duration_sec: {frames / fps if fps else 0:.3f}")
    if (width, height) != (args.width, args.height):
        raise RuntimeError(f"Unexpected size: {width}x{height}")
    if abs(fps - args.fps) > 0.05:
        raise RuntimeError(f"Unexpected fps: {fps}")
    if frames != expected_frames:
        raise RuntimeError(f"Unexpected frame count: {frames}, expected {expected_frames}")

    with Path(args.manifest).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [int(r["label"]) for r in rows]
    if len(rows) != args.num_classes or len(set(labels)) != args.num_classes:
        raise RuntimeError("Manifest does not contain exactly one row per class.")
    print(f"  manifest rows: {len(rows)}")
    print(f"  unique labels: {len(set(labels))}")


def main() -> int:
    args = parse_args()
    if args.frames_per_image <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("fps, frames-per-image, width, and height must be positive.")

    session = make_session()
    label_names = load_label_names(session, args)
    if len(label_names) < args.num_classes:
        raise RuntimeError(f"Dataset exposes only {len(label_names)} labels, expected {args.num_classes}.")

    cache_path = Path(args.metadata_cache)
    if args.backend == "streaming":
        print("Streaming dataset rows and sampling one image per class without materializing the dataset...")
        samples = select_one_per_class_streaming(args, label_names)
    else:
        if cache_path.exists() and not args.refresh_selection:
            print(f"Loading selected rows cache: {cache_path}")
            samples = load_selection(cache_path)
            if not args.keep_cached_urls:
                samples = refresh_sample_urls(session, args, samples)
        else:
            print("Scanning metadata pages and sampling one row per class without downloading images...")
            samples = select_one_per_class(session, args, label_names)
            save_selection(cache_path, samples)
            print(f"Saved selected rows cache: {cache_path}")

    if len(samples) != args.num_classes or len({s.label for s in samples}) != args.num_classes:
        raise RuntimeError("Selection must contain exactly one sample per class.")

    make_video(session, args, samples)
    verify_outputs(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)