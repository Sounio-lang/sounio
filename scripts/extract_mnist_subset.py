#!/usr/bin/env python3
"""Extract a balanced subset of MNIST and write as a compact binary file.

Downloads MNIST if not cached locally, then extracts a balanced subset
(equal samples per digit) for use as an embedded test dataset.

Output binary format:
  [n_train: u16 LE][n_test: u16 LE]
  For each sample: [label: u8][pixels: 784 u8]

Usage:
  python3 scripts/extract_mnist_subset.py \\
    --n-train 80 --n-test 20 \\
    --out crates/souc/tests/data/mnist_100.bin
"""

import argparse
import gzip
import os
import struct
import tempfile
import urllib.request


MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url, dest_path):
    """Download a file from url to dest_path if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"  Cached: {dest_path}")
        return
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved: {dest_path}")


def read_idx_images(gz_path):
    """Parse IDX image file (gzipped) into list of 784-byte arrays."""
    with gzip.open(gz_path, "rb") as f:
        data = f.read()

    # IDX format: magic (4 bytes), n_images (4), n_rows (4), n_cols (4), then pixels
    magic = struct.unpack(">I", data[0:4])[0]
    if magic != 2051:
        raise ValueError(f"Bad magic for images: {magic} (expected 2051)")

    n_images = struct.unpack(">I", data[4:8])[0]
    n_rows = struct.unpack(">I", data[8:12])[0]
    n_cols = struct.unpack(">I", data[12:16])[0]

    if n_rows != 28 or n_cols != 28:
        raise ValueError(f"Unexpected dimensions: {n_rows}x{n_cols}")

    pixels_per_image = n_rows * n_cols  # 784
    images = []
    offset = 16
    for i in range(n_images):
        img = data[offset : offset + pixels_per_image]
        if len(img) != pixels_per_image:
            raise ValueError(f"Truncated image at index {i}")
        images.append(img)
        offset += pixels_per_image

    print(f"  Parsed {n_images} images ({n_rows}x{n_cols})")
    return images


def read_idx_labels(gz_path):
    """Parse IDX label file (gzipped) into list of ints."""
    with gzip.open(gz_path, "rb") as f:
        data = f.read()

    magic = struct.unpack(">I", data[0:4])[0]
    if magic != 2049:
        raise ValueError(f"Bad magic for labels: {magic} (expected 2049)")

    n_labels = struct.unpack(">I", data[4:8])[0]
    labels = list(data[8 : 8 + n_labels])

    if len(labels) != n_labels:
        raise ValueError(f"Truncated labels: got {len(labels)}, expected {n_labels}")

    print(f"  Parsed {n_labels} labels")
    return labels


def extract_balanced_subset(images, labels, n_total):
    """Extract a balanced subset with equal samples per digit (0-9).

    Args:
        images: list of 784-byte pixel arrays
        labels: list of int labels
        n_total: total number of samples (must be divisible by 10)

    Returns:
        (subset_images, subset_labels) sorted by label then index
    """
    if n_total % 10 != 0:
        raise ValueError(f"n_total={n_total} must be divisible by 10")

    per_digit = n_total // 10

    # Group indices by digit
    digit_indices = {d: [] for d in range(10)}
    for idx, label in enumerate(labels):
        digit_indices[label].append(idx)

    # Take first `per_digit` samples from each digit
    subset_images = []
    subset_labels = []
    for d in range(10):
        available = digit_indices[d]
        if len(available) < per_digit:
            raise ValueError(
                f"Digit {d} has only {len(available)} samples, need {per_digit}"
            )
        for i in range(per_digit):
            idx = available[i]
            subset_images.append(images[idx])
            subset_labels.append(labels[idx])

    return subset_images, subset_labels


def write_binary(out_path, train_images, train_labels, test_images, test_labels):
    """Write the binary file.

    Format:
      [n_train: u16 LE][n_test: u16 LE]
      For each train sample: [label: u8][pixels: 784 u8]
      For each test sample:  [label: u8][pixels: 784 u8]
    """
    n_train = len(train_images)
    n_test = len(test_images)

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<HH", n_train, n_test))

        # Training samples
        for img, lbl in zip(train_images, train_labels):
            f.write(struct.pack("B", lbl))
            f.write(bytes(img))

        # Test samples
        for img, lbl in zip(test_images, test_labels):
            f.write(struct.pack("B", lbl))
            f.write(bytes(img))

    total_bytes = 4 + (1 + 784) * (n_train + n_test)
    print(f"  Written: {out_path} ({total_bytes} bytes)")
    print(f"  Train: {n_train} samples, Test: {n_test} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a balanced MNIST subset as compact binary"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=80,
        help="Number of training samples (must be divisible by 10)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=20,
        help="Number of test samples (must be divisible by 10)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="crates/souc/tests/data/mnist_100.bin",
        help="Output binary file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded MNIST files (default: system temp)",
    )
    args = parser.parse_args()

    # Set up cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "mnist_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # Download MNIST files
    print("\n[1/4] Downloading MNIST files...")
    local_paths = {}
    for key, filename in MNIST_FILES.items():
        dest = os.path.join(cache_dir, filename)
        download_file(MNIST_BASE_URL + filename, dest)
        local_paths[key] = dest

    # Parse IDX files
    print("\n[2/4] Parsing IDX files...")
    train_images = read_idx_images(local_paths["train_images"])
    train_labels = read_idx_labels(local_paths["train_labels"])
    test_images = read_idx_images(local_paths["test_images"])
    test_labels = read_idx_labels(local_paths["test_labels"])

    # Extract balanced subsets
    print(f"\n[3/4] Extracting balanced subsets (train={args.n_train}, test={args.n_test})...")
    sub_train_imgs, sub_train_lbls = extract_balanced_subset(
        train_images, train_labels, args.n_train
    )
    sub_test_imgs, sub_test_lbls = extract_balanced_subset(
        test_images, test_labels, args.n_test
    )

    # Verify balance
    for split_name, lbls in [("train", sub_train_lbls), ("test", sub_test_lbls)]:
        counts = {}
        for l in lbls:
            counts[l] = counts.get(l, 0) + 1
        print(f"  {split_name} distribution: {dict(sorted(counts.items()))}")

    # Write output binary
    print(f"\n[4/4] Writing binary output...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_binary(args.out, sub_train_imgs, sub_train_lbls, sub_test_imgs, sub_test_lbls)

    print("\nDone.")


if __name__ == "__main__":
    main()
