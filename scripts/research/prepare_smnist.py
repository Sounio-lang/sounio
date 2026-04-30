#!/usr/bin/env python3
"""
Prepare Sequential MNIST data for Sounio O-SSM benchmark.

Downloads MNIST, selects a balanced subset, and writes a compact binary
file that Sounio can read via syscall6 (open/read).

Binary format:
  Header: 4 bytes magic (0x534D4E49 = "SMNI")
          4 bytes N_TRAIN (uint32 LE)
          4 bytes N_TEST (uint32 LE)
          4 bytes SEQ_LEN (uint32 LE) = 784
  Data:   For each sample:
            1 byte label (0-9)
            784 bytes pixel values (0-255), row-major, grayscale

We use 500 train + 100 test (balanced: 50+10 per digit).
Total file size: 16 + 600 * 785 = 16 + 471000 = ~460KB
"""

import struct
import gzip
import os
import urllib.request

MNIST_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DATA_DIR = "/tmp/mnist_raw"
OUTPUT = "/workspace/sounio/data/smnist_600.bin"

def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            url = MNIST_URL_BASE + f
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, path)
            print(f"  → {path}")
        else:
            print(f"  Already have {path}")

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2051
        n = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        data = f.read(n * rows * cols)
        return [data[i*784:(i+1)*784] for i in range(n)]

def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2049
        n = struct.unpack('>I', f.read(4))[0]
        return list(f.read(n))

def select_balanced(images, labels, per_class):
    """Select `per_class` samples from each digit 0-9."""
    selected = []
    counts = [0] * 10
    for img, lbl in zip(images, labels):
        if counts[lbl] < per_class:
            selected.append((lbl, img))
            counts[lbl] += 1
        if all(c >= per_class for c in counts):
            break
    return selected

def write_binary(train_data, test_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x534D4E49))  # magic "SMNI"
        f.write(struct.pack('<I', len(train_data)))
        f.write(struct.pack('<I', len(test_data)))
        f.write(struct.pack('<I', 784))
        # Train data
        for label, pixels in train_data:
            f.write(struct.pack('B', label))
            f.write(pixels)
        # Test data
        for label, pixels in test_data:
            f.write(struct.pack('B', label))
            f.write(pixels)
    print(f"Wrote {output_path}: {os.path.getsize(output_path)} bytes")
    print(f"  {len(train_data)} train + {len(test_data)} test samples")

def main():
    download_mnist()

    train_images = load_mnist_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))

    print(f"Loaded {len(train_images)} train, {len(test_images)} test images")

    # Select balanced subsets
    train_data = select_balanced(train_images, train_labels, per_class=50)  # 500 total
    test_data = select_balanced(test_images, test_labels, per_class=10)    # 100 total

    print(f"Selected {len(train_data)} train, {len(test_data)} test (balanced)")

    # Verify balance
    for split_name, data in [("Train", train_data), ("Test", test_data)]:
        counts = [0] * 10
        for lbl, _ in data:
            counts[lbl] += 1
        print(f"  {split_name}: {counts}")

    write_binary(train_data, test_data, OUTPUT)

if __name__ == "__main__":
    main()
