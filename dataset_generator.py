"""
Dataset Generator for Classical Image Encryption ML
Generates a diverse, balanced benchmark dataset across:
  - Class 0: Plaintext (diverse synthetic and photographic patterns)
  - Class 1: Weak Cipher (Arnold Cat Map spatial scrambling only)
  - Class 2: Strong Chaos (Arnold Cat Map + Logistic Map chaotic permutation)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

from arnold_cat_map import arnold_cat_map, logistic_map_scramble
from feature_extractor import extract_crypto_features, FEATURE_NAMES

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)


def generate_synthetic_plain_image(size=64, pattern_type="gradient"):
    """
    Generate a diverse plain image pattern of shape (size, size, 3) in uint8 [0, 255].
    """
    arr = np.zeros((size, size, 3), dtype=np.float64)
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))

    if pattern_type == "linear_gradient":
        angle = np.random.uniform(0, 2 * np.pi)
        proj = x * np.cos(angle) + y * np.sin(angle)
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        c_shift = np.random.uniform(0.5, 1.5, 3)
        for c in range(3):
            arr[:, :, c] = np.clip(proj * 255 * c_shift[c], 0, 255)

    elif pattern_type == "radial_gradient":
        cx, cy = np.random.uniform(0.3, 0.7, 2)
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r = r / (r.max() + 1e-8)
        freq = np.random.uniform(1.0, 4.0)
        base = 0.5 + 0.5 * np.cos(2 * np.pi * freq * r)
        for c in range(3):
            arr[:, :, c] = np.clip(base * 255 * np.random.uniform(0.7, 1.0), 0, 255)

    elif pattern_type == "checkerboard":
        freq = np.random.choice([4, 8, 16])
        grid = ((x * freq).astype(int) + (y * freq).astype(int)) % 2
        col1 = np.random.randint(20, 100, 3)
        col2 = np.random.randint(150, 240, 3)
        for c in range(3):
            arr[:, :, c] = np.where(grid == 0, col1[c], col2[c])

    elif pattern_type == "sinusoidal_waves":
        f1 = np.random.uniform(2, 8)
        f2 = np.random.uniform(2, 8)
        phase = np.random.uniform(0, 2 * np.pi)
        wave = 0.5 + 0.5 * np.sin(2 * np.pi * f1 * x + phase) * np.cos(2 * np.pi * f2 * y)
        for c in range(3):
            arr[:, :, c] = np.clip(wave * 255 * np.random.uniform(0.6, 1.0), 0, 255)

    elif pattern_type == "geometric_shapes":
        base_color = np.random.randint(200, 255, 3)
        arr[:] = base_color
        # Add random rectangles and circles
        for _ in range(np.random.randint(2, 6)):
            x0, y0 = np.random.randint(5, size - 20, 2)
            w0, h0 = np.random.randint(10, 25, 2)
            color = np.random.randint(0, 180, 3)
            arr[y0:min(y0 + h0, size), x0:min(x0 + w0, size)] = color

    elif pattern_type == "coherent_noise":
        # Smooth smoothed Gaussian pseudo-noise
        raw = np.random.randn(size, size, 3)
        # 2D Gaussian filter kernel
        kernel_size = 5
        k_1d = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2)
        k_2d = np.outer(k_1d, k_1d)
        k_2d /= k_2d.sum()
        for c in range(3):
            smoothed = np.pad(raw[:, :, c], kernel_size // 2, mode="reflect")
            # Convolve
            filtered = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    filtered[i, j] = np.sum(smoothed[i:i + kernel_size, j:j + kernel_size] * k_2d)
            f_norm = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8)
            arr[:, :, c] = f_norm * 255

    else:
        # Default smooth ramp
        arr = (x[:, :, None] * 0.5 + y[:, :, None] * 0.5) * 255

    # Add slight natural sensor noise
    noise = np.random.normal(0, 2.0, (size, size, 3))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr


def load_real_image_patches(size=64, num_patches=100):
    """
    Extract realistic photographic patches from existing project images.
    """
    patches = []
    real_paths = [
        "decrypted_image_gui.png",
        "test_decrypted_user_img.png",
        "static/sample.png",
    ]
    for p in real_paths:
        if os.path.exists(p):
            try:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                if w >= size and h >= size:
                    for _ in range(num_patches // len(real_paths) + 10):
                        rx = np.random.randint(0, w - size + 1)
                        ry = np.random.randint(0, h - size + 1)
                        crop = np.array(img.crop((rx, ry, rx + size, ry + size)))
                        patches.append(crop)
            except Exception as e:
                print(f"Notice: skipped {p}: {e}")

    return patches


def build_cryptographic_dataset(n_samples_per_class=400, img_size=64, random_seed=42):
    """
    Constructs a balanced 3-class dataset:
      - 0: Plaintext
      - 1: Weak_Cipher (Arnold Only)
      - 2: Strong_Chaos (Arnold + Logistic)
    """
    np.random.seed(random_seed)
    print(f"Generating balanced dataset: {n_samples_per_class} samples per class ({n_samples_per_class * 3} total)...")

    # 1. Collect base plain images
    pattern_types = [
        "linear_gradient",
        "radial_gradient",
        "checkerboard",
        "sinusoidal_waves",
        "geometric_shapes",
        "coherent_noise",
    ]

    real_patches = load_real_image_patches(size=img_size, num_patches=150)
    print(f"Extracted {len(real_patches)} real photographic patches.")

    base_plain_images = []
    # Add real photographic crops first
    for patch in real_patches[: n_samples_per_class // 3]:
        base_plain_images.append(patch)

    # Fill remainder with rich synthetic patterns
    while len(base_plain_images) < n_samples_per_class:
        ptype = np.random.choice(pattern_types)
        img = generate_synthetic_plain_image(size=img_size, pattern_type=ptype)
        base_plain_images.append(img)

    base_plain_images = base_plain_images[:n_samples_per_class]

    # 2. Generate the 3 classes and extract cryptographic features
    rows = []
    image_arrays = []

    for i, plain_img in enumerate(base_plain_images):
        # Class 0: Plaintext
        feat_0 = extract_crypto_features(plain_img)
        feat_0["sample_id"] = f"sample_{i:04d}_plain"
        feat_0["class_label"] = 0
        feat_0["class_name"] = "Plaintext"
        rows.append(feat_0)
        image_arrays.append(plain_img)

        # Class 1: Weak Cipher (Arnold Cat Map Only)
        # Periodic and preserves histogram exactly
        k_iter = np.random.choice([1, 2, 3])
        weak_img = arnold_cat_map(plain_img, iterations=k_iter)
        feat_1 = extract_crypto_features(weak_img)
        feat_1["sample_id"] = f"sample_{i:04d}_weak"
        feat_1["class_label"] = 1
        feat_1["class_name"] = "Weak_Cipher"
        rows.append(feat_1)
        image_arrays.append(weak_img)

        # Class 2: Strong Chaos (Arnold + Logistic Map)
        # Maximum Shannon diffusion and near-zero correlation
        x_seed = np.random.uniform(0.15, 0.85)
        r_param = np.random.uniform(3.75, 4.0)
        strong_img = logistic_map_scramble(weak_img, x=x_seed, r=r_param)
        feat_2 = extract_crypto_features(strong_img)
        feat_2["sample_id"] = f"sample_{i:04d}_strong"
        feat_2["class_label"] = 2
        feat_2["class_name"] = "Strong_Chaos"
        rows.append(feat_2)
        image_arrays.append(strong_img)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATASET_DIR, "crypto_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"✔ Dataset features saved to {csv_path} (Shape: {df.shape})")

    # Also save compact NPZ for raw image reference
    npz_path = os.path.join(DATASET_DIR, "crypto_images.npz")
    np.savez_compressed(
        npz_path,
        images=np.array(image_arrays, dtype=np.uint8),
        labels=df["class_label"].values,
    )
    print(f"✔ Dataset image matrices saved to {npz_path}")

    # Display dataset balance and summary
    print("\nDataset Class Distribution:")
    print(df["class_name"].value_counts())
    print("\nFeature Summary across classes (Mean values):")
    summary_cols = ["entropy", "corr_h", "grad_h", "chi2_uniformity"]
    print(df.groupby("class_name")[summary_cols].mean())

    return df


if __name__ == "__main__":
    build_cryptographic_dataset(n_samples_per_class=400, img_size=64)
