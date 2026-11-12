"""
Cryptographic Feature Extractor for Classical & Chaotic Image Encryption
Extracts rigorous statistical, spatial correlation, and information-theoretic features
for machine learning classification and security auditing.
"""

import numpy as np


def extract_crypto_features(img_matrix):
    """
    Extract a dictionary of numerical features characterizing cryptographic diffusion,
    confusion, and spatial decorrelation from an image numpy array (H, W, 3) or (H, W).
    """
    if img_matrix.ndim == 3:
        # Standard luminance formula (Rec. 601)
        gray = 0.2989 * img_matrix[:, :, 0] + 0.5870 * img_matrix[:, :, 1] + 0.1140 * img_matrix[:, :, 2]
    else:
        gray = img_matrix.astype(np.float64)

    H, W = gray.shape
    flat = gray.flatten()
    N_pixels = len(flat)

    # 1. Global Shannon Information Entropy
    # Max possible = 8.0 bits/symbol for 256 gray levels
    int_flat = np.clip(np.round(flat).astype(np.int32), 0, 255)
    counts = np.bincount(int_flat, minlength=256)
    probs = counts[counts > 0] / float(N_pixels)
    entropy = float(-np.sum(probs * np.log2(probs)))

    # 2. Chi-Square Uniformity Test
    # Distance from ideal uniform distribution (E = N/256 for all bins)
    expected_bin_count = N_pixels / 256.0
    chi2_uniformity = float(np.sum((counts - expected_bin_count) ** 2 / expected_bin_count))

    # 3. Energy / Uniformity (sum of squared probabilities)
    # Higher for non-uniform plain images, lower (1/256 = 0.0039) for uniform noise
    energy = float(np.sum(probs ** 2))

    # 4. Adjacent Pixel Correlations (Horizontal, Vertical, Diagonal)
    def calc_correlation(a, b):
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a * std_b == 0:
            return 0.0
        cov = np.cov(a, b)[0, 1]
        return float(np.clip(cov / (std_a * std_b), -1.0, 1.0))

    # Horizontal pairs: (i, j) and (i, j+1)
    x_h = gray[:, :-1].flatten()
    y_h = gray[:, 1:].flatten()
    corr_h = calc_correlation(x_h, y_h)

    # Vertical pairs: (i, j) and (i+1, j)
    x_v = gray[:-1, :].flatten()
    y_v = gray[1:, :].flatten()
    corr_v = calc_correlation(x_v, y_v)

    # Diagonal pairs: (i, j) and (i+1, j+1)
    x_d = gray[:-1, :-1].flatten()
    y_d = gray[1:, 1:].flatten()
    corr_d = calc_correlation(x_d, y_d)

    # 5. Mean Absolute Differences (Gradient magnitudes)
    grad_h = float(np.mean(np.abs(gray[:, :-1] - gray[:, 1:])))
    grad_v = float(np.mean(np.abs(gray[:-1, :] - gray[1:, :])))

    # 6. Local Block Entropy (8x8 blocks) to evaluate local diffusion homogeneity
    block_h, block_w = 8, 8
    block_entropies = []
    if H >= block_h and W >= block_w:
        for r in range(0, H - block_h + 1, block_h):
            for c in range(0, W - block_w + 1, block_w):
                b_flat = np.clip(np.round(gray[r:r + block_h, c:c + block_w]).astype(np.int32), 0, 255).flatten()
                b_counts = np.bincount(b_flat, minlength=256)
                b_probs = b_counts[b_counts > 0] / float(len(b_flat))
                b_ent = -np.sum(b_probs * np.log2(b_probs))
                block_entropies.append(b_ent)

    if block_entropies:
        local_entropy_mean = float(np.mean(block_entropies))
        local_entropy_std = float(np.std(block_entropies))
    else:
        local_entropy_mean = entropy
        local_entropy_std = 0.0

    # 7. Global Pixel Intensity Standard Deviation
    pixel_std = float(np.std(gray))

    # 8. Inter-Channel Cross-Correlation (for color images)
    if img_matrix.ndim == 3 and img_matrix.shape[2] >= 3:
        r_flat = img_matrix[:, :, 0].flatten().astype(np.float64)
        g_flat = img_matrix[:, :, 1].flatten().astype(np.float64)
        b_flat = img_matrix[:, :, 2].flatten().astype(np.float64)
        corr_rg = calc_correlation(r_flat, g_flat)
        corr_rb = calc_correlation(r_flat, b_flat)
        interchannel_corr = float((abs(corr_rg) + abs(corr_rb)) / 2.0)
    else:
        interchannel_corr = 0.0

    return {
        "entropy": entropy,
        "chi2_uniformity": chi2_uniformity,
        "energy": energy,
        "corr_h": corr_h,
        "corr_v": corr_v,
        "corr_d": corr_d,
        "grad_h": grad_h,
        "grad_v": grad_v,
        "local_entropy_mean": local_entropy_mean,
        "local_entropy_std": local_entropy_std,
        "pixel_std": pixel_std,
        "interchannel_corr": interchannel_corr,
    }


FEATURE_NAMES = [
    "entropy",
    "chi2_uniformity",
    "energy",
    "corr_h",
    "corr_v",
    "corr_d",
    "grad_h",
    "grad_v",
    "local_entropy_mean",
    "local_entropy_std",
    "pixel_std",
    "interchannel_corr",
]

CLASS_LABELS = {
    0: "Plaintext",
    1: "Weak_Cipher (Arnold Only)",
    2: "Strong_Chaos (Arnold + Logistic)",
}

CLASS_SEVERITY = {
    0: {"status": "Unencrypted", "risk": "High Vulnerability", "color": "#f43f5e"},
    1: {"status": "Weak Cipher", "risk": "Medium Vulnerability (Predictable Periodic Orbit)", "color": "#f59e0b"},
    2: {"status": "Cryptographically Secure", "risk": "Low Vulnerability (High Diffusion & Decorrelation)", "color": "#10b981"},
}
