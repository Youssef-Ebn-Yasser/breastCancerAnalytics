"""
BUSI Ultrasound Analytics & Image Quality Assessment (IQA) Pipeline.
Modality: Breast Ultrasound (B-mode Sonography with Paired Segmentation Masks)
Dataset: Breast Ultrasound Images Dataset (BUSI)

Scope:
1. Dataset Characterization & Metadata Loading (Normal vs Benign vs Malignant with Paired Lesion Masks)
2. Inherent Acoustic Challenges Analysis (Speckle noise, acoustic shadows, hypoechoic contrast)
3. Multi-Metric IQA (Laplacian, Tenengrad, Speckle Index, ENL, CNR with Masks, Shannon Entropy, SNR)
4. Statistical Analysis & Visual Quality Distributions
5. Enriched Dataset IQA Metadata Export
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

# Add project root to path for shared imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.iqa_metrics import (
    compute_laplacian_variance,
    compute_tenengrad,
    compute_rms_contrast,
    compute_michelson_contrast,
    compute_shannon_entropy,
    compute_snr_mad,
    compute_cnr,
    compute_ultrasound_speckle_metrics,
    extract_all_iqa_metrics,
)

# ---------------------------------------------------------
# 1. Dataset Discovery & Synthetic Fallback
# ---------------------------------------------------------
KAGGLE_SEARCH_PATHS = [
    "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_masks",
    "/kaggle/input/breast-ultrasound-images-dataset",
    "/kaggle/input/busi-dataset",
    "./data/busi",
    "../data/busi",
]


def find_dataset_path() -> str:
    """
    Search standard Kaggle input directories and local directories.
    """
    for p in KAGGLE_SEARCH_PATHS:
        if os.path.exists(p):
            print(f"[INFO] Found BUSI dataset at: {p}")
            return p
    return None


def generate_synthetic_ultrasound_samples(output_dir: str, num_samples: int = 60) -> str:
    """
    Generate realistic synthetic B-mode ultrasound images with paired lesion masks.
    Simulates:
    - Classes: normal, benign (smooth oval hypoechoic), malignant (irregular hypoechoic with acoustic shadow)
    - Acoustic speckle noise (Rayleigh-like multiplicative noise)
    - Posterior acoustic attenuation / shadowing
    - Paired binary lesion ground-truth masks
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = ["normal", "benign", "malignant"]

    print(f"[INFO] Generating {num_samples} simulated BUSI ultrasound images at: {output_dir}")
    np.random.seed(42)

    for i in range(num_samples):
        cls = classes[i % len(classes)]
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        h, w = 256, 256
        base = np.zeros((h, w), dtype=np.float32)
        base[0:int(h * 0.25), :] = 80.0
        base[int(h * 0.25):int(h * 0.75), :] = 120.0
        base[int(h * 0.75):, :] = 60.0

        mask = np.zeros((h, w), dtype=np.uint8)

        if cls != "normal":
            cx = np.random.randint(int(w * 0.35), int(w * 0.65))
            cy = np.random.randint(int(h * 0.35), int(h * 0.65))

            if cls == "benign":
                rx = np.random.randint(25, 40)
                ry = np.random.randint(15, 25)
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                base[mask > 0] = np.random.uniform(40, 55)
            else:
                rx = np.random.randint(20, 35)
                ry = np.random.randint(25, 45)
                cv2.ellipse(mask, (cx, cy), (rx, ry), 15, 0, 360, 255, -1)
                for _ in range(10):
                    px = cx + np.random.randint(-rx, rx)
                    py = cy + np.random.randint(-ry, ry)
                    cv2.circle(mask, (px, py), np.random.randint(6, 12), 255, -1)
                base[mask > 0] = np.random.uniform(25, 45)
                base[cy + ry:h, max(0, cx - rx):min(w, cx + rx)] *= 0.45

        speckle = np.random.gamma(shape=4.0, scale=0.25, size=(h, w))
        us_img = np.clip(base * speckle, 0, 255).astype(np.uint8)

        if i % 4 == 0:
            us_img = cv2.GaussianBlur(us_img, (5, 5), 0)

        img_name = f"{cls} ({i + 1}).png"
        mask_name = f"{cls} ({i + 1})_mask.png"

        cv2.imwrite(os.path.join(cls_dir, img_name), us_img)
        cv2.imwrite(os.path.join(cls_dir, mask_name), mask)

    return output_dir


# ---------------------------------------------------------
# 2. Metadata Loading & Characterization
# ---------------------------------------------------------
def load_busi_metadata(data_dir: str) -> pd.DataFrame:
    """
    Parse BUSI ultrasound directory pairing primary scans with lesion masks.
    """
    image_paths = (
        glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    )

    primary_images = [p for p in image_paths if "_mask" not in Path(p).name.lower()]

    records = []
    for path in primary_images:
        p = Path(path)
        fname = p.name
        stem = p.stem
        parent = p.parent.name.lower()

        cls = "normal"
        if "benign" in parent or "benign" in fname.lower():
            cls = "benign"
        elif "malignant" in parent or "malignant" in fname.lower():
            cls = "malignant"

        mask_candidate = p.parent / f"{stem}_mask{p.suffix}"
        if not mask_candidate.exists():
            mask_candidate = p.parent / f"{stem}_mask_1{p.suffix}"
        mask_path = str(mask_candidate) if mask_candidate.exists() else None

        records.append({
            "image_id": fname,
            "file_path": str(p),
            "mask_path": mask_path,
            "ground_truth_label": cls,
            "has_mask": mask_path is not None,
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} BUSI ultrasound images from {data_dir}.")
    return df


# ---------------------------------------------------------
# 3. IQA Extraction
# ---------------------------------------------------------
def extract_dataset_iqa(df: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
    """
    Extract IQA metrics tailored for ultrasound, including speckle index, ENL, and CNR.
    """
    sample_df = df.head(max_samples).copy()
    print(f"[INFO] Extracting IQA metrics across {len(sample_df)} ultrasound scans...")

    iqa_list = []
    for idx, row in sample_df.iterrows():
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = None
        if row["mask_path"] and os.path.exists(row["mask_path"]):
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        metrics = extract_all_iqa_metrics(img, mask=mask, modality="ultrasound")
        metrics["image_id"] = row["image_id"]
        iqa_list.append(metrics)

    iqa_df = pd.DataFrame(iqa_list)
    merged_df = pd.merge(sample_df, iqa_df, on="image_id")
    return merged_df


# ---------------------------------------------------------
# 4. Statistical Analysis & Visualizations
# ---------------------------------------------------------
def generate_quality_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generate diagnostic plots for BUSI ultrasound data.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Class Distribution
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="ground_truth_label", hue="ground_truth_label", palette="mako", legend=False)
    plt.title("BUSI Ultrasound: Diagnostic Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Sonographic Category")
    plt.ylabel("Scan Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=200)
    plt.close()

    # 2. Quality Metric Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df["laplacian_variance"], kde=True, ax=axes[0, 0], color="purple")
    axes[0, 0].set_title("Sharpness: Laplacian Variance", fontweight="bold")

    sns.histplot(df["speckle_index"], kde=True, ax=axes[0, 1], color="teal")
    axes[0, 1].set_title("Acoustic Speckle Index (Cs = Std/Mean)", fontweight="bold")

    sns.histplot(df["shannon_entropy"], kde=True, ax=axes[1, 0], color="darkblue")
    axes[1, 0].set_title("Shannon Information Entropy (bits)", fontweight="bold")

    sns.histplot(df["snr_db"], kde=True, ax=axes[1, 1], color="forestgreen")
    axes[1, 1].set_title("Signal-to-Noise Ratio (dB)", fontweight="bold")

    plt.suptitle("BUSI Ultrasound: Quality Metric Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_quality_metrics_distribution.png"), dpi=200)
    plt.close()

    # 3. CNR Comparison (Benign vs Malignant)
    has_cnr = df[df["cnr"] > 0]
    if len(has_cnr) > 5:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=has_cnr, x="ground_truth_label", y="cnr", hue="ground_truth_label", palette="Set2", legend=False)
        plt.title("Contrast-to-Noise Ratio (CNR) by Tumor Classification", fontsize=13, fontweight="bold")
        plt.ylabel("Lesion-to-Background CNR")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_cnr_by_class.png"), dpi=200)
        plt.close()

    # 4. Correlation Heatmap
    metric_cols = [
        "laplacian_variance", "speckle_index", "enl", "rms_contrast",
        "shannon_entropy", "snr_db", "composite_quality_score"
    ]
    plt.figure(figsize=(8, 6))
    corr = df[metric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", cbar=True)
    plt.title("Correlation Matrix of Ultrasound IQA & Speckle Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_metric_correlation_matrix.png"), dpi=200)
    plt.close()

    # 5. Visual Montage: High Quality vs Acoustic Dropout
    sorted_df = df.sort_values(by="composite_quality_score", ascending=False)
    best_samples = sorted_df.head(3)
    worst_samples = sorted_df.tail(3)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (_, row) in enumerate(best_samples.iterrows()):
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(
            f"HIGH QUALITY\nScore: {row['composite_quality_score']:.2f} | Speckle: {row['speckle_index']:.2f}",
            color="green", fontsize=10, fontweight="bold"
        )
        axes[0, i].axis("off")

    for i, (_, row) in enumerate(worst_samples.iterrows()):
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(
            f"DEGRADED / DROPOUT\nScore: {row['composite_quality_score']:.2f} | Speckle: {row['speckle_index']:.2f}",
            color="red", fontsize=10, fontweight="bold"
        )
        axes[1, i].axis("off")

    plt.suptitle("BUSI: Visual Comparison of Ultrasound Scan Quality", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_quality_comparison_montage.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved all diagnostic visualization figures to: {output_dir}")


# ---------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------
def main():
    print("Starting BUSI Ultrasound Analytics Pipeline...")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = find_dataset_path()
    if data_dir is None:
        synth_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        data_dir = generate_synthetic_ultrasound_samples(synth_dir, num_samples=60)

    df = load_busi_metadata(data_dir)
    df_iqa = extract_dataset_iqa(df, max_samples=200)

    # Statistical Summary
    stats = df_iqa[[
        "laplacian_variance", "tenengrad_gradient", "speckle_index", "enl",
        "rms_contrast", "shannon_entropy", "snr_db", "composite_quality_score"
    ]].describe()
    print("\n" + "=" * 60)
    print("BUSI IMAGE QUALITY ASSESSMENT: STATISTICAL SUMMARY")
    print("=" * 60)
    print(stats)
    stats.to_csv(os.path.join(output_dir, "busi_iqa_statistical_summary.csv"))

    generate_quality_visualizations(df_iqa, output_dir)

    csv_out = os.path.join(output_dir, "busi_quality_metadata.csv")
    df_iqa.to_csv(csv_out, index=False)
    print(f"[INFO] Exported enriched clinical IQA metadata to: {csv_out}")
    print("\nBUSI Analysis Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
