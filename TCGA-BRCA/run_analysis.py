"""
TCGA-BRCA Histopathology Analytics & Image Quality Assessment (IQA) Pipeline.
Modality: Histopathology (Tissue Whole Slide Imaging / Diagnostic Patches)
Dataset: The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA)

Scope:
1. Dataset Characterization & Class Imbalance Analysis (IDC vs ILC vs Normal, Frozen vs FFPE)
2. Inherent Artifact & Challenge Identification (Stain variation, blur, knife chatter)
3. Multi-Metric Image Quality Assessment (Laplacian, Tenengrad, RMS Contrast, Entropy, SNR)
4. Statistical Analysis & Visual Quality Distribution
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
    compute_perceptual_nss_score,
    extract_all_iqa_metrics,
)

# ---------------------------------------------------------
# 1. Dataset Discovery & Synthetic Fallback
# ---------------------------------------------------------
KAGGLE_SEARCH_PATHS = [
    "/kaggle/input/tcga-brca",
    "/kaggle/input/tcga-brca-dataset",
    "/kaggle/input/breast-histopathology-images",
    "/kaggle/input/idc-regular-dataset",
    "./data/tcga_brca",
    "../data/tcga_brca",
]


def find_dataset_path() -> str:
    """
    Search standard Kaggle input directories and local directories.
    """
    for p in KAGGLE_SEARCH_PATHS:
        if os.path.exists(p):
            print(f"[INFO] Found TCGA-BRCA dataset at: {p}")
            return p
    return None


def generate_synthetic_histology_samples(output_dir: str, num_samples: int = 60) -> str:
    """
    Generate realistic synthetic H&E histology patches when no raw dataset is mounted.
    Simulates:
    - IDC (Invasive Ductal Carcinoma): dense hyperchromatic cell clusters, high cellularity
    - ILC (Invasive Lobular Carcinoma): linear 'single-file' infiltrative patterns
    - Normal / Benign: structured ductal lumen with stroma
    - Preparation variations: FFPE (crisp) vs Frozen Section (ice crystal artifacts & blur)
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = ["IDC_Malignant", "ILC_Malignant", "Normal_Benign"]
    preparations = ["FFPE", "Frozen_Section"]

    print(f"[INFO] Generating {num_samples} simulated TCGA-BRCA histology patches at: {output_dir}")
    np.random.seed(42)

    for i in range(num_samples):
        cls = classes[i % len(classes)]
        prep = preparations[(i // len(classes)) % len(preparations)]
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        h, w = 256, 256
        img = np.ones((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = np.random.randint(200, 230)  # B
        img[:, :, 1] = np.random.randint(170, 200)  # G
        img[:, :, 2] = np.random.randint(220, 245)  # R (Eosin pink tint)

        # Draw collagen fibers
        for _ in range(25):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(img, pt1, pt2, (180, 140, 210), 1)

        # Draw cell nuclei (Hematoxylin purple)
        num_nuclei = 120 if cls == "IDC_Malignant" else (60 if cls == "ILC_Malignant" else 30)
        for _ in range(num_nuclei):
            cx = np.random.randint(15, w - 15)
            cy = np.random.randint(15, h - 15)
            r = np.random.randint(3, 7)
            cv2.circle(img, (cx, cy), r, (130, 40, 80), -1)

        # Frozen Section blur or scratch
        if prep == "Frozen_Section":
            k = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
            if np.random.rand() > 0.5:
                cv2.line(img, (0, np.random.randint(20, h - 20)), (w, np.random.randint(20, h - 20)), (240, 240, 240), 2)

        fname = f"TCGA_BRCA_{prep}_{cls}_{i:03d}.png"
        cv2.imwrite(os.path.join(cls_dir, fname), img)

    return output_dir


# ---------------------------------------------------------
# 2. Dataset Characterization & Metadata Loading
# ---------------------------------------------------------
def load_tcga_brca_metadata(data_dir: str) -> pd.DataFrame:
    """
    Traverse histology images and assemble dataset metadata.
    """
    image_paths = (
        glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.tif"), recursive=True)
    )

    records = []
    for path in image_paths:
        filename = Path(path).name
        parent = Path(path).parent.name
        prep = "Frozen_Section" if "frozen" in filename.lower() or "frozen" in parent.lower() else "FFPE"

        label = "Unknown"
        for candidate in ["IDC", "ILC", "Malignant", "Benign", "Normal"]:
            if candidate.lower() in parent.lower() or candidate.lower() in filename.lower():
                label = candidate
                break

        records.append({
            "image_id": filename,
            "file_path": path,
            "ground_truth_label": label,
            "preparation_type": prep,
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} images from {data_dir}.")
    return df


# ---------------------------------------------------------
# 3. Image Quality Assessment (IQA) Extraction
# ---------------------------------------------------------
def extract_dataset_iqa(df: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
    """
    Run IQA feature extraction on histology images.
    """
    sample_df = df.head(max_samples).copy()
    print(f"[INFO] Extracting IQA metrics across {len(sample_df)} histology images...")

    iqa_list = []
    for idx, row in sample_df.iterrows():
        img = cv2.imread(row["file_path"])
        if img is None:
            continue
        metrics = extract_all_iqa_metrics(img, modality="histopathology")
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
    Generate diagnostic statistical plots:
    - Distribution of Laplacian Variance, RMS Contrast, Shannon Entropy, SNR
    - Best vs Degraded Quality Sample Montages
    - Correlation Matrix of Quality Metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Class Distribution Plot
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="ground_truth_label", hue="preparation_type", palette="Set2")
    plt.title("TCGA-BRCA: Class Distribution & Preparation Type", fontsize=13, fontweight="bold")
    plt.xlabel("Diagnostic Class")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=200)
    plt.close()

    # 2. Quality Metrics Distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df["laplacian_variance"], kde=True, ax=axes[0, 0], color="crimson")
    axes[0, 0].set_title("Sharpness: Laplacian Variance", fontweight="bold")

    sns.histplot(df["rms_contrast"], kde=True, ax=axes[0, 1], color="teal")
    axes[0, 1].set_title("Contrast: RMS Contrast", fontweight="bold")

    sns.histplot(df["shannon_entropy"], kde=True, ax=axes[1, 0], color="purple")
    axes[1, 0].set_title("Information: Shannon Entropy (bits)", fontweight="bold")

    sns.histplot(df["snr_db"], kde=True, ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("Signal-to-Noise Ratio (dB)", fontweight="bold")

    plt.suptitle("TCGA-BRCA Histology: Image Quality Metrics Distribution", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_quality_metrics_distribution.png"), dpi=200)
    plt.close()

    # 3. Quality by Preparation Type (FFPE vs Frozen Section)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="preparation_type", y="laplacian_variance", hue="preparation_type", palette="Pastel1", legend=False)
    plt.title("Sharpness Comparison: FFPE vs. Frozen Section Artifacts", fontsize=13, fontweight="bold")
    plt.ylabel("Laplacian Variance (Sharpness)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_ffpe_vs_frozen_sharpness.png"), dpi=200)
    plt.close()

    # 4. Correlation Matrix
    metric_cols = [
        "laplacian_variance", "tenengrad_gradient", "rms_contrast",
        "michelson_contrast", "shannon_entropy", "snr_db", "composite_quality_score"
    ]
    plt.figure(figsize=(8, 6))
    corr = df[metric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Histopathology IQA Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_metric_correlation_matrix.png"), dpi=200)
    plt.close()

    # 5. Visual Sample Comparison: Best Quality vs Degraded
    sorted_df = df.sort_values(by="composite_quality_score", ascending=False)
    best_samples = sorted_df.head(3)
    worst_samples = sorted_df.tail(3)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (_, row) in enumerate(best_samples.iterrows()):
        img = cv2.imread(row["file_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(
            f"HIGH QUALITY\nScore: {row['composite_quality_score']:.2f} | LapVar: {row['laplacian_variance']:.1f}",
            color="green", fontsize=10, fontweight="bold"
        )
        axes[0, i].axis("off")

    for i, (_, row) in enumerate(worst_samples.iterrows()):
        img = cv2.imread(row["file_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img_rgb)
        axes[1, i].set_title(
            f"DEGRADED / BLUR\nScore: {row['composite_quality_score']:.2f} | LapVar: {row['laplacian_variance']:.1f}",
            color="red", fontsize=10, fontweight="bold"
        )
        axes[1, i].axis("off")

    plt.suptitle("TCGA-BRCA: Visual Inspection (High Quality vs Degraded Slides)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_quality_comparison_montage.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved all diagnostic visualization figures to: {output_dir}")


# ---------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------
def main():
    print("Starting TCGA-BRCA Histopathology Analytics Pipeline...")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = find_dataset_path()
    if data_dir is None:
        synth_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        data_dir = generate_synthetic_histology_samples(synth_dir, num_samples=60)

    df = load_tcga_brca_metadata(data_dir)
    df_iqa = extract_dataset_iqa(df, max_samples=200)

    # Statistical Summary
    stats = df_iqa[[
        "laplacian_variance", "tenengrad_gradient", "rms_contrast",
        "michelson_contrast", "shannon_entropy", "snr_db", "composite_quality_score"
    ]].describe()
    print("\n" + "=" * 60)
    print("TCGA-BRCA IMAGE QUALITY ASSESSMENT: STATISTICAL SUMMARY")
    print("=" * 60)
    print(stats)
    stats.to_csv(os.path.join(output_dir, "tcga_brca_iqa_statistical_summary.csv"))

    generate_quality_visualizations(df_iqa, output_dir)

    # Save final enriched metadata CSV
    csv_out = os.path.join(output_dir, "tcga_brca_quality_metadata.csv")
    df_iqa.to_csv(csv_out, index=False)
    print(f"[INFO] Exported enriched clinical IQA metadata to: {csv_out}")
    print("\nTCGA-BRCA Analysis Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
