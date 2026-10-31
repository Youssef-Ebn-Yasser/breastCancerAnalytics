"""
CBIS-DDSM Mammography Analytics & Image Quality Assessment (IQA) Pipeline.
Modality: Mammography (Screening Projection Radiography: CC & MLO Views)
Dataset: Curated Breast Imaging Subset of DDSM (CBIS-DDSM)

Features:
1. Dataset Characterization & Metadata Loading (Masses vs Calcifications, Benign vs Malignant, BI-RADS)
2. Inherent Radiographic Challenges Analysis (Film grain, dense tissue masking, scanner artifacts)
3. Multi-Metric Image Quality Assessment (Laplacian, Tenengrad, RMS, Michelson, Entropy, SNR)
4. Statistical Analysis & Visual Quality Distributions
5. Learned Decision Tree for Quality-Aware Adaptive Clinical Routing
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
    extract_all_iqa_metrics,
)
from common.adaptive_decision_tree import (
    AdaptiveClinicalDecisionTree,
    generate_ground_truth_routing_policy,
)

# ---------------------------------------------------------
# 1. Dataset Discovery & Synthetic Fallback
# ---------------------------------------------------------
KAGGLE_SEARCH_PATHS = [
    "/kaggle/input/cbis-ddsm-breast-cancer-image-dataset",
    "/kaggle/input/cbis-ddsm",
    "/kaggle/input/ddsm-mammography",
    "./data/cbis_ddsm",
    "../data/cbis_ddsm",
]


def find_dataset_path() -> str:
    """
    Search standard Kaggle input directories and local directories.
    """
    for p in KAGGLE_SEARCH_PATHS:
        if os.path.exists(p):
            print(f"[INFO] Found CBIS-DDSM dataset at: {p}")
            return p
    return None


def generate_synthetic_mammography_samples(output_dir: str, num_samples: int = 60) -> str:
    """
    Generate realistic synthetic mammogram samples when no raw dataset is mounted.
    Simulates:
    - Projection views: CC (Craniocaudal) and MLO (Mediolateral Oblique)
    - Abnormality types: Masses (spiculated/circumscribed) and Microcalcifications
    - Pathology ground truth: BENIGN vs MALIGNANT
    - Radiographic artifacts: film grain noise, low contrast in dense breasts, scanner scratches
    """
    os.makedirs(output_dir, exist_ok=True)
    views = ["CC", "MLO"]
    pathologies = ["BENIGN", "MALIGNANT"]
    abnormalities = ["Mass", "Calcification"]

    print(f"[INFO] Generating {num_samples} simulated CBIS-DDSM mammograms at: {output_dir}")
    np.random.seed(42)

    for i in range(num_samples):
        view = views[i % 2]
        pathology = pathologies[(i // 2) % 2]
        abnormality = abnormalities[(i // 4) % 2]
        subfolder = os.path.join(output_dir, f"{pathology}_{abnormality}")
        os.makedirs(subfolder, exist_ok=True)

        h, w = 384, 256
        img = np.zeros((h, w), dtype=np.uint8)

        # Draw anatomical breast profile (arc extending from left edge)
        center = (0, int(h * 0.5))
        axes = (int(w * 0.90), int(h * 0.45))
        cv2.ellipse(img, center, axes, 0, -90, 90, 85, -1)

        # Add pectoral muscle on MLO view (triangular high density in upper chest)
        if view == "MLO":
            pts = np.array([[0, 0], [int(w * 0.45), 0], [0, int(h * 0.40)]], np.int32)
            cv2.fillPoly(img, [pts], 160)

        # Add fibroglandular parenchyma (swirling clouds of glandular tissue)
        dense_parenchyma = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(dense_parenchyma, center, (int(w * 0.65), int(h * 0.35)), 0, -90, 90, 50, -1)
        dense_parenchyma = cv2.GaussianBlur(dense_parenchyma, (35, 35), 0)
        img = cv2.add(img, dense_parenchyma)

        # Inject abnormality lesion
        cx = np.random.randint(int(w * 0.25), int(w * 0.70))
        cy = np.random.randint(int(h * 0.30), int(h * 0.70))

        if abnormality == "Mass":
            radius = np.random.randint(12, 22)
            # Malignant masses exhibit irregular/spiculated borders
            if pathology == "MALIGNANT":
                cv2.circle(img, (cx, cy), radius, 220, -1)
                for _ in range(8):
                    ang = np.random.uniform(0, 2 * np.pi)
                    spk_x = int(cx + (radius + 10) * np.cos(ang))
                    spk_y = int(cy + (radius + 10) * np.sin(ang))
                    cv2.line(img, (cx, cy), (spk_x, spk_y), 210, 2)
            else:
                # Benign circumscribed oval mass
                cv2.ellipse(img, (cx, cy), (radius, int(radius * 0.7)), 30, 0, 360, 190, -1)
        else:
            # Microcalcification clusters (fine bright specks)
            n_specks = 15 if pathology == "MALIGNANT" else 6
            for _ in range(n_specks):
                sx = cx + np.random.randint(-12, 12)
                sy = cy + np.random.randint(-12, 12)
                cv2.circle(img, (sx, sy), 1, 250, -1)

        # Add digitized film noise and scanner artifact variations
        noise = np.random.normal(0, np.random.uniform(3, 14), (h, w)).astype(np.float32)
        noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Mild blur or scanner scratch on subset
        if i % 5 == 0:
            noisy_img = cv2.GaussianBlur(noisy_img, (5, 5), 0)
        if i % 7 == 0:
            cv2.line(noisy_img, (np.random.randint(0, w), 0), (np.random.randint(0, w), h), 235, 1)

        fname = f"CBIS_DDSM_{view}_{pathology}_{abnormality}_{i:03d}.png"
        cv2.imwrite(os.path.join(subfolder, fname), noisy_img)

    return output_dir


# ---------------------------------------------------------
# 2. Metadata Loading & Characterization
# ---------------------------------------------------------
def load_cbis_ddsm_metadata(data_dir: str) -> pd.DataFrame:
    """
    Traverse mammograms and parse clinical labels.
    """
    image_paths = (
        glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.dcm"), recursive=True)
    )

    records = []
    for path in image_paths:
        fname = Path(path).name
        parent = Path(path).parent.name
        
        view = "MLO" if "mlo" in fname.lower() or "mlo" in parent.lower() else "CC"
        pathology = "MALIGNANT" if ("malignant" in fname.lower() or "malignant" in parent.lower()) else "BENIGN"
        abnormality = "Calcification" if ("calc" in fname.lower() or "calc" in parent.lower()) else "Mass"

        records.append({
            "image_id": fname,
            "file_path": path,
            "ground_truth_label": pathology,
            "abnormality_type": abnormality,
            "mammographic_view": view,
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} mammograms from {data_dir}.")
    return df


# ---------------------------------------------------------
# 3. IQA Extraction
# ---------------------------------------------------------
def extract_dataset_iqa(df: pd.DataFrame, max_samples: int = 500) -> pd.DataFrame:
    """
    Compute IQA metrics tailored for mammography.
    """
    sample_df = df.head(max_samples).copy()
    print(f"[INFO] Extracting IQA metrics across {len(sample_df)} mammograms...")

    iqa_list = []
    for idx, row in sample_df.iterrows():
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        metrics = extract_all_iqa_metrics(img, modality="mammography")
        metrics["image_id"] = row["image_id"]
        iqa_list.append(metrics)

    iqa_df = pd.DataFrame(iqa_list)
    merged_df = pd.merge(sample_df, iqa_df, on="image_id")

    # Simulate realistic CNN inference confidence for mammograms
    np.random.seed(42)
    base_conf = 0.58 + 0.32 * merged_df["composite_quality_score"]
    noise = np.random.normal(0, 0.05, len(merged_df))
    merged_df["cnn_confidence"] = np.clip(base_conf + noise, 0.45, 0.99)

    return merged_df


# ---------------------------------------------------------
# 4. Statistical Analysis & Visualizations
# ---------------------------------------------------------
def generate_quality_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generate comprehensive diagnostic plots for CBIS-DDSM mammography.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Class & Abnormality Distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="ground_truth_label", hue="abnormality_type", palette="Set1")
    plt.title("CBIS-DDSM: Pathology Distribution by Abnormality Type", fontsize=13, fontweight="bold")
    plt.xlabel("Pathology Class")
    plt.ylabel("Mammogram Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=200)
    plt.close()

    # 2. Quality Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df["laplacian_variance"], kde=True, ax=axes[0, 0], color="navy")
    axes[0, 0].set_title("Sharpness: Laplacian Variance", fontweight="bold")

    sns.histplot(df["rms_contrast"], kde=True, ax=axes[0, 1], color="darkcyan")
    axes[0, 1].set_title("Contrast: RMS Contrast", fontweight="bold")

    sns.histplot(df["shannon_entropy"], kde=True, ax=axes[1, 0], color="purple")
    axes[1, 0].set_title("Shannon Information Entropy (bits)", fontweight="bold")

    sns.histplot(df["snr_db"], kde=True, ax=axes[1, 1], color="coral")
    axes[1, 1].set_title("Signal-to-Noise Ratio (dB)", fontweight="bold")

    plt.suptitle("CBIS-DDSM Mammography: Quality Metric Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_quality_metrics_distribution.png"), dpi=200)
    plt.close()

    # 3. Contrast Comparison: Masses vs Calcifications
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="abnormality_type", y="rms_contrast", hue="mammographic_view", palette="Pastel2")
    plt.title("RMS Contrast by Abnormality Type & View (CC vs MLO)", fontsize=13, fontweight="bold")
    plt.ylabel("RMS Contrast")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_contrast_by_abnormality.png"), dpi=200)
    plt.close()

    # 4. Correlation Heatmap
    metric_cols = [
        "laplacian_variance", "tenengrad_gradient", "rms_contrast",
        "michelson_contrast", "shannon_entropy", "snr_db", "composite_quality_score"
    ]
    plt.figure(figsize=(8, 6))
    corr = df[metric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title("Correlation Matrix of Mammography IQA Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_metric_correlation_matrix.png"), dpi=200)
    plt.close()

    # 5. Visual Comparison: High Quality vs Degraded Film Noise
    sorted_df = df.sort_values(by="composite_quality_score", ascending=False)
    best_samples = sorted_df.head(3)
    worst_samples = sorted_df.tail(3)

    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for i, (_, row) in enumerate(best_samples.iterrows()):
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(
            f"HIGH FIDELITY\nScore: {row['composite_quality_score']:.2f} | Lap: {row['laplacian_variance']:.1f}",
            color="green", fontsize=10, fontweight="bold"
        )
        axes[0, i].axis("off")

    for i, (_, row) in enumerate(worst_samples.iterrows()):
        img = cv2.imread(row["file_path"], cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(
            f"LOW CONTRAST / NOISE\nScore: {row['composite_quality_score']:.2f} | Lap: {row['laplacian_variance']:.1f}",
            color="red", fontsize=10, fontweight="bold"
        )
        axes[1, i].axis("off")

    plt.suptitle("CBIS-DDSM: Visual Comparison of Mammogram Quality Tiers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_quality_comparison_montage.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved all diagnostic visualization figures to: {output_dir}")


# ---------------------------------------------------------
# 5. Learned Decision Tree Training
# ---------------------------------------------------------
def train_and_evaluate_decision_tree(df: pd.DataFrame, output_dir: str):
    """
    Train and evaluate the Learned Decision Tree for mammography adaptive routing.
    """
    print("\n" + "=" * 60)
    print("CBIS-DDSM: LEARNED DECISION TREE ADAPTIVE ROUTING")
    print("=" * 60)

    tree = AdaptiveClinicalDecisionTree(max_depth=4, min_samples_leaf=4)
    results = tree.fit(df)

    print(f"Decision Tree Test Accuracy: {results['test_accuracy'] * 100:.2f}%\n")
    print("Feature Importances:")
    for feat, imp in results["feature_importances"].items():
        print(f"  - {feat:25s}: {imp:.4f}")

    print("\nLearned Decision Logic Rules:")
    print(results["rules_text"])

    df["predicted_routing_decision"] = tree.predict(df)

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="predicted_routing_decision", hue="predicted_routing_decision", palette="Dark2", legend=False)
    plt.title("Learned Decision Tree: Mammography Clinical Routing Actions", fontsize=13, fontweight="bold")
    plt.xlabel("Adaptive Routing Action")
    plt.ylabel("Number of Mammograms")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_adaptive_routing_distribution.png"), dpi=200)
    plt.close()

    csv_out = os.path.join(output_dir, "cbis_ddsm_quality_and_routing_metadata.csv")
    df.to_csv(csv_out, index=False)
    print(f"[INFO] Exported enriched clinical metadata to: {csv_out}")


# ---------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------
def main():
    print("Starting CBIS-DDSM Mammography Analytics Pipeline...")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = find_dataset_path()
    if data_dir is None:
        synth_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        data_dir = generate_synthetic_mammography_samples(synth_dir, num_samples=60)

    df = load_cbis_ddsm_metadata(data_dir)
    df_iqa = extract_dataset_iqa(df, max_samples=200)

    # Statistical Summary
    stats = df_iqa[[
        "laplacian_variance", "tenengrad_gradient", "rms_contrast",
        "michelson_contrast", "shannon_entropy", "snr_db", "composite_quality_score"
    ]].describe()
    print("\n" + "=" * 60)
    print("CBIS-DDSM IMAGE QUALITY ASSESSMENT: STATISTICAL SUMMARY")
    print("=" * 60)
    print(stats)
    stats.to_csv(os.path.join(output_dir, "cbis_ddsm_iqa_statistical_summary.csv"))

    generate_quality_visualizations(df_iqa, output_dir)
    train_and_evaluate_decision_tree(df_iqa, output_dir)
    print("\nCBIS-DDSM Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
