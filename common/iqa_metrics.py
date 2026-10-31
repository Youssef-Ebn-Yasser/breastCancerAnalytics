"""
Image Quality Assessment (IQA) Module for Medical Breast Cancer Imaging.

This module provides No-Reference (NR-IQA) and Full-Reference (FR-IQA) metrics
tailored for medical image modalities:
- Histopathology (TCGA-BRCA)
- Mammography (CBIS-DDSM)
- Ultrasound (BUSI)

Supported Metrics:
1. Sharpness: Laplacian Variance, Tenengrad Gradient
2. Contrast: RMS Contrast, Michelson Contrast, Local Weber Contrast
3. Information: Shannon Entropy
4. Noise & Signal: Robust SNR (via MAD of high frequencies), CNR (with mask)
5. Modality-Specific: Ultrasound Speckle Index & ENL
6. Perceptual Quality: Natural Scene Statistics (MSCN distribution fitting)
7. Full Reference (when paired): PSNR, SSIM
"""

import numpy as np
import cv2
from scipy.stats import entropy, gennorm


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure input image is single-channel 8-bit or float grayscale.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 1:
            return image[:, :, 0]
    return image


def compute_laplacian_variance(image: np.ndarray) -> float:
    """
    Compute Sharpness using Laplacian Variance.
    Higher values indicate sharp edges and fine cellular/microcalcification details.
    Lower values correspond to out-of-focus blur, motion blur, or severe smoothing.
    """
    gray = ensure_grayscale(image)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(np.var(laplacian))


def compute_tenengrad(image: np.ndarray, ksize: int = 3, threshold: float = 0.0) -> float:
    """
    Compute Tenengrad Gradient Sharpness based on Sobel operators.
    Measures the sum of squared gradient magnitudes above an optional noise threshold.
    """
    gray = ensure_grayscale(image)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag_sq = gx ** 2 + gy ** 2
    if threshold > 0:
        grad_mag_sq = np.where(grad_mag_sq > (threshold ** 2), grad_mag_sq, 0.0)
    return float(np.mean(grad_mag_sq))


def compute_rms_contrast(image: np.ndarray) -> float:
    """
    Compute Root Mean Square (RMS) Contrast.
    Defined as the standard deviation of normalized pixel luminance intensities [0, 1].
    Low values highlight washed-out, underexposed, or low-contrast tissue images.
    """
    gray = ensure_grayscale(image)
    norm = gray.astype(np.float64) / 255.0 if gray.max() > 1.0 else gray.astype(np.float64)
    return float(np.std(norm))


def compute_michelson_contrast(image: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Compute Michelson Contrast across the dynamic range.
    C_michelson = (I_max - I_min) / (I_max + I_min + epsilon)
    """
    gray = ensure_grayscale(image)
    i_min = float(np.percentile(gray, 1))  # robust to single outlier pixels
    i_max = float(np.percentile(gray, 99))
    return float((i_max - i_min) / (i_max + i_min + epsilon))


def compute_shannon_entropy(image: np.ndarray, num_bins: int = 256) -> float:
    """
    Compute Shannon Information Entropy in bits.
    Measures the average information and textural richness in the image histogram.
    Images with blank glass slide backgrounds or saturation exhibit lower entropy.
    """
    gray = ensure_grayscale(image)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(entropy(hist, base=2))


def compute_snr_mad(image: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Estimate Signal-to-Noise Ratio (SNR in dB) blindly using Median Absolute Deviation (MAD)
    of high-frequency Laplacian residual.
    """
    gray = ensure_grayscale(image)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_signal = float(np.mean(gray))
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    # Noise standard deviation estimation via MAD (Donoho & Johnstone method)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    noise_sigma = float(mad / 0.6745)
    if noise_sigma < epsilon:
        return 50.0  # Cap at high SNR if virtually noise-free
    snr = mean_signal / (noise_sigma + epsilon)
    snr_db = 20.0 * np.log10(max(snr, epsilon))
    return float(np.clip(snr_db, -10.0, 50.0))


def compute_cnr(image: np.ndarray, mask: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR) between lesion region and surrounding background tissue.
    Requires a binary mask where lesion > 0.
    """
    gray = ensure_grayscale(image)
    if mask is None:
        return 0.0
    binary_mask = (mask > 0).astype(bool)
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[:, :, 0]

    lesion_pixels = gray[binary_mask]
    bg_pixels = gray[~binary_mask]

    if len(lesion_pixels) < 10 or len(bg_pixels) < 10:
        return 0.0

    mu_lesion = np.mean(lesion_pixels)
    mu_bg = np.mean(bg_pixels)
    var_lesion = np.var(lesion_pixels)
    var_bg = np.var(bg_pixels)

    cnr = abs(mu_lesion - mu_bg) / np.sqrt(var_lesion + var_bg + epsilon)
    return float(cnr)


def compute_ultrasound_speckle_metrics(image: np.ndarray) -> dict:
    """
    Compute speckle noise metrics for Ultrasound images:
    - Speckle Index (Cs): Standard deviation / Mean of homogeneous region.
    - Equivalent Number of Looks (ENL): Mean^2 / Var. Higher ENL implies effective speckle suppression.
    """
    gray = ensure_grayscale(image)
    h, w = gray.shape
    # Sample a central patch representing tissue parenchyma
    patch = gray[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
    mean_val = float(np.mean(patch))
    std_val = float(np.std(patch))
    if mean_val < 1e-4:
        return {"speckle_index": 0.0, "enl": 0.0}
    cs = std_val / mean_val
    enl = (mean_val ** 2) / (std_val ** 2 + 1e-6)
    return {"speckle_index": float(cs), "enl": float(enl)}


def compute_mscn_coefficients(image: np.ndarray, window_size: int = 7, sigma: float = 7.0 / 6.0) -> np.ndarray:
    """
    Compute Mean Subtracted Contrast Normalized (MSCN) coefficients used in BRISQUE / NSS.
    """
    gray = ensure_grayscale(image).astype(np.float64)
    kernel = cv2.getGaussianKernel(window_size, sigma)
    window = np.outer(kernel, kernel.transpose())
    mu = cv2.filter2D(gray, -1, window)
    mu_sq = mu * mu
    sigma_sq = cv2.filter2D(gray * gray, -1, window) - mu_sq
    sigma_val = np.sqrt(np.maximum(sigma_sq, 0.0))
    mscn = (gray - mu) / (sigma_val + 1.0)
    return mscn


def compute_perceptual_nss_score(image: np.ndarray) -> float:
    """
    Compute No-Reference Natural Scene Statistics (NSS) proxy score.
    Fits Generalized Gaussian Distribution (GGD) shape parameter alpha to MSCN coefficients.
    Natural, high-fidelity images exhibit alpha near 1.8 - 2.2.
    Heavy blur or synthetic noise deviations drive alpha away from natural bounds.
    Returns normalized quality index [0 (worst) to 1 (best)].
    """
    try:
        mscn = compute_mscn_coefficients(image)
        flat = mscn.ravel()
        # Estimate shape parameter alpha
        params = gennorm.fit(flat)
        alpha = float(params[0])
        # Gaussian distribution corresponds to alpha=2.0; compute distance to 2.0
        dist = abs(alpha - 2.0)
        quality = 1.0 / (1.0 + dist)
        return float(np.clip(quality, 0.0, 1.0))
    except Exception:
        return 0.5


def compute_psnr(image: np.ndarray, ref_image: np.ndarray) -> float:
    """
    Full-Reference Peak Signal-to-Noise Ratio (PSNR in dB).
    """
    gray = ensure_grayscale(image).astype(np.float64)
    ref = ensure_grayscale(ref_image).astype(np.float64)
    mse = np.mean((gray - ref) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim(image: np.ndarray, ref_image: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between image and reference.
    """
    gray = ensure_grayscale(image).astype(np.float64)
    ref = ensure_grayscale(ref_image).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(ref, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray * gray, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray * ref, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def extract_all_iqa_metrics(
    image: np.ndarray, mask: np.ndarray = None, modality: str = "generic"
) -> dict:
    """
    Extract a comprehensive feature vector of Image Quality Assessment metrics.

    Parameters:
        image: np.ndarray of the medical image (RGB or Grayscale).
        mask: optional binary mask of tumor/lesion region.
        modality: "histopathology", "mammography", "ultrasound", or "generic".

    Returns:
        Dictionary of computed metrics and composite quality score.
    """
    gray = ensure_grayscale(image)
    h, w = gray.shape[:2]

    lap_var = compute_laplacian_variance(gray)
    tenengrad = compute_tenengrad(gray)
    rms_contrast = compute_rms_contrast(gray)
    michelson = compute_michelson_contrast(gray)
    entropy_val = compute_shannon_entropy(gray)
    snr_db = compute_snr_mad(gray)
    nss_quality = compute_perceptual_nss_score(gray)

    results = {
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(w / (h + 1e-6)),
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "laplacian_variance": lap_var,
        "tenengrad_gradient": tenengrad,
        "rms_contrast": rms_contrast,
        "michelson_contrast": michelson,
        "shannon_entropy": entropy_val,
        "snr_db": snr_db,
        "perceptual_nss_score": nss_quality,
    }

    if mask is not None:
        results["cnr"] = compute_cnr(gray, mask)
    else:
        results["cnr"] = 0.0

    if modality.lower() == "ultrasound":
        us_metrics = compute_ultrasound_speckle_metrics(gray)
        results.update(us_metrics)
    elif modality.lower() == "histopathology":
        # Staining color saturation metric (for RGB histology slides)
        if len(image.shape) == 3 and image.shape[2] >= 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            results["stain_saturation_mean"] = float(np.mean(sat))
            results["stain_saturation_std"] = float(np.std(sat))
        else:
            results["stain_saturation_mean"] = 0.0
            results["stain_saturation_std"] = 0.0
    elif modality.lower() == "mammography":
        # Percentage of high-density fibroglandular parenchyma (pixels > 120)
        tissue_mask = gray > 15
        dense_ratio = float(np.sum((gray > 120) & tissue_mask) / (np.sum(tissue_mask) + 1e-6))
        results["breast_density_ratio"] = dense_ratio

    # Compute a normalized Composite Quality Score (0.0 to 1.0)
    # Derived from sigmoid normalization of normalized sharpness, contrast, SNR, and entropy
    norm_lap = np.clip(np.log1p(lap_var) / 8.0, 0.0, 1.0)
    norm_contrast = np.clip(rms_contrast * 4.0, 0.0, 1.0)
    norm_snr = np.clip((snr_db - 5.0) / 25.0, 0.0, 1.0)
    norm_ent = np.clip(entropy_val / 8.0, 0.0, 1.0)

    composite_score = float(
        0.30 * norm_lap + 0.25 * norm_contrast + 0.25 * norm_snr + 0.20 * norm_ent
    )
    results["composite_quality_score"] = float(np.clip(composite_score, 0.0, 1.0))

    # Quality Tier Classification
    if results["composite_quality_score"] >= 0.70:
        results["quality_tier"] = "High_Quality"
    elif results["composite_quality_score"] >= 0.45:
        results["quality_tier"] = "Medium_Quality"
    elif results["composite_quality_score"] >= 0.25:
        results["quality_tier"] = "Low_Quality"
    else:
        results["quality_tier"] = "Degraded"

    return results
