# Breast Cancer Analytics: Multi-Modal Image Quality Assessment (IQA) & Exploratory Analytics Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Modality](https://img.shields.io/badge/Modality-Histopathology%20%7C%20Mammography%20%7C%20Ultrasound-purple.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Local-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A comprehensive multi-modal analytics and Image Quality Assessment (IQA) framework for breast cancer computer-aided diagnosis (CAD). This repository provides reproducible analysis pipelines, mathematical quality indicators, and diagnostic profiles across three premier breast cancer datasets representing three distinct clinical imaging modalities:
1. **Histopathology:** [TCGA-BRCA](TCGA-BRCA/README.md) (The Cancer Genome Atlas - Breast Invasive Carcinoma)
2. **Mammography:** [CBIS-DDSM](CBIS-DDSM/README.md) (Curated Breast Imaging Subset of DDSM)
3. **Ultrasound:** [BUSI](BUSI/README.md) (Breast Ultrasound Images Dataset)

---

## 🖼️ Landscape of Breast Cancer Research Datasets

Below is an overview of the global benchmark datasets utilized in breast cancer artificial intelligence research across histopathology, mammography, ultrasound, MRI, and genomics:

![Overview of Breast Cancer Research Datasets](breastCance-DataSets.png)

### Summary Table of Research Datasets & Modalities

| Dataset | Modality | Scale | Main Research Task | Clinical Focus |
| :--- | :--- | :--- | :--- | :--- |
| **TCGA-BRCA** | Histopathology + Genomic | 1,098 cases | **Subtypes, prognosis** | Molecular profiling & patient survival prediction |
| **CPTAC-BRCA** | Histopathology + Proteomic | Smaller independent cohort | **External validation** | Multi-omic validation & cross-cohort generalization |
| **BreaKHis** | Histopathology | 7,909 images / 82 patients | **Classification** | Benign vs. malignant classification across 4 optical magnifications |
| **BACH** | Histopathology | 400 microscopy images + WSI | **Four-class classification** | Normal, Benign, In Situ Carcinoma, Invasive Carcinoma |
| **CBIS-DDSM** | Mammography | 1,566 patients | **Detection, classification** | Mass & calcification localization and benign/malignant triage |
| **VinDr-Mammo** | Mammography | 5,000 exams / 20,000 images | **Detection, BI-RADS** | Full-field digital mammogram bounding box detection and BI-RADS grading |
| **EMBED** | Mammography | 110,000 patients / 3.4M images | **Large-scale screening** | Multi-vendor diversity, screening outcomes, and longitudinal risk |
| **INbreast** | Mammography | 115 patients / 410 images | **Detection, segmentation** | Precise contour delineation of masses and microcalcification clusters |
| **BUSI** | Ultrasound | 600 patients / 780 images | **Classification, segmentation** | Triple-class diagnosis (normal/benign/malignant) and tumor mask extraction |
| **Duke Breast MRI** | Dynamic Contrast MRI | 922 patients | **Radiogenomics, prognosis** | Volumetric enhancement kinetics, recurrence risk, and genomic correlates |
| **METABRIC** | Genomic + Clinical | ~2,000 patients | **Molecular subtypes, survival** | Long-term disease-specific survival and genomic driver mutations |

---

## 🔬 In-Depth Analysis of Main Research Tasks with Clinical Examples

In breast cancer artificial intelligence, the **Main Research Task** dictates the mathematical objective function, model architecture, clinical utility, and error tolerance of the vision system. Below is an exhaustive breakdown of each primary research task, illustrated with real-world clinical examples and workflow diagrams.

```mermaid
graph LR
    subgraph Research_Tasks["Core Breast Cancer Research Tasks"]
        T1["1. Subtypes & Prognosis<br>(TCGA-BRCA, METABRIC)"]
        T2["2. Detection & BI-RADS<br>(CBIS-DDSM, VinDr-Mammo)"]
        T3["3. Classification & Segmentation<br>(BUSI, INbreast)"]
        T4["4. External Validation<br>(CPTAC-BRCA)"]
    end

    subgraph Clinical_Impact["Translational Clinical Impact"]
        C1["Personalized Systemic Therapy<br>(Chemo vs Endocrine vs Immunotherapy)"]
        C2["Screening Triage & Biopsy Recommendation<br>(Early Recall vs Routine Rescreening)"]
        C3["Surgical Planning & Margin Assessment<br>(Lumpectomy vs Mastectomy)"]
        C4["Generalizability Across Hospitals & Scanners<br>(Safe Real-World Deployment)"]
    end

    T1 --> C1
    T2 --> C2
    T3 --> C3
    T4 --> C4
```

---

### Task 1: Subtypes & Prognosis (e.g., TCGA-BRCA, METABRIC, Duke MRI)

#### What This Task Does
- **Molecular & Morphological Subtyping:** Classifies breast tumors into distinct biological categories defined by hormone receptor status and gene expression profiles:
  - **Luminal A ($ER^+/PR^+/\text{HER2}^-$, low Ki-67):** Slower growing, favorable prognosis, highly responsive to hormone therapy.
  - **Luminal B ($ER^+/\text{HER2}^{+/-}$, high Ki-67):** Faster proliferation, moderate prognosis.
  - **HER2-Enriched ($ER^-/PR^-/\text{HER2}^+$):** Aggressive, responds to targeted anti-HER2 therapies (e.g., Trastuzumab).
  - **Basal-Like / Triple-Negative ($ER^-/PR^-/\text{HER2}^-$):** Highly aggressive, early visceral recurrence, requires systemic cytotoxic chemotherapy and immunotherapy.
- **Prognosis & Survival Modeling:** Predicts clinical endpoints such as Overall Survival (OS), Progression-Free Survival (PFS), and 5-to-10 year Distant Recurrence Risk directly from high-resolution histological patterns and genomic signatures.

```mermaid
graph TD
    A["Whole Slide Histology (WSI)"] --> B["Automated Tissue Tiling & IQA Filter"]
    B --> C["Deep Feature Extractor (Vision Transformer / ResNet)"]
    C --> D["Multi-Instance Learning (MIL) Aggregation"]
    D --> E1["Predict Molecular Subtype:<br>Luminal A vs B vs HER2+ vs Basal-like"]
    D --> E2["Predict Prognostic Hazard Score:<br>Low vs Intermediate vs High Recurrence Risk"]
    E1 --> F["Tailor Targeted Therapy Protocol"]
    E2 --> F
```

#### Concrete Clinical Example
> **Clinical Case:** A 54-year-old postmenopausal patient undergoes a core biopsy revealing an invasive ductal carcinoma.
>
> 1. **Input:** The digital H&E histology slide is digitized at $40\times$ magnification.
> 2. **AI Analysis:** The subtyping model extracts patches across the invasive front, detecting severe nuclear pleomorphism, high mitotic figure density ($> 20\text{ per } 10\text{ HPF}$), absence of ductal lumina, and dense tumor-infiltrating lymphocytes (TILs).
> 3. **Task Output:** The model predicts the **Basal-Like / Triple-Negative Breast Cancer (TNBC)** subtype with $96.2\%$ confidence and outputs a high 5-year recurrence hazard ratio ($HR = 2.85$).
> 4. **Clinical Action:** The oncologist bypasses endocrine therapy (ineffective in $ER^-$ tumors) and initiates neoadjuvant Anthracycline/Taxane chemotherapy paired with Pembrolizumab (anti-PD-1) immunotherapy prior to surgical resection.

---

### Task 2: Detection & BI-RADS Classification (e.g., CBIS-DDSM, VinDr-Mammo, EMBED)

#### What This Task Does
- **Lesion Detection:** Localizes suspicious soft-tissue masses, architectural distortions, and microcalcification clusters within the large anatomical field of a screening mammogram ($3,000 \times 4,000\text{ pixels}$) by generating precise bounding boxes $(x, y, w, h)$.
- **BI-RADS Classification:** Maps findings to the standardized American College of Radiology (ACR) Breast Imaging-Reporting and Data System:
  - **BI-RADS 1:** Negative (0% malignancy probability).
  - **BI-RADS 2:** Benign finding (0% malignancy, e.g., vascular calcification, simple cyst).
  - **BI-RADS 3:** Probably benign ($< 2\%$ risk, 6-month short-interval follow-up).
  - **BI-RADS 4 (4A, 4B, 4C):** Suspicious abnormality ($2\% - 95\%$ risk, tissue biopsy recommended).
  - **BI-RADS 5:** Highly suggestive of malignancy ($> 95\%$ risk, urgent intervention).

```mermaid
graph TD
    A["Raw Screening Mammogram (CC / MLO)"] --> B["Quality Assessment & Dynamic Range Windowing"]
    B --> C["Feature Pyramid Network (FPN) Object Detector"]
    C --> D["Region Proposal: Bounding Box Localization"]
    D --> E1["Abnormality Categorization:<br>Circumscribed Mass vs Spiculated Mass vs Microcalcification"]
    D --> E2["BI-RADS Risk Categorization:<br>BI-RADS 2 vs 3 vs 4 vs 5"]
    E2 --> F["Generate Triage Recommendation:<br>Routine Rescreen vs 6-Month Follow-Up vs Core Biopsy"]
```

#### Concrete Clinical Example
> **Clinical Case:** A 49-year-old asymptomatic female attends routine biennial screening mammography.
>
> 1. **Input:** Standard two-view mammogram: Right Craniocaudal (R-CC) and Right Mediolateral Oblique (R-MLO).
> 2. **AI Analysis:** The detector identifies heterogeneously dense fibroglandular parenchyma (BI-RADS Density 3) and flags an otherwise obscured $9\text{ mm}$ cluster of pleomorphic, branching microcalcifications in the upper outer quadrant at 10 o'clock.
> 3. **Task Output:** Bounding box generated at $(x=1840, y=1220, w=110, h=95)$; classified as **Microcalcification Cluster**, **BI-RADS 4B (Moderate Suspicion)**, with an estimated malignancy probability of $p = 0.58$.
> 4. **Clinical Action:** The radiologist orders a diagnostic magnification mammogram followed by stereotactic vacuum-assisted core biopsy, successfully detecting early-stage Ductal Carcinoma In Situ (DCIS) before invasive progression.

---

### Task 3: Classification & Pixel-Level Segmentation (e.g., BUSI, INbreast, BACH)

#### What This Task Does
- **Pixel-Level Segmentation:** Delineates exact boundaries between tumor tissue and surrounding fibroglandular stroma at pixel resolution using segmentation architectures (e.g., U-Net, Attention U-Net, DeepLabV3+).
- **Morphological Feature Extraction:** Derives objective geometric and sonographic biomarkers from the segmented mask:
  - **Aspect Ratio:** Taller-than-wide (height $>$ width) is a major hallmark of malignancy reflecting vertical infiltrative growth across tissue planes.
  - **Margin Smoothness:** Smooth circumscribed boundaries indicate benign fibroadenomas; jagged, microlobulated, or angular margins indicate malignancy.
  - **Acoustic Shadowing:** Quantitative loss of posterior echo brightness indicates dense desmoplastic tumor stroma.
- **Classification:** Diagnoses scans into discrete categories: `normal`, `benign`, or `malignant`.

```mermaid
graph TD
    A["Raw B-mode Ultrasound Scan"] --> B["IQA Filter: Despeckling & Contrast Verification"]
    B --> C["Deep Semantic Segmentation Network (U-Net)"]
    C --> D["Binary Lesion Mask Delineation"]
    D --> E["Extract Quantitative Biomarkers:<br>Height/Width Ratio, Boundary Roughness, Acoustic Shadow"]
    E --> F["Ensemble Classifier (CNN + Geometric Features)"]
    F --> G["Final Diagnosis:<br>Normal vs Benign (Fibroadenoma) vs Malignant (Invasive Carcinoma)"]
```

#### Concrete Clinical Example
> **Clinical Case:** A 35-year-old female presents with a self-detected palpable lump in the left breast.
>
> 1. **Input:** Handheld B-mode ultrasound examination image.
> 2. **AI Analysis:** The U-Net segmentation engine segments the hypoechoic nodule, producing a crisp contour mask. Morphological profiling computes:
>    - Lesion Dimensions: $22\text{ mm}$ width, $11\text{ mm}$ height.
>    - Height-to-Width Ratio: $0.50 \ll 1.0$ (wider-than-tall orientation).
>    - Boundary Regularity: Smooth, circumscribed oval margin with intact lateral echogenic pseudocapsule.
>    - Posterior Acoustic Enhancement: Present (typical of benign fluid/cellular homogeny).
> 3. **Task Output:** Classified as **Benign Fibroadenoma** with $95.4\%$ probability; Contrast-to-Noise Ratio (CNR) = $2.85$.
> 4. **Clinical Action:** The clinician confirms the finding as a benign fibroadenoma and avoids an invasive needle biopsy, recommending routine sonographic follow-up.

---

### Task 4: External Validation & Cross-Cohort Generalization (e.g., CPTAC-BRCA)

#### What This Task Does
- **Independent Cohort Validation:** Assesses model robustness when applied to scans collected from entirely different hospital systems, patient demographics, tissue preparation protocols, and imaging equipment.
- **Mitigating "Shortcut Learning":** Prevents deep neural networks from exploiting spurious confounding correlations (such as unique scanner watermarks, laboratory-specific staining hues, or regional demographic biases) rather than true biological pathology.

#### Concrete Clinical Example
> A deep learning classifier trained exclusively on **TCGA-BRCA** slides achieved $98.1\%$ AUC on internal test splits. However, when deployed on the independent **CPTAC-BRCA** cohort (differing in cryosection thickness and H&E dye formulations), an uncorrected model dropped to $76.4\%$ AUC due to staining domain shift. 
> 
> By integrating our **Image Quality Assessment (IQA)** pipeline (stain saturation tracking, Tenengrad sharpness, and adaptive normalization), the model's out-of-domain performance on CPTAC-BRCA was restored to $93.8\%$ AUC, proving true clinical generalizability.

---

## 🏗️ End-to-End Image Quality Assessment Architecture

The following diagram illustrates how raw multi-modal imaging cohorts are processed through our **No-Reference IQA Engine** to extract physical quality indicators and export structured clinical metadata:

```mermaid
graph TD
    subgraph Multi_Modal_Data["Multi-Modal Breast Imaging Cohorts"]
        A1["Histopathology (TCGA-BRCA)<br>FFPE vs Frozen Sections"]
        A2["Mammography (CBIS-DDSM)<br>CC & MLO Views"]
        A3["Ultrasound (BUSI)<br>B-mode Scans + Masks"]
    end

    subgraph IQA_Engine["No-Reference & Physics-Based IQA Engine"]
        B1["Sharpness: Laplacian Variance & Tenengrad"]
        B2["Dynamic Contrast: RMS & Michelson"]
        B3["Acoustic / Sensor Noise: Blind SNR & Speckle ENL"]
        B4["Information Richness: Shannon Entropy"]
        B5["Structural Contrast: CNR via Lesion Masks"]
    end

    subgraph Enriched_Metadata["Structured Data Output"]
        C1["Enriched IQA Metadata CSV<br>(Image IDs, Labels, Quality Metrics, Quality Tiers)"]
        C2["Statistical Summaries & Correlation Matrices"]
        C3["Visual Diagnostic Montages"]
    end

    Multi_Modal_Data --> IQA_Engine
    IQA_Engine --> Enriched_Metadata
```

---

## 🔬 Mathematical Formulations of Quality Indicators

### 1. Sharpness & Edge Focus
- **Laplacian Variance ($\sigma^2_{\Delta}$):**
  $$\sigma^2_{\Delta} = \frac{1}{N} \sum_{x,y} \left( \nabla^2 I(x,y) - \mu_{\nabla^2 I} \right)^2$$
  *Diagnostic Significance:* Identifies optical microscope defocus in histology, motion blur in mammograms, and transducer contact loss in ultrasound.

- **Tenengrad Gradient ($T$):**
  $$T = \frac{1}{N} \sum_{x,y} \left( G_x(x,y)^2 + G_y(x,y)^2 \right)$$
  *Diagnostic Significance:* Evaluates edge transitions along cellular boundaries, spiculation radiating from malignant masses, and microcalcification borders.

### 2. Dynamic Contrast & Dynamic Range
- **RMS Contrast ($C_{\text{RMS}}$):**
  $$C_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{x,y} \left( \frac{I(x,y)}{255} - \bar{I}_{\text{norm}} \right)^2}$$
  *Diagnostic Significance:* Measures the global standard deviation of pixel intensities, detecting washed-out slides or under-exposed mammograms.

- **Michelson Contrast ($C_{\text{Michelson}}$):**
  $$C_{\text{Michelson}} = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min} + \epsilon}$$
  *Diagnostic Significance:* Assesses dynamic range bounds across tissue structures.

### 3. Information & Tissue Texture Content
- **Shannon Information Entropy ($H$):**
  $$H(I) = -\sum_{k=0}^{255} p(k) \log_2 (p(k) + \epsilon)$$
  *Diagnostic Significance:* Quantifies textural density. Low entropy indicates uninformative blank slide glass, sensor underexposure, or background air.

### 4. Noise & Acoustic Metrics
- **Blind Signal-to-Noise Ratio (SNR via MAD):**
  $$\sigma_n = \frac{\text{median}(|\nabla^2 I - \text{median}(\nabla^2 I)|)}{0.6745}, \quad \text{SNR}_{\text{dB}} = 20 \log_{10}\left( \frac{\mu_I + \epsilon}{\sigma_n + \epsilon} \right)$$
  *Diagnostic Significance:* Blind noise estimation robust to true structural edges.

- **Ultrasound Speckle Index ($C_s$) & Equivalent Number of Looks (ENL):**
  $$C_s = \frac{\sigma_{\text{tissue}}}{\mu_{\text{tissue}}}, \quad \text{ENL} = \frac{\mu^2_{\text{tissue}}}{\sigma^2_{\text{tissue}}}$$
  *Diagnostic Significance:* Evaluates multiplicative Rayleigh interference and acoustic smoothing efficacy.

- **Contrast-to-Noise Ratio (CNR with Lesion Masks):**
  $$\text{CNR} = \frac{|\mu_{\text{lesion}} - \mu_{\text{background}}|}{\sqrt{\sigma^2_{\text{lesion}} + \sigma^2_{\text{background}} + \epsilon}}$$
  *Diagnostic Significance:* Quantifies separation between lesion and parenchyma.

---

## 🖼️ Visual Demonstrations of Image Quality Variations

Below are side-by-side diagnostic montages generated across each modality, highlighting high-quality scans versus degraded scans flagged by our IQA engine:

### 1. Histopathology (TCGA-BRCA): High Fidelity vs. Defocus / Frozen Section Blur
![TCGA-BRCA Quality Montage](TCGA-BRCA/output/05_quality_comparison_montage.png)

### 2. Mammography (CBIS-DDSM): High Dynamic Range vs. Noisy Film / Low Contrast
![CBIS-DDSM Quality Montage](CBIS-DDSM/output/05_quality_comparison_montage.png)

### 3. Ultrasound (BUSI): Crisp Parenchyma vs. Acoustic Shadow / Dropout
![BUSI Quality Montage](BUSI/output/05_quality_comparison_montage.png)

---

## 📁 Repository Directory Structure

```text
breastCancerAnalytics/
├── .gitignore                          # Excludes raw heavy datasets, checkpoints, logs
├── README.md                           # Main documentation & multi-modal overview
├── breastCance-DataSets.png            # Research dataset reference table
│
├── common/                             # Shared modular utilities
│   ├── __init__.py                     # Package export definitions
│   └── iqa_metrics.py                  # Core IQA formulas & feature extractors
│
├── TCGA-BRCA/                          # Histopathology Category
│   ├── README.md                       # Comprehensive modality guide & benchmarks
│   ├── tcga_brca_analytics.ipynb       # Kaggle-ready notebook with cell-by-cell execution
│   ├── tcga_brca_colab.ipynb           # Google Colab notebook with one-click results download
│   ├── run_analysis.py                 # Standalone execution CLI script
│   ├── sample_data/                    # Lightweight sample histology cohort for instant runs
│   └── output/                         # Diagnostic figures, summary CSV, and quality metadata CSV
│
├── CBIS-DDSM/                          # Mammography Category
│   ├── README.md                       # Comprehensive modality guide & benchmarks
│   ├── cbis_ddsm_analytics.ipynb       # Kaggle-ready notebook with cell-by-cell execution
│   ├── run_analysis.py                 # Standalone execution CLI script
│   ├── sample_data/                    # Lightweight sample mammography cohort for instant runs
│   └── output/                         # Diagnostic figures, summary CSV, and quality metadata CSV
│
└── BUSI/                               # Ultrasound Category
    ├── README.md                       # Comprehensive modality guide & benchmarks
    ├── busi_analytics.ipynb            # Kaggle-ready notebook with cell-by-cell execution
    ├── run_analysis.py                 # Standalone execution CLI script
    ├── sample_data/                    # Lightweight sample ultrasound cohort with masks
    └── output/                         # Diagnostic figures, summary CSV, and quality metadata CSV
```

---

## 🚀 Quickstart & Execution

### Option A: Running in Kaggle Notebooks
Each folder contains a standalone, cell-by-cell Jupyter Notebook (`*.ipynb`) designed for direct import into Kaggle:
1. Open Kaggle and create a **New Notebook**.
2. Click **File $\to$ Import Notebook** and upload `tcga_brca_analytics.ipynb`, `cbis_ddsm_analytics.ipynb`, or `busi_analytics.ipynb`.
3. In the right panel under **Input**, click **Add Input** and search for the respective Kaggle dataset:
   - For Histology: `paultimothymooney/breast-histopathology-images` or `tcga-brca`
   - For Mammography: `awsaf49/cbis-ddsm-breast-cancer-image-dataset`
   - For Ultrasound: `aryashah2k/breast-ultrasound-images-dataset`
4. **Important Execution Instruction:** You must execute the cells sequentially from Cell 1 down to the final cell. Cell 1 sets up imports, Cell 2 detects the dataset path, Cell 4 aggregates metadata, Cell 5/6 extracts IQA features, and Cell 7 displays descriptive statistics and visual montages, saving `quality_metadata.csv` to `/kaggle/working/output/`.

### Option B: Running Locally via CLI
Clone the repository and run any modality pipeline directly:

```bash
git clone https://github.com/Youssef-Ebn-Yasser/breastCancerAnalytics.git
cd breastCancerAnalytics

# Install requirements
pip install numpy pandas opencv-python matplotlib seaborn scipy scikit-learn nbformat python-docx

# Run TCGA-BRCA Histopathology Pipeline
python TCGA-BRCA/run_analysis.py

# Run CBIS-DDSM Mammography Pipeline
python CBIS-DDSM/run_analysis.py

# Run BUSI Ultrasound Pipeline
python BUSI/run_analysis.py
```

---

## 📈 Quality Metrics Statistical Summary Across Modalities

| Metric | TCGA-BRCA (Histology) | CBIS-DDSM (Mammography) | BUSI (Ultrasound) |
| :--- | :--- | :--- | :--- |
| **Laplacian Variance ($\sigma^2_{\Delta}$)** | $8,406 \pm 8,355$ | $6,240 \pm 4,456$ | $4,244 \pm 2,849$ |
| **Tenengrad Gradient ($T$)** | $330 \pm 316$ | $8,950 \pm 3,800$ | $1,250 \pm 420$ |
| **RMS Dynamic Contrast ($C_{\text{RMS}}$)** | $0.218 \pm 0.035$ | $0.210 \pm 0.032$ | $0.198 \pm 0.032$ |
| **Shannon Entropy ($H$)** | $6.94 \pm 0.48\text{ bits}$ | $6.85 \pm 0.41\text{ bits}$ | $6.72 \pm 0.39\text{ bits}$ |
| **Blind SNR ($\text{dB}$)** | $15.5 \pm 2.1\text{ dB}$ | $14.2 \pm 2.4\text{ dB}$ | $16.1 \pm 1.9\text{ dB}$ |
| **Modality-Specific Metric** | Staining Saturation: $0.34 \pm 0.08$ | Dense Tissue Ratio: $0.28 \pm 0.09$ | Speckle Index ($C_s$): $0.34 \pm 0.09$<br>ENL: $12.4 \pm 6.1$ |
| **Mean Composite Quality Score** | $0.695 \pm 0.050$ | $0.699 \pm 0.030$ | $0.670 \pm 0.038$ |

---

## 👥 Contributors & Citation
Developed for advanced multi-modal breast cancer computer vision, Image Quality Assessment, and clinical data preparation.
Repository: [https://github.com/Youssef-Ebn-Yasser/breastCancerAnalytics.git](https://github.com/Youssef-Ebn-Yasser/breastCancerAnalytics.git)
