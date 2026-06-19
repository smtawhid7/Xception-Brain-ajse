# Xception-Based Brain Tumor Classification with GAN Augmentation and Multi-Technique Explainable AI

This repository contains the complete implementation accompanying the paper
**"Xception-Based Brain Tumor Classification with GAN Augmentation and Multi-Technique Explainable AI"**, submitted to the *AIUB Journal of Science and Engineering (AJSE)*.

All implementation code is consolidated in a single Jupyter notebook,
`gan-xception-journal-ready.ipynb`, organised into sequentially-executable
cells that reproduce every table and figure reported in the manuscript.

---

## 1. Repository Contents

```
repo/
├── gan-xception-journal-ready.ipynb   # Single notebook with all implementation code
└── README.md                          # This file
```

The notebook is self-contained. There are **no separate Python scripts,
configuration files, requirements files, or saved model-weight files** in
this repository. See Section 8 (Reproducibility Materials Scope) below for
the explicit list of what is and is not distributed.

---

## 2. What the Notebook Contains

The notebook is organised in sequential sections that mirror the manuscript:

1. **Data curation and de-duplication**
   - SHA-256 exact-duplicate removal
   - 64-bit perceptual hash (pHash) near-duplicate removal (Hamming distance ≤ 5)
   - Reproduces the 3,505 → 2,705 curation reported in Section IV.A of the paper

2. **Stratified 5-fold cross-validation setup**
   - Deterministic fold indices generated at runtime using `seed = 42`
   - 2,164 training / 541 validation per fold (real images only in validation)

3. **DCGAN training (fold-wise, augmentation-leakage-controlled)**
   - Independent DCGAN retrained from scratch inside every fold
   - Trained on that fold's training partition only
   - 300 synthetic images per class generated and added only to that fold's training set

4. **Two-stage classifier training**
   - Stage 1: Feature extraction (frozen backbone, head only) — 10 epochs
   - Stage 2: Deep fine-tuning (final 100 of 126 Xception layers) — 20 epochs
   - Baselines: DenseNet169, MobileNetV2, InceptionV3, EfficientNetV2-S, ConvNeXt-Tiny, Swin-Base

5. **Quantitative evaluation**
   - Accuracy, precision, recall, F1, per-class sensitivity/specificity, AUC
   - Expected Calibration Error (ECE)
   - Confusion matrix (Fold 3 reported)
   - ROC curves
   - 5-fold mean ± std with 95% BCa bootstrap confidence intervals

6. **Statistical testing**
   - Wilcoxon signed-rank tests across folds
   - McNemar's test on pooled predictions
   - Holm–Bonferroni correction for multiple comparisons

7. **Ablation study**
   - GAN augmentation contribution (+2.81%)
   - Fine-tuning contribution (+6.42%)
   - Transfer-learning contribution (+20.94%)

8. **Multi-technique Explainable AI evaluation (200 samples: 100 correct + 100 misclassified)**
   - Grad-CAM++, Score-CAM, LIME, SHAP (DeepSHAP)
   - Inter-method attribution agreement (IoU vs. Grad-CAM++ reference)
   - Deletion-AUC faithfulness with perturbation-baseline reference
   - Friedman test followed by Nemenyi post-hoc pairwise comparison

9. **External validation**
   - Zero-shot transfer to the Figshare T1-CE dataset (three overlapping classes)

10. **Figure generation**
    - All figures used in the manuscript are produced by cells in the notebook
    - See Section 7 below for the figure-to-cell mapping

---

## 3. Software Requirements

The notebook was developed and tested under the following stack. There is
no `requirements.txt` or `environment.yml` in this repository; the packages
below should be installed manually (or via `pip install <package>` for each
line) into a Python 3.8+ environment.

| Package        | Purpose                                          |
|----------------|--------------------------------------------------|
| `python>=3.8`  | Interpreter                                      |
| `tensorflow`   | Xception, DenseNet169, MobileNetV2, InceptionV3, EfficientNetV2-S, DCGAN |
| `torch`, `torchvision`, `timm` | ConvNeXt-Tiny, Swin-Base baselines |
| `numpy`, `pandas`, `scipy`, `scikit-learn` | Numerics, splits, statistics |
| `matplotlib`, `seaborn` | All figures                             |
| `opencv-python`, `scikit-image`, `Pillow` | Image I/O, pHash, SSIM   |
| `shap`, `lime` | XAI attribution                                  |
| `imagehash`    | Perceptual hashing for near-duplicate removal    |
| `jupyter`      | To execute the notebook                          |

### Hardware Used
- Dual NVIDIA Tesla T4 GPUs (16 GB VRAM each)
- ~5 GB disk space for the dataset; the notebook itself produces no large persisted artefacts

---

## 4. Dataset

The PMRAM Bangladeshi Brain Cancer MRI Dataset is **not** redistributed in
this repository. It is publicly available on Kaggle:

- **Source:** Kaggle — *PMRAM Bangladeshi Brain Cancer MRI Dataset*
- **Raw images:** 3,505 T1-weighted MRI slices
- **Curated (after de-duplication):** 2,705 images
  - Glioma: 675 · Meningioma: 665 · Normal: 690 · Pituitary: 675

### Expected Input Directory Structure

After downloading and extracting the dataset, point `DATASET_PATH` (top of
the notebook) to the directory containing the four class folders:

```
<DATASET_PATH>/
├── glioma/
├── meningioma/
├── no_tumor/        (used as "Normal" class)
└── pituitary_tumor/
```

The de-duplication cell in the notebook then produces the curated 2,705-image set automatically.

---

## 5. Configuration (defined inline in the notebook)

The following parameters are set near the top of the notebook. There is no
external config file:

```python
DATASET_PATH      = "<set to your local PMRAM Raw path>"
IMG_SIZE_GAN      = (128, 128)     # DCGAN resolution
IMG_SIZE_XCEP     = (299, 299)     # Xception input
LATENT_DIM        = 100            # DCGAN latent dimension
BATCH_SIZE        = 32
EPOCHS_GAN        = 2000           # Per-fold DCGAN training
N_GEN_PER_CLASS   = 300            # Synthetic images per class per fold
EPOCHS_FE         = 10             # Feature-extraction stage
EPOCHS_FT         = 20             # Fine-tuning stage
N_SPLITS          = 5              # Stratified k-fold
SEED              = 42             # Global random seed (numpy, tf, torch, python)
```

---

## 6. Reproduction Workflow

To reproduce all results in the paper:

1. Install the packages listed in Section 3.
2. Download the PMRAM dataset from Kaggle and set `DATASET_PATH`.
3. Open `gan-xception-journal-ready.ipynb` in Jupyter.
4. Execute the cells **top-to-bottom in order**. Each section corresponds to a
   manuscript subsection, and intermediate cells print the numbers and save
   the figures that appear in the paper.

Expected outcome (within ±0.5% depending on GPU/cuDNN version):
- Mean accuracy: **97.62% ± 0.51%** (95% CI 96.99–98.25%)
- Mean macro-AUC: **0.995**
- Mean ECE: **0.0148**

---

## 7. Mapping to Manuscript Tables and Figures

All tables and figures in the manuscript are produced by executing the
corresponding section of the notebook.

### Tables

| Manuscript Table | Content                                                                 | Notebook Section |
|------------------|-------------------------------------------------------------------------|------------------|
| Table I  (`tab1`)   | Comparative performance of seven architectures (5-fold CV, BCa CI)   | Section 5 (Quantitative Evaluation) |
| Table II (`tab1b`)  | Pairwise statistical comparison: Xception vs. baselines (Wilcoxon, McNemar, Holm) | Section 6 (Statistical Testing) |
| Table III (`tab2`)  | Per-fold accuracy and ECE for the Xception model                     | Section 5 |
| Table IV (`tab3`)   | Confusion matrix for the Xception model (Fold 3, 541 real validation images) | Section 5 |
| Table V (`tab4`)    | Per-class precision, recall, F1, sensitivity, specificity            | Section 5 |
| Table VI (`tab5`)   | Ablation study on Xception (mean over 5 folds)                       | Section 7 (Ablation) |
| Table VII (`tab:xai`)    | Quantitative XAI evaluation (IoU, deletion-AUC) on 200 samples  | Section 8 (XAI) |
| Table VIII (`tab:external`) | External validation on Figshare T1-CE                       | Section 9 (External Validation) |

### Figures

| Manuscript Figure | File                       | Content                                                                   | Notebook Section |
|-------------------|----------------------------|---------------------------------------------------------------------------|------------------|
| Figure 1 (`fig1`) | `dataset_distribution.png` | Representative T1-weighted MRI samples from the curated PMRAM dataset     | Section 1 (Data Curation) |
| Figure 2 (`fig2`) | TikZ diagram (in LaTeX)    | Overall pipeline architecture (no separate image file)                    | — |
| Figure 3 (`fig3`) | `classwise_performance.png`| Class-wise precision/recall/F1 for the Xception model                     | Section 5 |
| Figure 4 (`fig4`) | `confusion_matrix.png`     | Confusion matrix for Fold 3 (matches Table IV: 131 / 129 / 137 / 133)     | Section 5 |
| Figure 5 (`fig5`) | `roc_curve.png`            | Per-class ROC curves (macro-AUC = 0.995)                                  | Section 5 |
| Figure 6 (`fig6`) | `training_curves.png`      | Accuracy and loss curves for Fold 3 (feature-extraction + fine-tuning)    | Section 4 (Training) |
| Figure 7 (`fig7`) | `tsne_embedding.png`       | t-SNE feature embedding with class-name legend                            | Section 5 |
| Figure 8 (`fig8`) | `three_xai.png`            | Grad-CAM++, LIME, Score-CAM visualisations for a representative case      | Section 8 (XAI) |

---

## 8. Reproducibility Materials — Scope

To avoid any ambiguity flagged in earlier review rounds, this section makes
the scope of the distributed reproducibility materials explicit.

### Provided in this repository
- Single Jupyter notebook with the complete end-to-end pipeline
  (curation → fold splitting → DCGAN → classifier training → evaluation → XAI → external validation)
- Random-seed settings and inline configuration parameters
- This README with reproduction instructions and a figure/table-to-section map

### Generated deterministically at notebook runtime (not stored as files)
- The five stratified fold-split index sets (derived from `seed = 42` and the curated 2,705-image list)
- The five fold-specific DCGAN generators and the resulting synthetic images
- All trained classifier checkpoints
- All figures and tables shown in the manuscript

### Not provided (with rationale)
- **`requirements.txt` / `environment.yml`** — not distributed; the dependency list in Section 3 above plus the import cell of the notebook is the authoritative environment specification.
- **Trained model weight files** — not redistributed due to per-fold checkpoint size across seven architectures. The notebook contains the complete code needed to retrain every reported model under the same seeds and splits.
- **Patient identifiers / raw clinical metadata** — not available in the public PMRAM release; see the limitation note below.

---

## 9. Known Limitations

1. **Patient-level separation cannot be guaranteed.** The public PMRAM
   release distributes de-identified 2D slices without patient identifiers.
   The protocol provably eliminates *augmentation* leakage (synthetic images
   never reach the validation partition) and the perceptual-hash de-duplication
   removes near-identical slices, but it cannot substitute for true
   patient-grouped cross-validation. Accordingly, the benchmark is described
   as **augmentation-leakage-controlled**, not fully leakage-free.

2. **Single imaging modality.** Only T1-weighted MRI is used; multi-parametric
   MRI (T1, T2, FLAIR, contrast-enhanced) is identified as future work.

3. **External validation scope.** External validation is performed on a
   single additional dataset (Figshare) and only three overlapping classes.

4. **Numerical reproducibility.** Differences in GPU architecture, cuDNN
   version, and floating-point ordering may cause variations below
   approximately 0.5% accuracy relative to the reported values.

---

## 10. Citation

If you use this code, please cite the corresponding AJSE paper. A BibTeX
entry will be added upon final acceptance.

## 11. Contact

- S.M. Tawhid — 21-45470-3@student.aiub.edu
- Dip Nandi — dip.nandi@aiub.edu

## 12. Acknowledgments

- PMRAM Bangladeshi Brain Cancer MRI Dataset contributors (publicly released on Kaggle).
- American International University-Bangladesh (AIUB) for computational resources.
