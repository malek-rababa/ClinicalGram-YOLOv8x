# ClinicalGram-YOLOv8x
Six-notebook codebase implementing a 10-step failure-guided, instance-balanced pipeline for clinical Gram-stain bacteria detection with YOLOv8x (unbalanced baseline, split builder, augmentation gallery, balanced training/validation, metrics, and prediction–GT error analysis)
# ClinicalGram — Failure-Guided 10-Step Pipeline with YOLOv8x

Code and notebooks for clinical Gram-stain bacterial detection using YOLOv8x.  
This repo contains an **unbalanced baseline reproduction** and a **novel instance-balanced pipeline** guided by validation failures. It also includes paper-ready augmentation figures, class-distribution audits, and prediction–GT overlay error analysis.

> Classes (YOLO IDs): `0=neg_cocci`, `1=pos_cocci`, `2=neg_bacilli`, `3=pos_bacilli`

---

## 📁 Notebooks (6)

1. **ClinicalGram_4Class_YOLOv8x_Unbalanced_Baseline.ipynb**  
   Train/validate YOLOv8x on the *original unbalanced* clinical dataset (authors’ split).

2. **ClinicalGram_4Class_YOLOv8x_InstanceAware_Balanced_Splits.ipynb**  
   Build **balanced** train/val/test splits by selecting images to hit per-class instance targets.

3. **Paper_Figures_Augmentation_Gallery_Microscopy.ipynb**  
   One-image gallery applying 5 Albumentations (CLAHE, Brightness/Contrast, Elastic, strong GaussianBlur, moderate Sharpen) for the paper’s methods figure.

4. **ClinicalGram_YOLOv8x_BalancedSplit_AllAugs_EndToEnd.ipynb**  
   Train/validate YOLOv8x on the **balanced** split with the chosen augmentations; logs, plots, and best weights.

5. **ClinicalGram_YOLOv8x_Prediction-GT_Overlay_ErrorAnalysis.ipynb**  
   Render predictions with class-colored boxes, **dashed GT boxes**, and **red X** markers for IoU≥0.5 misclassifications.

6. **ClinicalGram_InstanceCounts_PerSplit_Audit.ipynb**  
   Per-split class-instance counters for **train / val / test** sanity checks.

---

## 🔟 Failure-Guided, Instance-Balanced Strategy (Overview)

1. **Train a baseline** on the unbalanced split (no rebalancing).  
2. **Evaluate on validation** and collect failure cases (misses, confusions, low-IoU).  
3. **Aggregate failure patterns** per class and per phenotype (cocci/bacilli, G±).  
4. **Map failures ↔ visual distortions** (blur, brightness, stain variability, elastic shape).  
5. **Select biologically-plausible augmentations** that target those distortions.  
6. **Construct instance-balanced splits** (greedy per-image instance accounting; keep val/test clean).  
7. **Adjust hyperparameters** (LR, weight decay, mosaic schedule) for the new data volume.  
8. **Retrain with targeted augs**, turning off harmful ones (e.g., mixup for small objects).  
9. **Validate again and iterate** on augs/thresholds using **validation only** (avoid leakage).  
10. **Produce paper artifacts**: PR curves, confusion matrices, overlay analyses, and figure galleries.

---

## 🛠️ Environment

- Python ≥ 3.10  
- CUDA-enabled PyTorch (tested with Torch 2.x + CUDA 12.x)  
- Ultralytics (YOLOv8) 8.x  
- Albumentations, OpenCV-Python, Matplotlib, Pandas, TQDM

```bash
# example (conda + pip)
conda create -n clinicalgram python=3.12 -y
conda activate clinicalgram
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics albumentations opencv-python matplotlib pandas tqdm

Data Layout

/home/<user>/Documents/dataset/
  Clinical/                     # raw labels/images (authors’ set)
    images/
    labels/
  clinical_datset/last_clinical_DS/orginal/   # authors' split with data.yaml
    train/{images,labels}
    val/{images,labels}
    test/{images,labels}
    data.yaml
  dataset/Allaugs/              # balanced split + augs experiment
    train/{images,labels}
    val/{images,labels}
    test/{images,labels}
    data.yaml


 Quickstart
Unbalanced baseline

Open ClinicalGram_4Class_YOLOv8x_Unbalanced_Baseline.ipynb

Set data.yaml to the authors’ split

Run training cell, then validation on the test split

Collect metrics (mAP@0.5 primary; also mAP@0.5:0.95)

Build balanced splits

Run ClinicalGram_4Class_YOLOv8x_InstanceAware_Balanced_Splits.ipynb

Check printed per-class instance counts for train/val/test

Verify data.yaml written under the balanced split folder

Balanced training

Open ClinicalGram_YOLOv8x_BalancedSplit_AllAugs_EndToEnd.ipynb

Train; results saved to .../results/YOLOv8x/

Use the plotting cell to visualize precision/recall/mAP curves

Error analysis & figures

Use ClinicalGram_YOLOv8x_Prediction-GT_Overlay_ErrorAnalysis.ipynb to export overlays with dashed GT and misclassification marks.

Use Paper_Figures_Augmentation_Gallery_Microscopy.ipynb to generate the augmentation gallery.


Optimizer AdamW, lr0=1.5e-3, cosine LR (lrf=0.1)

Warmup 5 epochs; mosaic early, close at epoch 10

No mixup (small objects), mild HSV jitter

imgsz=640, batch 16 (A6000-class GPUs), early-stopping patience 50

