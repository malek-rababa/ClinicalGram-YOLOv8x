# RUN — Failure-Guided 10-Step (Balanced, AllAugs, YOLOv8x)

**Date:** 2025-08-12  
**Commit:** <fc6cca5>  
**Notebook:** notebooks/YOLOv8x_ClinicalGram_Balanced_AllAugs_Pipeline.ipynb  
**Data:** data/data.yaml  (balanced split)  
**Hardware:** 1× A6000 (device=1) • imgsz=640 • batch=16

## Train config (key)
- epochs=50 • optimizer=AdamW • lr0=1.5e-3 • lrf=0.1 • momentum=0.9 • weight_decay=5e-4  
- warmup_epochs=5 • warmup_bias_lr=0.1  
- mosaic=1.0 (close_mosaic=10) • mixup=0.0  
- hsv_h=0.015 • hsv_s=0.7 • hsv_v=0.4  
- degrees=0 • translate=0.05 • scale=0.7 • shear=0  
- fliplr=0.5 • flipud=0.0  
- patience=50

## Eval settings
- split=test • conf=0.20 • iou=0.45 • half=True

## Final test metrics (Ultralytics)
- mAP@0.5: **0.939**
- mAP@0.5:0.95: **0.794**
- Precision: **0.934**
- Recall: **0.900**

*(Per-class mAP@0.5 — neg_cocci 0.933, pos_cocci 0.960, neg_bacilli 0.910, pos_bacilli 0.953)*

## Artifacts
- artifacts/results.csv
- artifacts/PR_curve.png
- artifacts/confusion_matrix.png
- artifacts/F1_curve.png

## Notes
- Instance-balanced split; targeted augs: CLAHE, RandomBrightnessContrast, Elastic, strong GaussianBlur, moderate Sharpen.
- These are the **main balanced** results for the paper.
