# RUN — Unbalanced Baseline (YOLOv8x)

**Date:** 2025-08-12  
**Commit:** fc6cca5  
**Notebook:** notebooks/ClinicalGram_4Class_YOLOv8x_Unbalanced_Baseline.ipynb  
**Data:** data/data.yaml (authors’ original 7:2:1 split; unbalanced)  
**Hardware:** 1× A6000 (device=1) • imgsz=640 • batch=16

## Train config (key)
- epochs=50 • optimizer=AdamW • lr0=1.5e-3 • lrf=0.1 • weight_decay=5e-4
- warmup_epochs=5 • warmup_bias_lr=0.1
- mosaic early (close_mosaic=10) • mixup=0
- hsv_h=0.015 • hsv_s=0.7 • hsv_v=0.4
- degrees=0 • translate=0.05 • scale=0.7 • shear=0
- fliplr=0.5 • flipud=0.0
- patience=50

## Eval settings
- split=test • conf=0.22 • iou=0.70 • half=True

## Final test metrics (Ultralytics)
- **mAP@0.5:** 0.895  
- **mAP@0.5:0.95:** 0.634  
- **Precision (P):** 0.799  
- **Recall (R):** 0.868

_Per-class mAP@0.5:_  
- negative_cocci: **0.902**  
- positive_cocci: **0.905**  
- negative_bacilli: **0.928**  
- positive_bacilli: **0.843**

## Artifacts
- artifacts/results.csv
- artifacts/PR_curve.png
- artifacts/confusion_matrix.png
- artifacts/F1_curve.png

## Notes
- Trained **as-is** on unbalanced data (no rebalancing). This is the cited baseline.
