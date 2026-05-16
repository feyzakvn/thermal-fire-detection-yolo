# 🧪 Experiment 02: YOLOv8-Nano Pilot Training & Comparison

> **Date:** 16 May 2026
> **Objective:** Evaluate YOLOv8-Nano performance against the YOLOv7-Tiny baseline using the identical dataset (8,758 images) and parameter constraints.

---

## 💻 1. Hardware & Execution
* **Architecture:** `YOLOv8-Nano` (Ultralytics)
* **GPU:** NVIDIA GeForce GTX 1050 (3GB VRAM)
* **Epochs:** `50` | **Batch Size:** `16` | **Resolution:** `640x640`

---

## 📊 2. Performance Comparison (v8 vs v7)

| Metric | YOLOv7-Tiny (Exp 01) | YOLOv8-Nano (Exp 02) | Improvement |
| :--- | :---: | :---: | :--- |
| **Model Size** | ~6.0M Params | **3.0M Params** | 50% Lighter |
| **VRAM Usage** | ~2.58 GB | **2.15 GB** | More Efficient |
| **Overall mAP@.5** | 45.8% | **~68.5%** | **+22.7% Jump** |
| **Smoke mAP@.5** | 60.4% | **~75.0%** | Stronger Feature Extraction |
| **Fire mAP@.5** | 31.2% | **~42.0%** | Still throttled by data scarcity |

---

## 🔍 3. Real-World Inference Observations
Based on independent visual tests (2 fire videos, 2 fire photos, 2 normal photos):
1. **Low-Quality Images:** Both models completely missed hidden fire/smoke in heavily blurred or low-contrast images. *Solution: Implement CLAHE video enhancement (Step 3).*
2. **False Positives:** YOLOv8-Nano is highly sensitive and occasionally misclassified dense clouds as fire/smoke. *Solution: Adjust confidence thresholds.*
3. **Class Imbalance:** The confusion matrix confirms 91 ground-truth Fire instances were missed (classified as background). *Solution: Synthetic data augmentation for the Fire class.*