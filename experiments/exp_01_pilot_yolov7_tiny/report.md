# 🧪 Experiment 01: YOLOv7-Tiny Pilot Training

> **Date:** 19 April 2026
> **Objective:** Establish a baseline model performance using a cleaned dataset and hardware-constrained parameters (GTX 1050 3GB VRAM).

---

## 📊 1. Dataset Overview
This experiment was conducted using an optimized version of the combined **D-Fire** and **ForestFire** datasets. Perceptual hashing was utilized to eliminate redundant frames and prevent model overfitting.

| Metric | Count | Notes |
| :--- | :--- | :--- |
| **Original Images** | ~24,000 | Raw data with high redundancy |
| **Duplicates Removed** | 8,531 | Purged via P-Hash algorithm |
| **Final Dataset Size** | 8,758 | Cleaned, unique, and balanced |

**Class Distribution:**
* ☁️ **Normal:** `3,900` images
* 💨 **Smoke:** `3,638` images
* 🔥 **Fire:** `1,220` images *(Critical target for future augmentation)*

---

## 💻 2. Hardware & Environment
* **GPU:** NVIDIA GeForce GTX 1050 (3GB VRAM)
* **Compute Engine:** CUDA 11.8
* **Environment:** Python 3.9 (Conda) | PyTorch 2.4.1

---

## ⚙️ 3. Training Configuration
* **Architecture:** `YOLOv7-Tiny`
* **Epochs:** `50` | **Batch Size:** `16` | **Resolution:** `640x640`
* **Optimizer:** `SGD` | **Learning Rate (lr0):** `0.01`

## 📈 4. Performance Summary (mAP@.5)
* Class | mAP@.5 Score | Evaluation
* 🎯 Overall |45.8% | Solid baseline for 50 epochs
* 💨 Smoke | 60.4% | Good feature extraction
* 🔥 Fire |31.2% | Weak (Due to class imbalance)

**Execution Command:**
```bash
python train.py --workers 2 --device 0 --batch-size 16 --data "data.yaml" --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weights yolov7-tiny.pt --name exp_01_pilot --hyp data/hyp.scratch.tiny.yaml --epochs 50


