# 🧪 Experiment 03: Valid D-Fire Baseline (YOLOv8-Nano)

> **Date:** 19 May 2026
> **Objective:** Establish a mathematically valid, leak-free baseline using exclusively the pure D-Fire dataset, resolving previous architecture flaws (class mismatch, logical drop bugs, and lack of deterministic split seed).

---

## 💻 1. Hardware & Execution Environment
* **Architecture:** `YOLOv8-Nano` (Ultralytics)
* **Training GPU:** NVIDIA GeForce GTX 1050 (3GB VRAM)
* **Parameters:** `50 Epochs` | `Batch Size: 16` | `Resolution: 640x640`
* **Reproducibility:** Enabled via `random.seed(42)` in the pipeline split.

---

## 📊 2. Dataset Distribution (Stratified Split)
By fixing the logic bug in `master_splitter.py` that previously deleted dual-label (Fire+Smoke) frames, the fire sample size increased by ~5x without structural augmentation. "Normal" class was replaced with empty annotation files (background negatives).

* 🔥 **Fire-Only Samples:** 1,168
* 💨 **Smoke-Only Samples:** 4,666
* 🌋 **Dual-Label (Fire + Smoke) Samples:** 4,653
* ☁️ **Background (Negative Samples):** 9,838
* **Total Valid Images:** 20,325

| Split | Image Count | Percentage |
| :--- | :---: | :---: |
| **Train** | 16,258 | 80% |
| **Validation** | 2,030 | 10% |
| **Test** | 2,037 | 10% |

> **Dataset Variance Note:** While the original Pedro et al. 2022 dataset reports 5,867 smoke-only images, the Roboflow v2 version utilized in this pipeline is distributed with a total of 20,325 images. As confirmed by the dataset source, no image augmentation techniques were applied. This discrepancy in total counts is attributed to Roboflow's server-side preprocessing and deduplication handling. Therefore, a direct 1:1 numerical comparison with the original Pedro et al. manuscript should be avoided.

---

## 📈 3. Quantitative Evaluation 

### 3.1. Training-Time Monitoring (Validation Set)
*Used strictly for loss calculation and hyperparameter tuning during the training phase.*

| Target Class | Images | Instances | Precision (P) | Recall (R) | mAP@.5 | mAP@.5-.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | 2,030 | 2,507 | 0.760 | 0.701 | **0.763** | 0.436 |
| **Fire** | 581 | 1,464 | 0.706 | 0.625 | **0.694** | 0.366 |
| **Smoke** | 931 | 1043 | 0.815 | 0.777 | **0.832** | 0.506 |

### 3.2. Unseen Data Evaluation (Test Set)
*The true baseline performance evaluated on the strictly isolated test partition (`model.val(split='test')`).*

| Target Class | Images | Instances | mAP@.5 | 
| :--- | :---: | :---: | :---: | 
| **All Classes** | 2,037 | 2,471 | **0.768** | 

---

## 🧠 4. Core Engineering Insights

1. **Fire Recall Boost:** Recovering the 4,653 mixed images fundamentally trained the bounding box regression network, lifting Fire mAP from the previous ~31% to a robust **69.4%** (Validation) and an overall dataset accuracy of **76.8%** (Test).
2. **Edge Deployability & Hardware Profiling:** The model clocks **9.3 ms inference time per image** on the local training compute (NVIDIA GTX 1050). To validate real-world edge deployment, the model was compiled into a TensorRT engine (`best.engine`) and profiled on the target edge hardware.

**Deployment Performance Comparison:**

| Metric | Development Compute (GTX 1050) | Target Edge Device (Jetson Orin Nano) |
| :--- | :--- | :--- |
| **Model Format** | ONNX (`best.onnx`) | TensorRT Engine (`best.engine`) |
| **Precision** | FP32 | FP16 |
| **Inference Latency** | 9.3 ms | **3.72 ms** |

This hardware optimization successfully satisfies and exceeds the real-time threshold required for edge environments.