# Automated Fingerprint Pattern Classification Using a CNN
**Parul University (NAAC A++) · Image Processing · Deep Learning · Biometrics**

> **Authors:** Abhishek Dave (2403031077002), Rudra Pavasiya (2403031077018), Aditya Raj (2303031070002), Krutarth Valera (2303031070009)  
> **Faculty Mentors:** Prof. Anuradha Gharghe, Prof. Archana Dubey, Prof. Saurav Bhardwaj

---

## 📌 Executive Summary
This application implements an **end-to-end Automated Fingerprint Pattern Classification System** based on deep Convolutional Neural Networks (CNN) to categorize fingerprints into the **3 core biometric pattern classes**:
1. **Arch** (`arch`): Ridges enter from one side, rise in the center, and exit the opposite side without core or delta.
2. **Loop** (`loop`): Ridges enter and exit from the same side, curving back around a central core with exactly one delta.
3. **Whorl** (`whorl`): Concentric circular or spiral ridges around a central core with two deltas.

The system replaces manual, slow, observer-dependent human review with a fully automated, reproducible learning pipeline capable of rigorous held-out test scoring, near-miss borderline diagnosis, and step-by-step visual preprocessing inspection.

---

## 🏗️ Architecture & Pipeline Overview

```
Raw Fingerprint Scan
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Preprocessing Pipeline (Slide 7)              │
│ 1. Grayscale Conversion & Standard Resizing (128×128)  │
│ 2. Noise Reduction (Gaussian / Bilateral Filtering)    │
│ 3. Ridge Contrast Enhancement (CLAHE)                  │
│ 4. Fingerprint Region Segmentation (Otsu + Morphology) │
│ 5. Pixel Intensity Normalization [0.0, 1.0]            │
│ 6. Ridge Orientation & Vector Gradient Field Extraction│
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: CNN Feature Extractor & Classifier (Slide 8) │
│ • Stage 1: Conv2D(32) + BatchNorm + ReLU + MaxPool     │
│ • Stage 2: Conv2D(64) + BatchNorm + ReLU + MaxPool     │
│ • Stage 3: Conv2D(128) + BatchNorm + ReLU + MaxPool    │
│ • Stage 4: Conv2D(128) + BatchNorm + AdaptiveAvgPool   │
│ • Head: Flatten (2048) -> Dense(128) -> Dropout(0.35)  │
│         -> Dense(3) + Softmax Output                   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Rigorous Evaluation & Near-Miss (Slide 11-14) │
│ • Accuracy, Precision, Recall, F1 (Macro & Per-Class)  │
│ • 3×3 Confusion Matrix Breakdown                       │
│ • Borderline / Near-Miss Ambiguity Detection (Margin)  │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Demonstration & Deployment (Slide 13)         │
│ • Modern Interactive Dark-Themed Web Application       │
│ • REST API with OpenAPI Swagger docs                   │
│ • CLI subcommands for dataset gen, train, eval, predict│
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Installation
Ensure Python 3.10+ is available. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
Execute the comprehensive test suite (100% pass rate across 43 tests):
```bash
pytest -v
```

### 3. Launch the Interactive Web Application
Start the FastAPI server:
```bash
python -m src.cli serve --host 127.0.0.1 --port 8000
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.
Interactive features available:
- **Live Classifier:** Drag-and-drop any scan or click sample presets (Arch, Loop, Whorl).
- **Preprocessing Stages Visualizer:** Inspect Stage 1 (Input) through Stage 6 (Orientation Field).
- **Probability Distribution:** Real-time softmax probability bars and confidence gauge.
- **Near-Miss Detector:** Alerts when the margin between top-2 predictions is borderline.
- **Evaluation Dashboard:** Interactive confusion matrix and per-class metrics.
- **REST API Documentation:** Available at `http://127.0.0.1:8000/docs`.

---

## 🛠️ Command-Line Interface (CLI)

The package provides a full CLI via `python -m src.cli`:

```bash
# Auto-label unlabelled scans from Kaggle SOCOFing
python auto_label.py --input "C:\Users\Asus\Downloads\archive\SOCOFing\Real" --output data --offset 0 --max-images 600

# Inspect the 6-stage preprocessing pipeline on a single image
python -m src.cli preprocess --input data/arch/arch_00000.bmp --output output/preprocessed

# Train the CNN with stratified splits, class weights, and early stopping on real data
python -m src.cli train --data data --output models/best_fingerprint_cnn.pth --epochs 25 --lr 0.0008 --report-json output/real_eval_report.json

# Evaluate on a dataset and print confusion matrix & metrics
python -m src.cli evaluate --model models/best_fingerprint_cnn.pth --data data --output-json output/real_eval_report.json

# Classify a single fingerprint image
python -m src.cli predict --image data/whorl/whorl_00000.bmp --model models/best_fingerprint_cnn.pth
```

---

## 📊 Rigorous Evaluation Metrics (Slides 11, 12, 14)

On held-out test splits, the system evaluates:
1. **Accuracy**: Fraction of test prints classified correctly.
2. **Precision**: $\frac{TP}{TP + FP}$ (Macro and per-class for Arch, Loop, Whorl).
3. **Recall**: $\frac{TP}{TP + FN}$ (Macro and per-class for Arch, Loop, Whorl).
4. **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
5. **Confusion Matrix**: Full 3×3 matrix identifying systematic inter-class misclassifications.
6. **Near-Miss Borderline Detection**: Identifies instances where $\text{Prob}_1 - \text{Prob}_2 < 0.20$ to surface difficult Loop vs. Whorl or Arch vs. Loop boundary ambiguities.

---

## 🛡️ Risk & Mitigation Strategies (Slide 14)

| Challenge | Engineering Mitigation Implemented |
|---|---|
| **Limited / Imbalanced Data** | Data augmentation (`BiometricAugmentation` with rotation, translation, contrast) + inverse class-weighted `CrossEntropyLoss`. |
| **Similar Inter-Class Patterns** | Confusion matrix error tracking & automated near-miss margin alerts. |
| **Variable Scan Quality** | CLAHE adaptive contrast equalization & Otsu morphological segmentation. |
| **Overfitting on Small Data** | Dropout (0.35), batch normalization, and validation early stopping with patience. |
| **Compute Cost** | Compact 4-stage convolutional feature extractor running smoothly on CPU/GPU. |

---

## 📁 Repository Structure
```
fingerprint_pattern_workflow/
├── src/
│   ├── constants.py       # Classes, aliases, anatomical metadata, thresholds
│   ├── preprocessing.py   # Grayscale, resize, noise filter, CLAHE, Otsu, orientation field
│   ├── dataset.py         # Sherlock-Monro generator, augmentation, dataset, stratification
│   ├── model.py           # Deep CNN architecture (Slide 8)
│   ├── trainer.py         # Training loop, early stopping, test evaluation
│   ├── evaluation.py      # Metrics suite, confusion matrix, near-miss analysis
│   ├── inference.py       # Production inference service & visualization exporter
│   ├── api.py             # FastAPI REST service & UI host
│   └── cli.py             # Command-line interface with subcommands
├── static/
│   ├── index.html         # Modern web application UI
│   ├── styles.css         # Dark theme biometric stylesheet
│   └── app.js             # Frontend reactive controller
├── models/
│   └── best_fingerprint_cnn.pth # Trained model checkpoint
├── data/                  # 1,200 categorized real SOCOFing prints (arch, loop, whorl)
├── output/                # Real evaluation report (83.70% test accuracy)
├── tests/
│   ├── test_preprocessing.py # Preprocessing unit tests
│   ├── test_dataset.py       # Dataset & augmentation unit tests
│   ├── test_model.py         # CNN architecture & checkpoint tests
│   ├── test_trainer.py       # Training loop tests
│   ├── test_evaluation.py    # Metric & confusion matrix tests
│   ├── test_inference.py     # Inference service tests
│   ├── test_api.py           # REST API endpoint tests
│   └── test_e2e.py           # Full 5-phase end-to-end integration test
├── pytest.ini             # Pytest configuration
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── WORK_SUMMARY.md        # Session summary & resumption documentation
```
