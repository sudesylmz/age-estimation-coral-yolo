# Age Estimation with CORAL Ordinal Regression  
### YOLO-Filtered UTKFace • Nested Multi-Task Learning (Gender → Race → Age)

This project focuses on **robust age estimation from face images** using **CORAL (Consistent Rank Logits)** based ordinal regression.  
Instead of treating age as a simple regression problem, the model explicitly captures the **ordinal nature of age** and enforces **monotonic ranking constraints**.

To improve data quality and reduce label noise, **YOLO-based face detection** is used to filter and optionally crop faces before training.

---

## Key Features

- CORAL Ordinal Regression for age estimation  
- YOLO-based face filtering and optional face cropping  
- Nested multi-task learning:
  - Gender → Race → Age (conditioned)
- Age-bin importance weighting for imbalanced data  
- ConvNeXt backbone (via timm)  
- Detailed error, bias, and age-group analysis  
- End-to-end PyTorch training pipeline  

---

## Dataset

### UTKFace Dataset


- **Age:** 1–90  
- **Gender:**  
  - 0 → Male  
  - 1 → Female  
- **Race:**  
  - 0 → White  
  - 1 → Black  
  - 2 → Asian  
  - 3 → Indian  
  - 4 → Other  

---

## YOLO-Based Data Filtering

Before training, every image is processed by a **YOLO face detector**:

- Images **without a detectable face are discarded**
- Optionally, **face crops** are used instead of full images
- This significantly reduces background noise and mislabeled samples

Typical outcome:
- ~70–80% of UTKFace images retained
- Cleaner and more stable training data

---

## Model Architecture

### Nested Multi-Task Design


### Why Nested?

- Gender provides strong prior information for race
- Gender + race together improve age estimation
- Encourages shared representations and better generalization

---

## CORAL Ordinal Regression

Instead of predicting a single age value:

- Define **K age thresholds**
- Each threshold answers:  
  **“Is age ≥ t ?”**
- Model outputs **K binary logits**
- A single shared logit + threshold-specific biases guarantees:

P(age ≥ t₁) ≥ P(age ≥ t₂) for t₁ < t₂


### Threshold Configuration

[2, 4, 6, ..., 90] → 45 thresholds


### Age Prediction

Expected Rank = Σ sigmoid(logit_k)
Predicted Age = linear interpolation between AGE_MIN and AGE_MAX


---

## Loss Functions

### Age (Main Task)
- CORAL loss (binary cross-entropy per threshold)
- Age-bin importance weighting to handle imbalance

### Auxiliary Tasks
- Gender classification (CrossEntropy + label smoothing)
- Race classification (CrossEntropy + label smoothing)

### Total Loss

L = 1.5 × Age_CORAL + 0.3 × Gender + 0.2 × Race

---

## Training Setup

- Optimizer: **AdamW**
- Learning Rate: **1e-4**
- Scheduler: **Cosine Annealing**
- Batch Size: **64**
- Epochs: **30**
- Early Stopping: **Patience = 7**

---

## Evaluation Metrics

- **Age MAE (years)** ← primary metric  
- Gender Accuracy  
- Race Accuracy  
- Age-bin-wise MAE  
- Error distribution  
- CORAL bias monotonicity check  

---

## Example Results

Test Results (Best Model)
Age MAE : ~4.4 years
Gender Accuracy : ~92%
Race Accuracy : ~78%


Performance improves significantly on underrepresented age groups due to ordinal modeling and importance weighting.

---

## Analysis & Visualizations

The project automatically generates:

- Training curves (loss, MAE, accuracy)
- Prediction vs ground truth scatter plots
- Error histograms
- Learned CORAL bias visualization
- Age-group-wise MAE comparison

Saved outputs:

training_curves_coral.png
yolo_filtering_stats.png
test_analysis_coral.png

---

## How to Run & Live Demo

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset by placing UTKFace images in the following directory:

data/utkface/UTKFace/*.jpg

Train the model using:

python train.py

Live Demo – Face Detection & Age-Gender Analysis (Gradio)

This project also includes a real-time interactive demo built with Gradio.
The application performs live face detection and age–gender analysis on webcam input, uploaded images, and uploaded video files.

The demo uses a YOLO-based face detector followed by a ConvNeXt-based nested multi-task model that predicts gender, race, and age (normalized to the range 1–90).

To run the Gradio application, install the required demo dependencies:

pip install gradio ultralytics timm opencv-python torch torchvision

Make sure the following model files are located in the project root directory:

face_yolo_best.pt
best_age_model_balanced.pth

Launch the application with:

python app.py

After launch, the demo will be available at:

http://localhost:7861

Note
The demo uses a lightweight inference-oriented model for real-time performance.
The CORAL ordinal regression model is used in the main training pipeline and evaluation experiments.
