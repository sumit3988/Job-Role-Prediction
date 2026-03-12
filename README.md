# 🎓 Predicting Job Roles from Educational Background

An end-to-end Machine Learning project that predicts a person's likely job role based on their educational background, skills, and experience — complete with a beautiful interactive Streamlit web app.

---

## 🗂 Project Structure

```
job_role_prediction_project/
│
├── dataset.csv                # Synthetic dataset (1200 rows)
├── generate_dataset.py        # Script to regenerate the dataset
├── data_preprocessing.py      # Data cleaning, encoding, splitting
├── train_model.py             # Model training, evaluation & plots
├── model.pkl                  # Saved best model (auto-generated)
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
│
├── artifacts/                 # Auto-generated encoder artifacts
│   ├── cat_encoders.pkl
│   ├── target_encoder.pkl
│   └── feature_names.pkl
│
└── plots/                     # Auto-generated visualizations
    ├── job_role_distribution.png
    ├── feature_importance.png
    ├── model_comparison.png
    └── confusion_matrix.png
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project folder
```bash
cd job_role_prediction_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python generate_dataset.py
```

### 4. Train models & generate visualizations
```bash
python train_model.py
```

### 5. Launch the web app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📊 Dataset

| Column | Description | Type |
|---|---|---|
| `Degree` | Highest degree (B.Tech, BCA, MCA …) | Categorical |
| `Major` | Field of study | Categorical |
| `Skills` | Comma-separated skills (Python, Java …) | Multi-value |
| `Certifications` | Cloud/tech certification or None | Categorical |
| `GPA` | GPA out of 10 | Numeric |
| `Internship_Experience` | Number of internships (0–3) | Numeric |
| `Number_of_Projects` | Personal/academic projects | Numeric |
| `Job_Role` | **Target** – job role to predict | Categorical |

**Size:** 1200 rows · 8 columns · ~3% synthetic missing values

**Target Classes:**
- Data Scientist
- Backend Developer
- Frontend Developer
- DevOps Engineer
- Cloud Engineer
- AI Engineer
- Software Developer

---

## 🤖 ML Pipeline

### Preprocessing (`data_preprocessing.py`)
- Missing values: numeric → median, categorical → "Unknown"
- Skills column → **multi-hot encoding** (18 binary features)
- Degree / Major / Certifications → **Label Encoding**
- Train/Test Split: **80 / 20** (stratified)

### Models Trained (`train_model.py`)
| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model, max_iter=1000 |
| Decision Tree | max_depth=15 |
| Random Forest | 200 estimators, max_depth=20 |
| SVM | RBF kernel, C=10 |

### Evaluation Metrics
- Accuracy · Precision · Recall · F1-Score (weighted)

The **best model** (highest accuracy) is automatically saved as `model.pkl`.

---

## 📈 Visualizations

Four plots are saved to the `plots/` directory and are also viewable inside the web app:

| Plot | Description |
|---|---|
| `job_role_distribution.png` | Bar chart of target class distribution |
| `feature_importance.png` | Top 20 features from Random Forest |
| `model_comparison.png` | Grouped bar chart comparing all 4 models |
| `confusion_matrix.png` | Confusion matrix of the best model |

---

## 🌐 Web Application

The Streamlit app (`app.py`) provides:
- **Input form**: Degree, Major, Skills (multi-select), Certifications, GPA slider, Internship count, Project count
- **Prediction**: Predicted job role displayed instantly
- **Confidence chart**: Horizontal bar chart showing probability for each role
- **Visualization gallery**: All training plots in a tabbed expander

---

## ⚙️ Requirements

```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.0
matplotlib==3.8.2
seaborn==0.13.2
joblib==1.3.2
streamlit==1.31.0
```

Python **3.9+** recommended.

---

## 📖 How It Works

```
Educational Input
     │
     ▼
Feature Engineering ──► Multi-hot Skills + Label Encoding
     │
     ▼
ML Model (best of 4) ──► Predicted Job Role + Confidence
     │
     ▼
Streamlit UI ──► Interactive web interface
```

---

## 🙌 Credits

Built with ❤️ using Python · scikit-learn · Streamlit · pandas · seaborn
