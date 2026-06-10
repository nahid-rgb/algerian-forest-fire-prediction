# 🔥 Algerian Forest Fire — FWI Prediction

A machine learning web application that predicts the **Fire Weather Index (FWI)** based on weather and environmental conditions in Algeria. Built with Flask and deployed on Render.

🌐 **Live Demo:** [https://algerian-forest-fire-prediction-0z2k.onrender.com](https://algerian-forest-fire-prediction-0z2k.onrender.com)

---

## 📌 What is FWI?

The **Fire Weather Index (FWI)** is a numeric rating of fire intensity used by fire services to assess wildfire danger.

- Higher FWI → More dangerous fire conditions
- Range in this dataset: 0 to 31.1

---

## 📊 About the Dataset

The dataset covers two regions of Algeria and was collected from June to September 2012.

| Property | Details |
|----------|---------|
| Total Records | 244 (122 per region) |
| Regions | Bejaia (northeast) + Sidi-Bel Abbes (northwest) |
| Period | June – September 2012 |
| After cleaning | 243 usable records |
| Fire cases | 137 fire, 106 not fire |
| Target variable | FWI (Fire Weather Index) |

> From EDA: August had the most forest fires in both regions. Most fires occurred across June, July, and August.

---

## 📥 Input Features

The original dataset has 11 attributes. After removing `day`, `month`, `year` (not needed for prediction) and adding `Region`, the model uses these 9 features:

| Feature | Description | Range |
|---------|-------------|-------|
| **Temperature** | Max temperature at noon in Celsius | 22 – 42°C |
| **RH** | Relative Humidity in % | 21 – 90% |
| **Ws** | Wind Speed in km/h | 6 – 29 km/h |
| **Rain** | Total rainfall for the day in mm | 0 – 16.8 mm |
| **FFMC** | Fine Fuel Moisture Code — moisture of fine surface fuels. Higher = drier = higher fire risk | 28.6 – 92.5 |
| **DMC** | Duff Moisture Code — moisture of loosely compacted organic matter | 1.1 – 65.9 |
| **ISI** | Initial Spread Index — expected rate of fire spread | 0 – 18.5 |
| **Classes** | Whether fire occurred: 1 = Fire, 0 = Not Fire | 0 or 1 |
| **Region** | 0 = Bejaia (northeast), 1 = Sidi-Bel Abbes (northwest) | 0 or 1 |

> Note: Two FWI system features — `DC` (Drought Code) and `BUI` (Buildup Index) — were removed during preprocessing due to high multicollinearity with other features (correlation > 0.85).

---

## 📤 Output

| Output | Description |
|--------|-------------|
| **FWI** | Predicted Fire Weather Index (rounded to 2 decimal places) |

---

## 🤖 ML Pipeline

### 1. Data Cleaning
- Removed rows with null values
- Stripped whitespace from column names
- Converted columns to correct data types (int/float)
- Encoded `Classes`: `not fire` → 0, `fire` → 1
- Added `Region` column: Bejaia → 0, Sidi-Bel Abbes → 1

### 2. Feature Selection
Removed features with correlation > 0.85 to handle multicollinearity:
- Dropped: `DC` and `BUI`
- Remaining features: 9

### 3. Train/Test Split
- Split: 75% train / 25% test (`random_state=42`)
- Train: 182 samples, Test: 61 samples

### 4. Feature Scaling
Applied `StandardScaler` — fitted on training data, transformed both train and test sets.

### 5. Models Trained

| Model | Regularization | Cross Validation |
|-------|---------------|-----------------|
| Linear Regression | None | No |
| Lasso Regression | L1 | LassoCV (cv=5) |
| Ridge Regression ✅ | L2 | RidgeCV (cv=5) — best alpha = 1.0 |
| ElasticNet Regression | L1 + L2 | ElasticNetCV (cv=5) |

**Ridge Regression was selected** for deployment. It applies L2 regularization which handles multicollinearity well by penalizing large coefficients without eliminating features.

### 6. Model Saving
```python
import pickle
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(ridge, open('models/ridge.pkl', 'wb'))
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| Web Framework | Flask |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Production Server | Gunicorn |
| Deployment | Render.com |
| Version Control | Git + GitHub |

---

## 📁 Project Structure

```
algerian-forest-fire-prediction/
├── models/
│   ├── ridge.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── ridge_lasso_elastic.ipynb   ← EDA + Data Cleaning
│   └── model_training.ipynb        ← Feature Engineering + Model Training
├── templates/
│   └── home.html
├── application.py
├── Procfile
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/nahid-rgb/algerian-forest-fire-prediction.git
cd algerian-forest-fire-prediction
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
python application.py
```

Open browser at `http://127.0.0.1:5000`

---

## 🌐 Deployment

Deployed on **Render.com** (free tier) using Gunicorn as the production server.

- Live URL: [https://algerian-forest-fire-prediction-0z2k.onrender.com](https://algerian-forest-fire-prediction-0z2k.onrender.com)
---
