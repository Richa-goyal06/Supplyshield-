# 🚢 Supply Chain Disruption Prediction System

> A machine learning system that predicts supply chain disruptions before they happen — helping logistics teams proactively manage risk across global shipping routes.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)


---

## Overview

Supply chain disruptions cost the global economy trillions annually. This project builds a **binary classification system** to predict whether a shipment will face a disruption, based on geopolitical risk, weather events, carrier reliability, and logistical factors.

Trained on **5,000 historical shipment records** spanning multiple global ports and transport modes, the final deployed model achieves a **ROC-AUC of 0.817** using Gradient Boosting.

---

## Dataset

| Property | Value |
|---|---|
| Records | 5,000 shipments |
| Features | 14 columns (raw) |
| Target | `Disruption_Occurred` (binary) |
| Transport Modes | Sea, Air, Road, Rail |
| Ports Covered | Shanghai, Dubai, Rotterdam, Busan, Singapore, Hamburg, and more |

**Key numerical features:**

| Feature | Mean | Median |
|---|---|---|
| Distance (km) | 7,704 | 7,750 |
| Weight (MT) | 246 | 243 |
| Fuel Price Index | 2.85 | 2.84 |
| Geopolitical Risk Score | 5.08 | 5.10 |
| Carrier Reliability Score | 0.75 | 0.76 |
| Lead Time (Days) | 19.4 | 8.25 |

> **Note:** Lead Time is highly right-skewed (mean >> median), indicating occasional extreme delays — a strong signal for disruption risk.

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of all numerical features
- Correlation heatmap to identify multicollinearity
- Key finding: `Lead_Time_Days` has the strongest correlation with `Disruption_Occurred` (r = 0.27), followed by `Geopolitical_Risk_Score` (r = 0.23)

### 2. Feature Engineering
Custom interaction and derived features created to boost signal:

```python
df['Risk_x_Fuel']   = df['Geopolitical_Risk_Score'] * df['Fuel_Price_Index']
df['Weight_per_km'] = df['Weight_MT'] / df['Distance_km']
df['Carrier_Unreliability'] = 1 - df['Carrier_Reliability_Score']
```

Temporal features extracted from shipment date:
```python
df['Month']     = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter']   = df['Date'].dt.quarter
```

### 3. Feature Selection — Mutual Information

Mutual Information scores were computed to rank feature relevance independent of model assumptions:

| Rank | Feature | MI Score |
|---|---|---|
| 1 | Weather_Condition_Hurricane | 0.122 |
| 2 | Lead_Time_Days | 0.055 |
| 3 | Geopolitical_Risk_Score | 0.020 |
| 4 | Origin_Port_Shanghai | 0.018 |
| 5 | Weather_Condition_Storm | 0.015 |

> Hurricane and storm weather conditions dominate both Mutual Information and Random Forest importance rankings — confirming extreme weather as the primary disruption driver.

### 4. Model Training & Comparison

Three baseline classifiers were trained and evaluated using stratified 80/20 train-test split with 5-fold cross-validation:

| Model | ROC-AUC | F1 (Disruption) | F1 (No Disruption) |
|---|---|---|---|
| **Logistic Regression** | 0.829 | 0.758 | 0.710 |
| **Gradient Boosting**  | 0.817 | 0.775 | 0.641 |
| **Random Forest** | 0.806 | 0.770 | 0.607 |

**Gradient Boosting** was selected as the final model for its superior F1 score on the disruption class — the class that matters most for business impact — and its robustness to overfitting.

---

## Top Predictive Features

From the Random Forest importance analysis, the most influential features are:

```
Weather_Condition_Hurricane   ████████████  0.119
Geopolitical_Risk_Score       ███████████   0.095
Lead_Time_Days                ███████████   0.093
Risk_x_Fuel (engineered)      ███████        0.060
Distance_km                   ██████         0.056
Weight_MT                     ██████         0.055
```

---

## Project Structure

```
supply-chain-disruption/
│
├── data/
│   └── shipments.csv              # Historical shipment records
│
├── notebooks/
│   └── analysis.ipynb             # EDA, feature engineering, model training
│
├── models/
│   └── gradient_boosting.pkl      # Serialized final model
│
├── app/
│   └── streamlit_app.py           # Streamlit deployment interface
│
├── requirements.txt
└── README.md
```

---


### Prerequisites

```bash
python >= 3.9
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/supply-chain-disruption.git
cd supply-chain-disruption

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will launch at `http://localhost:8501` where you can input shipment details and get real-time disruption probability predictions.

---

## Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
streamlit>=1.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 📈 Model Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
```

---

## 🔭 Future Scope

| Feature | Description | Priority |
|---|---|---|
| 🌦️ Real-time Weather APIs | Integrate live weather feeds (OpenWeatherMap, NOAA) to enrich predictions with current conditions | High |
| 📡 Live Risk Dashboard | Streaming dashboard tracking geopolitical risk scores by region in real time | High |
| 🔔 Alert System | Email/Slack notifications when disruption probability exceeds a configurable threshold | Medium |
| 🗺️ Route Optimization | Recommend alternative routes when high disruption risk is detected | Medium |
| 🔄 Model Retraining Pipeline | Automated weekly retraining with new shipment data using MLflow tracking | Low |
| 🌐 Multi-modal Risk Scoring | Incorporate news sentiment analysis via NLP for geopolitical risk estimation | Low |

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

- LinkedIn:  linkedin.com/in/richa-goyal-248444298)
