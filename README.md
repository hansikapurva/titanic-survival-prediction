# 🚢 Titanic Survival Prediction


## Overview

A full-stack machine learning web application that predicts whether a Titanic passenger would have survived, based on their details. Built with Flask (Python) on the backend and vanilla JavaScript + Chart.js on the frontend.

**What it does:**
- Trains a Random Forest + XGBoost ensemble model on the Titanic dataset
- Lets users predict survival by entering passenger details (class, sex, age, fare, etc.)
- Shows EDA charts — survival by gender, class, age distribution
- Shows model metrics — accuracy, precision, recall, feature importance
- Supports batch prediction via JSON API


## Project Structure

```
titanic-survival-prediction/
├── app.py                  ← Flask backend + ML model training
├── requirements.txt        ← Python dependencies
├── README.md               ← You are here
├── templates/
│   └── index.html          ← Main HTML page (served by Flask)
└── static/
    ├── css/
    │   └── style.css       ← All styles
    └── js/
        └── main.js         ← Charts, API calls, interactions
```

---

## Setup & Run

### Step 1 — Clone the repo

```bash
git clone https://github.com/hansikapurva/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you get errors with optional packages, the core app only needs:
> `flask flask-cors pandas numpy scikit-learn xgboost`

### Step 3 — Run the Flask server

```bash
python app.py
```

You will see:
```
 * Running on http://127.0.0.1:5000
```

### Step 4 — Open in browser

```
http://localhost:5000
```

---

## Requirements

`requirements.txt`:
```
flask
flask-cors
pandas
numpy
scikit-learn
xgboost
```

---

## Using the Real Titanic Dataset

The app uses a realistic synthetic dataset by default. To use the **real Titanic dataset**:

1. Download `train.csv` from [Kaggle](https://www.kaggle.com/c/titanic/data)
2. Place it in the project root as `titanic.csv`
3. In `app.py`, replace the synthetic data block in `train_model()` with:

```python
df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
```

---

## API Reference

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main web app |
| GET | `/api/stats` | Model accuracy, EDA data, feature importance |
| POST | `/api/predict` | Single passenger prediction |
| POST | `/api/batch_predict` | Multiple passenger predictions |
| GET | `/api/sample_passengers` | 8 sample Titanic passengers |

### POST `/api/predict`

**Request:**
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 28,
  "fare": 71.5,
  "sibsp": 0,
  "parch": 0,
  "embarked": "C"
}
```

**Response:**
```json
{
  "survived": true,
  "probability": 87.3,
  "not_survived_prob": 12.7
}
```

### POST `/api/batch_predict`

**Request:**
```json
{
  "passengers": [
    { "Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C" },
    { "Pclass": 3, "Sex": "male",   "Age": 22, "SibSp": 0, "Parch": 0, "Fare": 7.25,  "Embarked": "S" }
  ]
}
```

**Response:**
```json
[
  { "passenger": 1, "survived": true,  "probability": 89.1 },
  { "passenger": 2, "survived": false, "probability": 14.2 }
]
```

---

## Model Details

| Property | Value |
|----------|-------|
| Primary model | Random Forest |
| Comparison model | XGBoost |
| Estimators (RF) | 1200 trees, max_depth = 10 |
| Estimators (XGB) | 800 trees, learning_rate = 0.03 |
| Train / Test split | 80% / 20% |
| Accuracy (RF) | ~80% |
| Accuracy (XGB) | ~82% |

**Features used:**

| Feature | Description |
|---------|-------------|
| Pclass | Ticket class (1st / 2nd / 3rd) |
| Sex | Passenger sex |
| Age | Age in years |
| SibSp | Siblings / spouses aboard |
| Parch | Parents / children aboard |
| Fare | Ticket fare (£) |
| Embarked | Port of embarkation (S / C / Q) |
| FamilySize | SibSp + Parch + 1 |
| IsAlone | 1 if travelling alone |
| AgeBin | Age grouped into 5 bins |
| FareBin | Fare grouped into 5 quantiles |
| Title | Extracted from passenger name (Mr / Mrs / Miss / Master etc.) |

**Key survival insights from data:**
- Females survived at ~74%, males at ~19%
- 1st class passengers survived at ~63%, 3rd class at ~24%
- Children under 16 had higher survival rates
- Passengers from Cherbourg (C) had slightly better odds

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3, Flask, Flask-CORS |
| ML Models | Random Forest, XGBoost (scikit-learn) |
| Feature Engineering | Pandas, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4 |
| Fonts | Playfair Display + DM Sans (Google Fonts) |

---

## Screenshots

> Run the app at `http://localhost:5000` to see:
> - Stats strip (total passengers, survived, survival rate)
> - Model accuracy strip (accuracy, precision, recall)
> - Passenger details form with live prediction
> - EDA charts (overall survival, by gender, by class, age distribution)
> - Feature importance bar chart

---

## License

This project was built for academic purposes at NITTE University. Free to use and modify for educational projects.
