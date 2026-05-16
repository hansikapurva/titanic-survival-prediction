# 🚢 Titanic Survival Prediction
**NITTE University · Team 1NT23CB016 / 020 / 035**
Hansika Purva · Inaya Firdous · Nandini

---

## What this project does
A full-stack machine learning web app that:
- Trains a **Random Forest** + **XGBoost** model on the Titanic dataset
- Lets users **predict survival** by entering passenger details
- Shows **EDA charts** (survival by gender, class, age distribution)
- Shows **model metrics** (accuracy, confusion matrix, feature importance)
- Supports **batch prediction** via JSON input

---

## Project structure
```
titanic_project/
├── app.py                  ← Flask backend + ML model
├── requirements.txt        ← Python dependencies
├── README.md
├── templates/
│   └── index.html          ← Main HTML page
└── static/
    ├── css/
    │   └── style.css       ← All styles
    └── js/
        └── main.js         ← Charts + API calls + interactions
```

---

## Setup & Run (3 steps)

### Step 1 — Install Python dependencies
```bash
cd titanic_project
pip install -r requirements.txt
```
> If you get errors with catboost/lightgbm, just remove those lines from requirements.txt — the core app only needs flask, flask-cors, pandas, numpy, scikit-learn, xgboost.

### Step 2 — Run the Flask server
```bash
python app.py
```
You'll see:
```
 * Running on http://127.0.0.1:5000
```

### Step 3 — Open in browser
Go to: **http://localhost:5000**

---

## Using your own titanic.csv
The app generates a realistic synthetic dataset by default. To use the **real Titanic dataset**:

1. Download `train.csv` from https://www.kaggle.com/c/titanic/data
2. Place it in the `titanic_project/` folder as `titanic.csv`
3. In `app.py`, replace the `train_model()` function's data generation block with:
```python
df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
```

---

## API Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main web app |
| GET | `/api/stats` | Model accuracy, EDA data, feature importance |
| POST | `/api/predict` | Single passenger prediction |
| POST | `/api/batch_predict` | Multiple passenger predictions |
| GET | `/api/sample_passengers` | 8 sample Titanic passengers |

### Example: POST /api/predict
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
Response:
```json
{
  "survived": true,
  "probability": 87.3,
  "not_survived_prob": 12.7
}
```

### Example: POST /api/batch_predict
```json
{
  "passengers": [
    { "Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C" },
    { "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 0, "Parch": 0, "Fare": 7.25, "Embarked": "S" }
  ]
}
```

---

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, Flask-CORS |
| ML Models | Random Forest, XGBoost (scikit-learn) |
| Feature Engineering | Pandas, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4 |
| Fonts | DM Serif Display + DM Sans (Google Fonts) |

---

## Model details
- **Algorithm**: Random Forest (1200 estimators, max_depth=10)
- **Comparison model**: XGBoost (800 estimators, learning_rate=0.03)
- **Features**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone, AgeBin, FareBin, Title
- **Train/Test split**: 80/20
- **Accuracy**: ~80% (Random Forest), ~82% (XGBoost)
