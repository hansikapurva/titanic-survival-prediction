from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import os

app = Flask(__name__)
CORS(app)

# ─── Global model + data ────────────────────────────────────────────────
model = None
feature_columns = None
model_stats = {}
sample_data = None

def engineer_features(df):
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,16,32,48,64,100], labels=[0,1,2,3,4]).astype(float)
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False, duplicates='drop')

    if 'Name' in df.columns:
        def extract_title(name):
            try:
                return name.split(',')[1].split('.')[0].strip()
            except:
                return 'Unknown'
        df['Title'] = df['Name'].apply(extract_title)
    else:
        df['Title'] = 'Unknown'

    cat_cols = ['Sex', 'Embarked', 'Title']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else 'Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def train_model():
    global model, feature_columns, model_stats, sample_data

    # Build synthetic-but-realistic Titanic-like dataset
    np.random.seed(42)
    n = 891

    pclass = np.random.choice([1,2,3], n, p=[0.24,0.21,0.55])
    sex = np.random.choice(['male','female'], n, p=[0.65,0.35])
    age = np.where(np.random.rand(n) < 0.2, np.nan,
                   np.clip(np.random.gamma(3, 10, n), 1, 80))
    sibsp = np.random.choice([0,1,2,3,4,5], n, p=[0.68,0.23,0.05,0.02,0.01,0.01])
    parch = np.random.choice([0,1,2,3,4], n, p=[0.76,0.13,0.08,0.02,0.01])
    fare = np.where(pclass==1,
                    np.random.gamma(5, 20, n),
                    np.where(pclass==2,
                             np.random.gamma(3, 7, n),
                             np.random.gamma(2, 4, n)))
    embarked = np.random.choice(['S','C','Q'], n, p=[0.72,0.19,0.09])

    # Survival probabilities matching historical data
    survived = np.zeros(n, dtype=int)
    for i in range(n):
        p = 0.20
        if sex[i] == 'female': p += 0.50
        if pclass[i] == 1: p += 0.20
        elif pclass[i] == 2: p += 0.08
        a = age[i] if not np.isnan(age[i]) else 28
        if a < 16: p += 0.15
        elif a > 60: p -= 0.10
        fs = sibsp[i] + parch[i] + 1
        if 2 <= fs <= 4: p += 0.05
        elif fs > 5: p -= 0.15
        if embarked[i] == 'C': p += 0.05
        survived[i] = 1 if np.random.rand() < np.clip(p, 0, 1) else 0

    df = pd.DataFrame({
        'Pclass': pclass, 'Sex': sex, 'Age': age,
        'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
        'Embarked': embarked, 'Survived': survived,
        'Name': ['Passenger, Mr. Test']*n
    })

    sample_data = df.drop('Survived', axis=1).head(10).copy()

    df_feat = engineer_features(df)
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',
                'FamilySize','IsAlone','AgeBin','FareBin','Title']
    features = [f for f in features if f in df_feat.columns]
    feature_columns = features

    X = df_feat[features]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=1200, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        class_weight='balanced', random_state=42
    )
    rf.fit(X_train, y_train)
    model = rf

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)
    importances = rf.feature_importances_.tolist()

    # EDA stats
    surv_rate = float(df['Survived'].mean())
    gender_surv = df.groupby('Sex')['Survived'].mean().to_dict()
    class_surv = df.groupby('Pclass')['Survived'].mean().to_dict()
    age_dist = df['Age'].dropna().tolist()
    fare_dist = df['Fare'].tolist()

    model_stats = {
        'accuracy': round(acc * 100, 2),
        'confusion_matrix': cm,
        'report': report,
        'feature_importances': [
            {'feature': f, 'importance': round(imp * 100, 2)}
            for f, imp in sorted(zip(features, importances),
                                 key=lambda x: x[1], reverse=True)
        ],
        'eda': {
            'survival_rate': round(surv_rate * 100, 2),
            'total': n,
            'survived': int(df['Survived'].sum()),
            'not_survived': int(n - df['Survived'].sum()),
            'gender_survival': {k: round(v*100, 2) for k,v in gender_surv.items()},
            'class_survival': {str(k): round(v*100, 2) for k,v in class_surv.items()},
            'age_bins': {
                'children': int((df['Age'].fillna(28) < 16).sum()),
                'young': int(((df['Age'].fillna(28) >= 16) & (df['Age'].fillna(28) < 32)).sum()),
                'adult': int(((df['Age'].fillna(28) >= 32) & (df['Age'].fillna(28) < 48)).sum()),
                'middle': int(((df['Age'].fillna(28) >= 48) & (df['Age'].fillna(28) < 64)).sum()),
                'senior': int((df['Age'].fillna(28) >= 64).sum()),
            }
        }
    }

train_model()


# ─── Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    return jsonify(model_stats)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        df = pd.DataFrame([{
            'Pclass': int(data.get('pclass', 3)),
            'Sex': data.get('sex', 'male'),
            'Age': float(data.get('age', 28)),
            'SibSp': int(data.get('sibsp', 0)),
            'Parch': int(data.get('parch', 0)),
            'Fare': float(data.get('fare', 15)),
            'Embarked': data.get('embarked', 'S'),
            'Name': 'Passenger, Mr. Test'
        }])
        df_feat = engineer_features(df)
        X = df_feat[feature_columns]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return jsonify({
            'survived': bool(pred),
            'probability': round(float(proba[1]) * 100, 1),
            'not_survived_prob': round(float(proba[0]) * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json.get('passengers', [])
    results = []
    for i, p in enumerate(data):
        try:
            df = pd.DataFrame([{
                'Pclass': int(p.get('Pclass', 3)),
                'Sex': p.get('Sex', 'male'),
                'Age': float(p.get('Age', 28)),
                'SibSp': int(p.get('SibSp', 0)),
                'Parch': int(p.get('Parch', 0)),
                'Fare': float(p.get('Fare', 15)),
                'Embarked': p.get('Embarked', 'S'),
                'Name': 'Passenger, Mr. Test'
            }])
            df_feat = engineer_features(df)
            X = df_feat[feature_columns]
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            results.append({
                'passenger': i+1,
                'survived': bool(pred),
                'probability': round(float(proba[1]) * 100, 1)
            })
        except Exception as e:
            results.append({'passenger': i+1, 'error': str(e)})
    return jsonify(results)


@app.route('/api/sample_passengers')
def sample_passengers():
    passengers = [
        {'Pclass':1,'Sex':'female','Age':38,'SibSp':1,'Parch':0,'Fare':71.28,'Embarked':'C','ActualSurvived':True},
        {'Pclass':3,'Sex':'male','Age':22,'SibSp':1,'Parch':0,'Fare':7.25,'Embarked':'S','ActualSurvived':False},
        {'Pclass':1,'Sex':'male','Age':35,'SibSp':0,'Parch':0,'Fare':53.1,'Embarked':'S','ActualSurvived':False},
        {'Pclass':3,'Sex':'female','Age':0.75,'SibSp':2,'Parch':1,'Fare':19.26,'Embarked':'S','ActualSurvived':True},
        {'Pclass':2,'Sex':'female','Age':27,'SibSp':0,'Parch':2,'Fare':11.13,'Embarked':'S','ActualSurvived':True},
        {'Pclass':3,'Sex':'male','Age':2,'SibSp':3,'Parch':1,'Fare':21.07,'Embarked':'S','ActualSurvived':False},
        {'Pclass':1,'Sex':'female','Age':58,'SibSp':0,'Parch':0,'Fare':146.52,'Embarked':'C','ActualSurvived':True},
        {'Pclass':2,'Sex':'male','Age':30,'SibSp':0,'Parch':0,'Fare':13.0,'Embarked':'S','ActualSurvived':False},
    ]
    return jsonify(passengers)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
