# xgboost_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\dell\\Desktop\\Old Firefox Data\\clean_data_1.csv")

    categorical_cols = [
        'Purchase Type', 'Payment Method', 'Railcard', 'Ticket Class',
        'Ticket Type', 'Price', 'Departure Station', 'Arrival Destination',
        'Date of Journey', 'Journey Status', 'Reason for Delay',
        'Journey_Happened','time lag', 'Route', 'Is Delayed', 'Weekday',
        'Delay_minutes', 'Peak Period'
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    datetime_cols = ['Date of Purchase', 'Time of Purchase', 'Date of Journey',
                     'Departure Time', 'Arrival Time', 'approximated time', 'Actual Arrival Time',
                     'time lag', 'Delay_minutes','actual time spent']

    def safe_datetime_conversion(series):
        return pd.to_datetime(series, errors='coerce').astype('int64') // 10**9

    for col in datetime_cols:
        df[col] = safe_datetime_conversion(df[col])

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    df['Refund Request'] = df['Refund Request'].map({'Yes': 1, 'No': 0})

    return df

# Model training function
def train_xgboost_model(X_train, y_train, scale_pos_weight):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

# Threshold tuning function
def find_balanced_threshold(y_true, y_proba, target_recall=0.75, min_precision=0.60):
    thresholds = np.arange(0.2, 0.8, 0.02)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if sum(y_pred) == 0:
            continue
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        if recall >= target_recall and precision >= min_precision and f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold

# Main Streamlit App
st.title("Refund Prediction using XGBoost")
df = load_data()
st.write("### Sample of Raw Data")
st.dataframe(df.head())

# Prepare features and target
y = df['Refund Request']
X = df.drop(['Refund Request', 'Transaction ID'], axis=1)
X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
model = train_xgboost_model(X_train, y_train, imbalance_ratio)

# Predictions and threshold tuning
y_proba = model.predict_proba(X_test)[:, 1]
best_threshold = find_balanced_threshold(y_test, y_proba)
y_pred = (y_proba >= best_threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Display metrics
st.write("### Model Evaluation Metrics")
st.write(f"**Threshold:** {best_threshold:.2f}")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")
st.write(f"**ROC AUC:** {roc_auc:.2f}")

# Feature Importance
st.write("### Feature Importances")
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
st.dataframe(importance_df.head(10))

# Optional: Prediction form
st.write("### Try It Yourself")
example_input = X.sample(1).copy()
user_input = {}

for col in X.columns:
    val = st.number_input(f"{col}", float(example_input[col].values[0]))
    user_input[col] = val

input_df = pd.DataFrame([user_input])
pred_prob = model.predict_proba(input_df)[0, 1]
pred_label = int(pred_prob >= best_threshold)

st.write(f"**Predicted Refund Probability:** {pred_prob:.2f}")
st.write(f"**Predicted Class:** {'Refund' if pred_label == 1 else 'No Refund'}")
