import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load the original dataset
df = pd.read_csv("https://raw.githubusercontent.com/Lavenderlime/your-repo-name/main/clean_data_1.csv")

# Drop 'Transaction ID'
if 'Transaction ID' in df.columns:
    df.drop('Transaction ID', axis=1, inplace=True)

# Convert target to binary
target = 'Refund Request'
df[target] = df[target].apply(lambda x: 1 if str(x).lower() in ['1', 'yes'] else 0)

# Store category mappings
category_mappings = {}

# Build category mappings before encoding
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')
    category_mappings[col] = dict(enumerate(df[col].cat.categories))
    # Invert for lookup: label -> code
    category_mappings[col] = {v: k for k, v in category_mappings[col].items()}
    df[col] = df[col].cat.codes

X = df.drop(columns=[target])
y = df[target]

# Train the model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# --- Streamlit UI ---
st.title("Refund Prediction using XGBoost")

st.markdown("""
This app uses an XGBoost model to predict whether a **refund** will be requested for a UK rail journey.  
- **Yes (1)** = Refund Requested → Revenue **loss**  
- **No (0)** = No Refund → Revenue **retained**
""")

st.subheader("Enter Journey Details")

user_inputs = {}

for col in X.columns:
    if "time" in col.lower():
        time_str = st.text_input(f"{col} (HH:MM)", "00:00")
        try:
            h, m = map(int, time_str.strip().split(":"))
            user_inputs[col] = h * 3600 + m * 60
        except:
            st.warning("Invalid time format. Use HH:MM")
            user_inputs[col] = 0

    elif "lag" in col.lower() or "spent" in col.lower() or "approximated" in col.lower():
        user_inputs[col] = st.number_input(f"{col} (in seconds)", min_value=0, value=0)

    elif "date" in col.lower():
        date_val = st.date_input(col)
        user_inputs[col] = int(pd.Timestamp(date_val).timestamp())

    elif col in category_mappings:
        options = list(category_mappings[col].keys())
        selected_label = st.selectbox(f"{col}", options)
        user_inputs[col] = category_mappings[col][selected_label]

    else:
        user_inputs[col] = st.number_input(f"{col}", value=int(df[col].mean()))

# --- Predict ---
if st.button("Predict Refund"):
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    label = "Yes (Refund Requested – Revenue Loss)" if prediction == 1 else "No (No Refund – Revenue Retained)"
    
    st.subheader("Prediction")
    st.success(f"Refund Prediction: **{label}**")
