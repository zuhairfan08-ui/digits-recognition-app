import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="🔢", layout="centered")

@st.cache_resource
def load_model():
    with open('digits_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('digits_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.title("🔢 Handwritten Digit Recognizer")
st.markdown("64 pixel values enter karo — model predict karega ke ye digit kaunsa number hai (0-9).")
st.divider()

option = st.radio("Input method chunno:", ["📋 Sample data se test karo", "✏️ Khud pixel values likho"])

if option == "📋 Sample data se test karo":
    df = pd.read_csv('digits_dataset.csv')
    sample_idx = st.slider("Sample row select karo", 0, len(df)-1, 0)
    sample_row = df.iloc[sample_idx]
    actual_label = int(sample_row['label'])
    pixel_values = sample_row.drop('label').values
    st.info(f"✅ Actual Label (sahi jawab): **{actual_label}**")
else:
    st.markdown("**64 pixel values enter karo (0-16 ke beech):**")
    cols = st.columns(8)
    pixel_values = []
    for i in range(64):
        with cols[i % 8]:
            val = st.number_input(f"p{i}", min_value=0.0, max_value=16.0, value=0.0, step=1.0, label_visibility="collapsed")
            pixel_values.append(val)
    pixel_values = np.array(pixel_values)

st.divider()

if st.button("🔍 Predict karo", use_container_width=True, type="primary"):
    input_array = np.array(pixel_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    st.divider()
    st.success(f"## 🎯 Predicted Digit: **{prediction}**")
    st.metric("Confidence", f"{max(probability)*100:.1f}%")
    st.divider()
    st.subheader("📊 All Digit Probabilities")
    prob_df = pd.DataFrame({'Digit': list(range(10)), 'Probability (%)': [round(p*100, 2) for p in probability]})
    st.bar_chart(prob_df.set_index('Digit'))

st.divider()
st.caption("Model: Best ML Classifier | Dataset: Digits Dataset (64 pixels)")