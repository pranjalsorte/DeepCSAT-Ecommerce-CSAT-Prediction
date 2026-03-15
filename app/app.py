import streamlit as st
import numpy as np
import tensorflow as tf

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(page_title="DeepCSAT Predictor")
st.title("📊 DeepCSAT - CSAT Score Prediction")

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/lstm_model.keras")
    return model

model = load_model()

# ======================================================
# USER INPUTS
# ======================================================

st.subheader("Enter Details")

channel_name = st.selectbox("Channel", ["Inbound", "Outcall"])
category = st.selectbox(
    "Category",
    ["Order Related", "Returns", "Cancellation", "Product Queries"]
)

product_category = st.selectbox(
    "Product Category",
    ["Electronics", "Fashion", "Home", "Other"]
)

item_price = st.number_input("Item Price", 0.0, 100000.0, 1000.0)

tenure_bucket = st.selectbox(
    "Tenure",
    ["0-30", "30-60", "60-90", ">90", "On Job Training"]
)

agent_shift = st.selectbox(
    "Agent Shift",
    ["Morning", "Evening", "Night"]
)

# ======================================================
# ENCODING MAPS
# ======================================================

channel_map = {"Inbound": 0, "Outcall": 1}

category_map = {
    "Order Related": 0,
    "Returns": 1,
    "Cancellation": 2,
    "Product Queries": 3,
}

product_map = {
    "Electronics": 0,
    "Fashion": 1,
    "Home": 2,
    "Other": 3,
}

tenure_map = {
    "0-30": 0,
    "30-60": 1,
    "60-90": 2,
    ">90": 3,
    "On Job Training": 4,
}

shift_map = {"Morning": 0, "Evening": 1, "Night": 2}

# ======================================================
# PREDICTION
# ======================================================

if st.button("🔮 Predict CSAT Score"):

    # MODEL EXPECTS 17 FEATURES
    input_vector = np.zeros(17)

    # Insert user inputs into first positions
    input_vector[0] = channel_map[channel_name]
    input_vector[1] = category_map[category]
    input_vector[2] = product_map[product_category]
    input_vector[3] = item_price
    input_vector[4] = tenure_map[tenure_bucket]
    input_vector[5] = shift_map[agent_shift]

    # reshape for LSTM → (batch, timestep, features)
    input_vector = input_vector.reshape(1, 1, 17)

    prediction = model.predict(input_vector)

    csat_score = np.argmax(prediction)

    st.success(f"✅ Predicted CSAT Score: {csat_score}")

    st.info("""
    CSAT Meaning:
    0 → Very Dissatisfied
    1 → Dissatisfied
    2 → Neutral
    3 → Satisfied
    4 → Very Satisfied
    """)