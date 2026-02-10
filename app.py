import streamlit as st
import numpy as np
import os

try:
    import joblib
except ModuleNotFoundError:
    joblib = None


st.set_page_config(
    page_title="HealTwin AI: Code Rakshaks",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

model_path = os.path.join(current_dir, "heart_model.bin")
features_path = os.path.join(current_dir, "features.pkl")


@st.cache_resource
def load_model():
    if joblib is None:
        st.error("Missing dependency: joblib")
        st.info("Add joblib to requirements.txt and redeploy the app.")
        return None, None

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        return None, None

    if not os.path.exists(features_path):
        st.error(f"Features file not found at: {features_path}")
        return None, None

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


model, feature_names = load_model()
risk_prob = None


def get_twin_svg(risk_score, aqi_level):
    heart_color = (
        "#ff4b4b" if risk_score > 0.7
        else "#ffa500" if risk_score > 0.4
        else "#00c851"
    )

    body_path = (
        "M50,10 L70,30 L60,30 L60,80 L70,180 "
        "L55,180 L50,100 L45,180 L30,180 "
        "L40,80 L40,30 L30,30 Z"
    )

    return f"""
    <svg width="250" height="400" viewBox="0 0 100 200">
        <circle cx="50" cy="50" r="45"
            fill="{ '#555' if aqi_level > 200 else '#e0f7fa' }"
            opacity="0.3">
            <animate attributeName="r" values="40;45;40" dur="4s" repeatCount="indefinite" />
        </circle>

        <path d="{body_path}" fill="#333" stroke="white" stroke-width="1" />

        <circle cx="50" cy="50" r="5" fill="{heart_color}">
            <animate attributeName="r" values="5;6;5"
                dur="{ '0.5s' if risk_score > 0.5 else '1s' }"
                repeatCount="indefinite" />
        </circle>

        <text x="20" y="190" fill="white" font-size="8">
            Risk: {risk_score:.1%}
        </text>
    </svg>
    """


with st.sidebar:
    st.title("Digital Twin v1.0")

    if model is not None:
        st.subheader("Environment Sensor")
        aqi = st.slider("Live AQI (Air Quality)", 0, 500, 150)

        st.subheader("Patient Vitals")
        age = st.slider("Age", 20, 80, 55)
        chol = st.slider("Cholesterol", 100, 400, 200)
        bp = st.slider("Resting BP", 90, 200, 120)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)

        user_input = [age, 1, cp, bp, chol, 0, 1, 150, 0, 1.0, 1, 0, 3]
        risk_prob = model.predict_proba([user_input])[0][1]

        st.markdown("Your Digital Body")
        st.markdown(get_twin_svg(risk_prob, aqi), unsafe_allow_html=True)
        st.caption(f"Twin ID: {id(model)}")

    else:
        st.warning("Model not loaded. Running in safe mode.")


st.title("HealTwin AI Coach")
st.markdown("Preventive Healthcare and Early Warning System")


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello. I am connected to your Digital Twin. "
            "Your environment data is active. How are you feeling today?"
        )
    }]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Ask about your health status..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if model is None or risk_prob is None:
        response = "Health analysis is unavailable because the model is not loaded."
    else:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            primary_driver = feature_names[np.argmax(importances)]
        else:
            primary_driver = "multiple contributing factors"

        if risk_prob > 0.7:
            response = (
                f"High risk detected ({risk_prob:.1%}). "
                f"Primary contributing factor is {primary_driver}. "
                "Please consult a cardiologist."
            )
        elif aqi > 300:
            response = (
                f"Environmental risk detected. AQI is {aqi}. "
                "Limit outdoor exposure."
            )
        else:
            response = (
                f"Risk level is low ({risk_prob:.1%}). "
                f"Maintain healthy {primary_driver} levels."
            )

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
