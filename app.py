import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="HealTwin AI: Code Rakshaks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED PATH LOGIC
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

model_path = os.path.join(current_dir, 'heart_model.bin')
features_path = os.path.join(current_dir, 'features.pkl')

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"Model NOT found at: {model_path}")
        st.warning("Please run your 'Model Training Notebook' to generate the .bin file.")
        return None, None
    
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

model, feature_names = load_model()

def get_twin_svg(risk_score, aqi_level):
    heart_color = "#ff4b4b" if risk_score > 0.7 else "#ffa500" if risk_score > 0.4 else "#00c851"
    
    body_path = "M50,10 L70,30 L60,30 L60,80 L70,180 L55,180 L50,100 L45,180 L30,180 L40,80 L40,30 L30,30 Z"
    
    svg = f"""
    <svg width="250" height="400" viewBox="0 0 100 200">
        <circle cx="50" cy="50" r="45" fill="{ '#555' if aqi_level > 200 else '#e0f7fa' }" opacity="0.3">
            <animate attributeName="r" values="40;45;40" dur="4s" repeatCount="indefinite" />
        </circle>
        
        <path d="{body_path}" fill="#333" stroke="white" stroke-width="1" />
        
        <circle cx="50" cy="50" r="5" fill="{heart_color}">
            <animate attributeName="r" values="5;6;5" dur="{ '0.5s' if risk_score > 0.5 else '1s' }" repeatCount="indefinite" />
        </circle>
        
        <text x="20" y="190" fill="white" font-size="8">Risk: {risk_score:.1%}</text>
    </svg>
    """
    return svg

with st.sidebar:
    st.title("🧬 Digital Twin v1.0")
    
    if model:
        st.subheader("Environment Sensor")
        aqi = st.slider("Live AQI (Air Quality)", 0, 500, 150)
        
        st.subheader("Patient Vitals")
        age = st.slider("Age", 20, 80, 55)
        chol = st.slider("Cholesterol", 100, 400, 200)
        bp = st.slider("Resting BP", 90, 200, 120)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)
        
        user_input = [age, 1, cp, bp, chol, 0, 1, 150, 0, 1.0, 1, 0, 3]

        risk_prob = model.predict_proba([user_input])[0][1]
        
        st.markdown("### Your Digital Body")
        st.markdown(get_twin_svg(risk_prob, aqi), unsafe_allow_html=True)
        st.caption(f"Twin ID: {id(model)}")

st.title("HealTwin AI Coach")
st.markdown("#### *Preventive Healthcare & Early Warning System*")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I am initialized and connected to your Digital Twin. I notice your environment AQI is updated. How are you feeling?"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about your health status..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if model:
        importances = model.feature_importances_
        top_feature_index = np.argmax(importances)
        primary_driver = feature_names[top_feature_index]
        
        response = ""
        if risk_prob > 0.7:
            response = f" **Alert:** Your Digital Twin is showing RED status ({risk_prob:.1%} Risk). \n\nMy analysis shows that **{primary_driver}** is the main contributing factor. Please consult a cardiologist."
        elif aqi > 300:
            response = f" **Environmental Warning:** Your internal vitals are stable, but the AQI is {aqi} (Hazardous). I recommend wearing a mask to protect your Twin's lung health."
        else:
            response = f" **Status Optimal:** Your risk is low ({risk_prob:.1%}). Keep maintaining your {primary_driver} levels!"
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        st.error("I cannot analyze without the Brain (Model). Please check the sidebar.")