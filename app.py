import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Pet Classifier",
    page_icon="🐾",
    layout="wide"
)
# ---------------- CSS ----------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- HEADER ----------------
st.markdown('<div class="title">🐾 AI Pet Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time AI predictions with analytics & insights</div>', unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
        return tf.keras.models.load_model("model.h5", compile=False)
model = load_model()

# ---------------- PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((150,150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1,2], gap="large")

# ---------------- LEFT PANEL ----------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    image = None

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=300)
        except UnidentifiedImageError:
            st.error("Invalid image file")

    if image and st.button("Analyze Image"):
        with st.spinner("Running model..."):
            processed = preprocess_image(image)
            pred = float(model.predict(processed)[0][0])

        pred = max(0, min(1, pred))

        if pred > 0.5:
            label = "Dog"
            conf = pred
        else:
            label = "Cat"
            conf = 1 - pred

        st.session_state.history.append({
            "Label": label,
            "Confidence": conf
        })

        st.success(f"{label} • {conf*100:.2f}% confidence")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT PANEL ----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)
        df["Step"] = range(1, len(df)+1)

        # -------- KPIs --------
        k1, k2, k3 = st.columns(3)

        k1.markdown(f"""
        <div class="metric-title">Total Predictions</div>
        <div class="metric-value">{len(df)}</div>
        """, unsafe_allow_html=True)

        k2.markdown(f"""
        <div class="metric-title">Avg Confidence</div>
        <div class="metric-value">{df['Confidence'].mean()*100:.1f}%</div>
        """, unsafe_allow_html=True)

        k3.markdown(f"""
        <div class="metric-title">Dog Ratio</div>
        <div class="metric-value">{(df['Label']=='Dog').mean()*100:.1f}%</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------- CHARTS --------
        c1, c2 = st.columns(2)

        # Donut chart
        with c1:
            dist = df["Label"].value_counts().reset_index()
            dist.columns = ["Label", "Count"]

            fig1 = px.pie(
                dist,
                names="Label",
                values="Count",
                hole=0.55
            )

            fig1.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig1, config={"displayModeBar": False})

        # Line chart
        with c2:
            fig2 = px.line(
                df,
                x="Step",
                y="Confidence",
                markers=True
            )

            fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig2, config={"displayModeBar": False})

        # -------- GAUGE --------
        last = df.iloc[-1]

        st.markdown("### Model Confidence")

        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(last["Confidence"]) * 100,
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "#ef4444"},
                    {'range': [50, 75], 'color': "#f59e0b"},
                    {'range': [75, 100], 'color': "#22c55e"},
                ],
            }
        ))

        fig3.update_layout(height=250, margin=dict(t=0, b=0))
        st.plotly_chart(fig3, config={"displayModeBar": False})

    else:
        st.info("Upload an image to start analytics")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY ----------------
st.markdown("## Prediction History")

if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history)
    hist["Confidence"] = hist["Confidence"].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(hist)
else:
    st.write("No predictions yet")