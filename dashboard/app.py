import os
import requests
import pandas as pd
import streamlit as st

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("⚡ Energy Forecasting Dashboard")

with st.sidebar:
    st.header("Configuration")
    city = st.text_input("Ville", os.getenv("CITY", "Paris"))
    horizon_days = st.slider("Horizon (jours)", 1, 14, 7)
    with_intervals = st.checkbox("Intervalles de confiance", True)
    if st.button("Lancer la prévision"):
        try:
            r = requests.post(
                f"{API_URL}/forecast",
                json={"horizon": horizon_days*24, "city": city, "with_intervals": with_intervals},
                timeout=60,
            )
            r.raise_for_status()
            st.session_state["forecast"] = r.json()
        except Exception as e:
            st.error(f"Erreur API: {e}")

data = st.session_state.get("forecast")
if not data:
    st.info("Configure la prévision dans la barre latérale puis clique sur *Lancer la prévision*.")
else:
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["timestamps"]),
        "yhat": data["yhat"],
        "yhat_lower": data.get("yhat_lower", [None]*len(data["yhat"])),
        "yhat_upper": data.get("yhat_upper", [None]*len(data["yhat"])),
    }).set_index("timestamp")
    st.subheader("Prévisions")
    st.line_chart(df[["yhat"]])
    if df["yhat_lower"].notna().any() and df["yhat_upper"].notna().any():
        st.area_chart(df[["yhat_lower","yhat_upper"]])
    st.subheader("Aperçu (table)")
    st.dataframe(df.tail(168))
    st.caption(f"Modèle: {data.get('model_name','?')} — version: {data.get('model_version','?')}")
