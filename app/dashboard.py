"""Dashboard Streamlit enrichi pour la visualisation des prévisions énergétiques."""

import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Configuration API
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

# Configuration de la page
st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("⚡ Dashboard de Prévision Énergétique")
st.markdown("---")

# Fonction pour récupérer les modèles disponibles
@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_available_models():
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Sidebar avec toutes les options
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # Section Modèle
    st.subheader("🤖 Modèle ML")
    models_info = get_available_models()
    
    if models_info:
        model_options = {}
        model_descriptions = {}
        for key, info in models_info["models"].items():
            model_options[info["name"]] = key
            model_descriptions[info["name"]] = info["description"]
        
        selected_model_name = st.selectbox(
            "Choisir le modèle",
            options=list(model_options.keys()),
            help="Sélectionner l'algorithme de machine learning"
        )
        selected_model = model_options[selected_model_name]
        
        # Afficher la description du modèle
        st.info(f"ℹ️ {model_descriptions[selected_model_name]}")
        
        # Support des intervalles
        model_supports_intervals = models_info["models"][selected_model]["supports_intervals"]
        if model_supports_intervals:
            st.success("✅ Ce modèle supporte les intervalles de confiance")
        else:
            st.warning("⚠️ Ce modèle ne supporte pas les intervalles")
    else:
        st.error("❌ Impossible de récupérer les modèles")
        selected_model = "linear"
        model_supports_intervals = False
    
    st.markdown("---")
    
    # Section Données
    st.subheader("📊 Source de Données")
    data_source = st.selectbox(
        "Source des données",
        options=["synthetic", "odre"],
        format_func=lambda x: {
            "synthetic": "🎲 Données Synthétiques",
            "odre": "🏭 RTE ODRÉ (Réelles)"
        }[x],
        help="Choisir entre données synthétiques ou réelles"
    )
    
    # Section Géographie
    st.subheader("🌍 Localisation")
    cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille"]
    city = st.selectbox("Ville", options=cities, help="Ville pour la prévision")
    
    st.markdown("---")
    
    # Section Prévision
    st.subheader("📈 Paramètres de Prévision")
    
    # Horizon temporel
    horizon_type = st.radio(
        "Type d'horizon",
        options=["hours", "days"],
        format_func=lambda x: {"hours": "⏰ Heures", "days": "📅 Jours"}[x]
    )
    
    if horizon_type == "hours":
        horizon_value = st.slider("Horizon (heures)", 1, 168, 24)
        horizon_hours = horizon_value
    else:
        horizon_value = st.slider("Horizon (jours)", 1, 14, 3)
        horizon_hours = horizon_value * 24
    
    # Options avancées
    st.subheader("⚙️ Options Avancées")
    
    with_intervals = st.checkbox(
        "Intervalles de confiance",
        value=model_supports_intervals,
        disabled=not model_supports_intervals,
        help="Afficher les bornes de prédiction (si supporté par le modèle)"
    )
    
    confidence_level = st.selectbox(
        "Niveau de confiance",
        options=[0.8, 0.9, 0.95],
        format_func=lambda x: f"{x*100:.0f}%",
        help="Niveau de confiance pour les intervalles"
    )
    
    # Granularité d'affichage
    display_granularity = st.selectbox(
        "Granularité d'affichage",
        options=["all", "daily_avg", "hourly_avg"],
        format_func=lambda x: {
            "all": "🔍 Tous les points",
            "daily_avg": "📊 Moyenne journalière", 
            "hourly_avg": "⏱️ Moyenne horaire"
        }[x]
    )

# Section principale
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📋 Résumé Configuration")
    config_summary = f"""
    **Modèle:** {selected_model_name if 'selected_model_name' in locals() else 'Linear Regression'}
    **Source:** {data_source.upper()}
    **Ville:** {city}
    **Horizon:** {horizon_hours}h ({horizon_value} {horizon_type[:-1]}{'s' if horizon_value > 1 else ''})
    **Intervalles:** {'Oui' if with_intervals else 'Non'}
    """
    st.markdown(config_summary)

with col1:
    if st.button("🚀 Lancer la Prévision", type="primary", use_container_width=True):
        with st.spinner("🔄 Génération des prévisions en cours..."):
            try:
                # Préparer la requête
                forecast_request = {
                    "horizon": horizon_hours,
                    "city": city,
                    "with_intervals": with_intervals,
                    "model_type": selected_model,
                    "data_source": data_source
                }
                
                # Appel API
                response = requests.post(
                    f"{API_URL}/forecast",
                    json=forecast_request,
                    timeout=60,
                )
                response.raise_for_status()
                forecast_data = response.json()
                
                # Stocker dans session state
                st.session_state["forecast"] = forecast_data
                st.session_state["config"] = forecast_request
                
                st.success(f"✅ Prévision générée avec {len(forecast_data['timestamps'])} points")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Erreur de connexion API: {e}")
            except Exception as e:
                st.error(f"❌ Erreur lors de la prévision: {e}")

# Affichage des résultats
if "forecast" in st.session_state:
    data = st.session_state["forecast"]
    config = st.session_state.get("config", {})
    
    st.markdown("---")
    st.header("📊 Résultats de la Prévision")
    
    # Créer le DataFrame
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["timestamps"]),
        "prediction": data["yhat"],
        "lower_bound": data.get("yhat_lower", [None]*len(data["yhat"])),
        "upper_bound": data.get("yhat_upper", [None]*len(data["yhat"])),
    }).set_index("timestamp")
    
    # Traitement selon la granularité
    if display_granularity == "daily_avg":
        df_display = df.resample('D').mean()
        title_suffix = " (Moyenne Journalière)"
    elif display_granularity == "hourly_avg":
        df_display = df.resample('H').mean()
        title_suffix = " (Moyenne Horaire)"
    else:
        df_display = df
        title_suffix = ""
    
    # Graphique principal avec Plotly
    fig = go.Figure()
    
    # Ligne de prédiction principale
    fig.add_trace(go.Scatter(
        x=df_display.index,
        y=df_display["prediction"],
        mode='lines',
        name='Prédiction',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Intervalles de confiance si disponibles
    if with_intervals and df_display["lower_bound"].notna().any():
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display["upper_bound"],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Borne Supérieure'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=df_display["lower_bound"],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name=f'Intervalle {confidence_level*100:.0f}%'
        ))
    
    # Configuration du graphique
    fig.update_layout(
        title=f"Prévision de Consommation Énergétique - {city}{title_suffix}",
        xaxis_title="Temps",
        yaxis_title="Consommation (MW)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Métriques et statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Prédiction Moyenne",
            f"{df['prediction'].mean():.1f} MW"
        )
    
    with col2:
        st.metric(
            "📈 Maximum Prédit",
            f"{df['prediction'].max():.1f} MW"
        )
    
    with col3:
        st.metric(
            "📉 Minimum Prédit",
            f"{df['prediction'].min():.1f} MW"
        )
    
    with col4:
        volatility = df['prediction'].std()
        st.metric(
            "📊 Volatilité",
            f"{volatility:.1f} MW"
        )
    
    # Tableau de données détaillé
    with st.expander("📋 Données Détaillées"):
        st.dataframe(
            df.reset_index().round(2),
            use_container_width=True,
            height=300
        )
    
    # Informations sur le modèle
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🤖 Modèle utilisé:** {data.get('model_name', 'Unknown')}
        **📊 Version:** {data.get('model_version', 'Unknown')}
        **🔧 Type:** {config.get('model_type', 'Unknown')}
        """)
    
    with col2:
        st.info(f"""
        **📍 Ville:** {config.get('city', 'Unknown')}
        **💾 Source:** {config.get('data_source', 'Unknown').upper()}
        **⏰ Points générés:** {len(data['timestamps'])}
        """)

else:
    st.info("👆 Configure les paramètres dans la barre latérale et lance une prévision pour voir les résultats.")

# Footer
st.markdown("---")
st.markdown("*Dashboard développé avec Streamlit - Energy Forecasting Service*")