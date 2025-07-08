import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="🔮 Prédiction de Churn",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .input-section h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .prediction-card h2 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .result-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .result-danger {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .fun-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .fun-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1rem;
    }
    
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    .stNumberInput > div > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    .analysis-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .analysis-section h2 {
        color: #2c3e50;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modele.joblib')
        return model
    except FileNotFoundError:
        st.error("🚨 Le fichier 'modele.joblib' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire que cette application.")
        return None

# Fonction pour faire la prédiction
def predict_churn(model, data):
    try:
        # Convertir les données en DataFrame
        df = pd.DataFrame([data])
        
        # Faire la prédiction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        return None, None

# Interface utilisateur
def main():
    # Header principal avec style
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Prédiction de Churn Client</h1>
        <p>Découvrez l'avenir de vos clients avec l'intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar améliorée
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">ℹ️ Informations</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: #2c3e50;">
        <p style="margin: 0; line-height: 1.6;">
            🤖 <strong>Cette application utilise un modèle de Machine Learning avancé</strong><br>
            📊 <strong>Prédiction en temps réel</strong><br>
            🎯 <strong>Précision optimisée</strong><br>
            💡 <strong>Insights actionables</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Colonnes pour organiser les inputs avec style
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h3>📍 Informations Générales</h3>', unsafe_allow_html=True)
        region = st.selectbox("🌍 Région", options=list(range(1, 15)), index=0)
        tenure = st.number_input("⏰ Durée (TENURE)", min_value=0, max_value=1000, value=12)
        montant = st.number_input("💰 Montant", min_value=0.0, value=1000.0, step=10.0)
        frequence_rech = st.number_input("🔄 Fréquence Recharge", min_value=0.0, value=5.0, step=0.1)
        revenue = st.number_input("💵 Revenue", min_value=0.0, value=500.0, step=10.0)
        arpu_segment = st.number_input("📈 ARPU Segment", min_value=0.0, value=100.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Données d\'Usage</h3>', unsafe_allow_html=True)
        frequence = st.number_input("📶 Fréquence", min_value=0.0, value=10.0, step=0.1)
        data_volume = st.number_input("📱 Volume de Données", min_value=0.0, value=1000.0, step=10.0)
        on_net = st.number_input("🌐 On Net", min_value=0.0, value=50.0, step=1.0)
        orange = st.number_input("🟠 Orange", min_value=0.0, value=30.0, step=1.0)
        tigo = st.number_input("🔵 Tigo", min_value=0.0, value=20.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<h3>🎯 Autres Métriques</h3>', unsafe_allow_html=True)
        mrg = st.selectbox("🎲 MRG", options=[0, 1], index=0)
        regularity = st.selectbox("📅 Régularité", options=[0, 1], index=1)
        top_pack = st.selectbox("⭐ Top Pack", options=[0, 1], index=0)
        freq_top_pack = st.number_input("🔝 Fréquence Top Pack", min_value=0.0, value=2.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton de prédiction stylisé
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    predict_button = st.button("🚀 Lancer la Prédiction Magique", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        # Animation de chargement
        with st.spinner('🔮 Consultation de la boule de cristal...'):
            # Préparer les données
            data = {
                'REGION': region,
                'TENURE': tenure,
                'MONTANT': montant,
                'FREQUENCE_RECH': frequence_rech,
                'REVENUE': revenue,
                'ARPU_SEGMENT': arpu_segment,
                'FREQUENCE': frequence,
                'DATA_VOLUME': data_volume,
                'ON_NET': on_net,
                'ORANGE': orange,
                'TIGO': tigo,
                'MRG': mrg,
                'REGULARITY': regularity,
                'TOP_PACK': top_pack,
                'FREQ_TOP_PACK': freq_top_pack
            }
            
            # Faire la prédiction
            prediction, probability = predict_churn(model, data)
            
            if prediction is not None:
                # Afficher les résultats avec style
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown('<h2>🎭 Résultats de la Prédiction</h2>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="result-danger">
                            <h3>⚠️ ALERTE CHURN</h3>
                            <p>Le client risque de résilier !</p>
                            <p>🎯 Action recommandée : Rétention urgente</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="result-success">
                            <h3>✅ CLIENT FIDÈLE</h3>
                            <p>Le client va probablement rester</p>
                            <p>🎉 Opportunité : Upselling</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if probability is not None:
                        churn_prob = probability[1] * 100
                        
                        # Métrique stylisée
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{churn_prob:.1f}%</h3>
                            <p>Probabilité de Churn</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Gauge chart amélioré
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = churn_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Risque de Churn (%)", 'font': {'size': 20, 'color': '#2c3e50'}},
                            delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue", 'thickness': 0.3},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 25], 'color': 'rgba(0, 255, 0, 0.3)'},
                                    {'range': [25, 50], 'color': 'rgba(255, 255, 0, 0.3)'},
                                    {'range': [50, 75], 'color': 'rgba(255, 165, 0, 0.3)'},
                                    {'range': [75, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(
                            height=350,
                            font={'color': "darkblue", 'family': "Poppins"},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Section d'analyse des données avec style
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.markdown('<h2>🎨 Analyse Visuelle du Profil Client</h2>', unsafe_allow_html=True)
    
    # Créer des visualisations interactives
    current_data = {
        'REGION': region,
        'TENURE': tenure,
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'MRG': mrg,
        'REGULARITY': regularity,
        'TOP_PACK': top_pack,
        'FREQ_TOP_PACK': freq_top_pack
    }
    
    # Graphique radar amélioré
    metrics_normalized = {
        'Revenue': min(revenue / 1000, 1),
        'ARPU': min(arpu_segment / 200, 1),
        'Data Volume': min(data_volume / 2000, 1),
        'Tenure': min(tenure / 100, 1),
        'Fréquence': min(frequence / 20, 1)
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics_normalized.values()),
        theta=list(metrics_normalized.keys()),
        fill='toself',
        name='Profil Client',
        line=dict(color='#667eea', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color='#2c3e50', size=12),
                gridcolor='rgba(102, 126, 234, 0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(color='#2c3e50', size=14, family='Poppins'),
                gridcolor='rgba(102, 126, 234, 0.3)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title={
            'text': "🎯 Profil Client (Valeurs Normalisées)",
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Poppins'},
            'x': 0.5
        },
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique en barres pour les métriques principales
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            x=list(metrics_normalized.keys()),
            y=list(metrics_normalized.values()),
            title="📊 Métriques Clés",
            color=list(metrics_normalized.values()),
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(
            title_font=dict(size=16, color='#2c3e50', family='Poppins'),
            showlegend=False,
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Graphique en secteurs pour la répartition des opérateurs
        operators = ['On Net', 'Orange', 'Tigo']
        values = [on_net, orange, tigo]
        
        fig_pie = px.pie(
            values=values,
            names=operators,
            title="📱 Répartition des Opérateurs",
            color_discrete_sequence=['#667eea', '#f093fb', '#ffeaa7']
        )
        fig_pie.update_layout(
            title_font=dict(size=16, color='#2c3e50', family='Poppins'),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer stylisé
    st.markdown("""
    <div class="footer">
        <h3>🎨 Application de Prédiction de Churn</h3>
        <p>Développée avec ❤️ en utilisant Streamlit & Plotly</p>
        <p>🚀 Propulsée par l'Intelligence Artificielle</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()