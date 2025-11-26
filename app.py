import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils import TextCleaner

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Maladie Cardiaque",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - style professionnel médical
st.markdown("""
<style>

    /* --- GLOBAL --- */
    .stApp {
        background-color: #1e3a8a !important;
    }

    .main .block-container {
        background: #1e3a8a !important;
        padding: 1.5rem !important;
        border-radius: 14px;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }

    /* --- TITRES --- */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 2.6rem !important;
    }

    /* --- TEXTES --- */
    p, label, span {
        color: #ffffff !important;
    }

    /* --- INPUTS --- */
    input, select, textarea {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        border-radius: 8px !important;
    }

    input:focus, select:focus, textarea:focus {
        border: 2px solid #93c5fd !important;
        box-shadow: 0px 0px 8px rgba(147,197,253,0.5) !important;
    }

    /* --- BUTTON --- */
    .stButton>button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 25px !important;
        font-size: 1rem;
        border: none !important;
        box-shadow: 0 4px 10px rgba(59,130,246,0.4);
    }
    .stButton>button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(59,130,246,0.5);
    }

    /* --- SUCCESS BOX --- */
    .stSuccess {
        background-color: #1e3a8a !important;
        border-left: 4px solid #60a5fa !important;
        color: #ffffff !important;
    }

    /* --- ERROR BOX --- */
    .stError {
        background-color: #1e3a8a !important;
        border-left: 4px solid #60a5fa !important;
        color: #ffffff !important;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #1e3a8a !important;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* --- EXPANDER --- */
    .streamlit-expanderHeader {
        background: #1e3a8a !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #3b82f6 !important;
    }

    /* --- DATAFRAME --- */
    .stDataFrame {
        background: #1e3a8a !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
    }

    /* --- HR --- */
    hr {
        border: none;
        height: 2px;
        background: #3b82f6 !important;
        margin: 25px 0;
    }

</style>
""", unsafe_allow_html=True)


# Charger le modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load('Model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Fichier Model.pkl introuvable. Veuillez d'abord exécuter main.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur: {str(e)}")
    st.stop()

# Titre professionnel
st.markdown("""
    <h1>
        🏥 Système de Prédiction du Risque Cardiaque
    </h1>
""", unsafe_allow_html=True)

# Message de bienvenue professionnel
st.markdown("""
    <div style='text-align: center; padding: 20px; background: #1e3a8a; 
                border-radius: 15px; margin-bottom: 30px; border: 2px solid #3b82f6;'>
        <h3 style='color: #ffffff; margin: 0;'>
            🔬 Intelligence Artificielle Médicale
        </h3>
        <p style='color: #ffffff; font-size: 1.1rem; margin-top: 10px;'>
            Analyse prédictive basée sur des algorithmes de Machine Learning
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Interface de saisie
st.markdown("""
    <div style='background: #1e3a8a; padding: 20px; border-radius: 20px; 
                margin-bottom: 30px; border: 2px solid #3b82f6;'>
        <h2 style='text-align: center; color: #ffffff; margin-bottom: 20px;'>
            📋 Données Cliniques du Patient
        </h2>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🩺 Paramètres Cardiovasculaires")
    
    sbp = st.number_input(
        "🫀 Pression Artérielle Systolique (mmHg)",
        min_value=80,
        max_value=250,
        value=120,
        help="💡 Valeur normale: 90-120 mmHg"
    )
    
    ldl = st.number_input(
        "🧪 Cholestérol LDL (mg/dL)",
        min_value=0,
        max_value=1000,
        value=150,
        help="💡 Valeur optimale: < 100 mg/dL"
    )
    
    adiposity = st.number_input(
        "📊 Indice d'Adiposité",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        help="💡 Indicateur de composition corporelle"
    )

with col2:
    st.markdown("### 👤 Informations Personnelles")
    
    obesity = st.number_input(
        "⚖️ Indice de Masse Corporelle (BMI)",
        min_value=10,
        max_value=50,
        value=25,
        help="💡 Normal: 18.5-24.9 | Surpoids: 25-29.9 | Obésité: ≥30"
    )
    
    age = st.number_input(
        "📅 Âge (années)",
        min_value=15,
        max_value=100,
        value=45,
        help="💡 Facteur de risque cardiovasculaire"
    )
    
    famhist = st.selectbox(
        "🧬 Antécédents Familiaux Cardiaques",
        options=["Present", "Absent"],
        help="💡 Historique familial de pathologies cardiaques"
    )

st.markdown("<hr>", unsafe_allow_html=True)

# Bouton de prédiction centré
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 ANALYSER LE PROFIL CLINIQUE", use_container_width=True, type="primary")

if predict_button:
    # Animation de chargement
    with st.spinner('⚙️ Analyse en cours...'):
        # Construire le dataframe
        input_data = pd.DataFrame({
            'sbp': [sbp],
            'ldl': [ldl],
            'adiposity': [adiposity],
            'famhist': [famhist],
            'obesity': [obesity],
            'age': [age]
        })
        
        # Prédiction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Résultats
            st.markdown("""
                <div style='text-align: center; margin-bottom: 30px;'>
                    <h2 style='color: #ffffff; font-size: 2.5rem;'>
                        📊 Résultats de l'Analyse Clinique
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics en grand
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                                padding: 30px; border-radius: 20px; text-align: center;
                                box-shadow: 0 10px 30px rgba(30, 58, 138, 0.6);'>
                        <p style='color: white; font-size: 1.2rem; margin: 0;'>Diagnostic</p>
                        <p style='color: white; font-size: 2.5rem; font-weight: 900; margin: 10px 0;'>
                            {}
                        </p>
                        <p style='color: white; font-size: 1rem; margin: 0;'>{}</p>
                    </div>
                """.format(
                    "⚠️ RISQUE" if prediction == 1 else "✅ NORMAL",
                    "Surveillance requise" if prediction == 1 else "Paramètres normaux"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                                padding: 30px; border-radius: 20px; text-align: center;
                                box-shadow: 0 10px 30px rgba(30, 58, 138, 0.6);'>
                        <p style='color: white; font-size: 1.2rem; margin: 0;'>Probabilité de Risque</p>
                        <p style='color: white; font-size: 3rem; font-weight: 900; margin: 10px 0;'>
                            {:.1f}%
                        </p>
                        <p style='color: white; font-size: 1rem; margin: 0;'>Score de risque</p>
                    </div>
                """.format(probability[1] * 100), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                                padding: 30px; border-radius: 20px; text-align: center;
                                box-shadow: 0 10px 30px rgba(30, 58, 138, 0.6);'>
                        <p style='color: white; font-size: 1.2rem; margin: 0;'>Probabilité Normale</p>
                        <p style='color: white; font-size: 3rem; font-weight: 900; margin: 10px 0;'>
                            {:.1f}%
                        </p>
                        <p style='color: white; font-size: 1rem; margin: 0;'>État cardiovasculaire</p>
                    </div>
                """.format(probability[0] * 100), unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Interprétation clinique
            st.markdown("### 🔬 Interprétation Clinique")
            
            if prediction == 1:
                st.markdown(f"""
                ### ⚠️ Profil à Risque Cardiovasculaire Détecté
                
                **Score de risque calculé: {probability[1]:.1%}**
                
                L'algorithme d'apprentissage automatique a identifié un profil à risque cardiovasculaire 
                élevé basé sur l'analyse multi-paramétrique des données cliniques.
                
                #### 🏥 Recommandations Médicales:
                
                1. **Consultation Spécialisée Urgente** 
                   - Consultation cardiologique dans les 7 jours
                   - Bilan cardiaque complet: ECG, échocardiographie, épreuve d'effort
                   - Examens complémentaires: Holter ECG, coronarographie si indiqué
                
                2. **Surveillance Renforcée**
                   - Mesure quotidienne de la pression artérielle (matin et soir)
                   - Bilan lipidique trimestriel (LDL, HDL, triglycérides)
                   - Suivi pondéral hebdomadaire et calcul IMC
                
                3. **Modifications du Mode de Vie**
                   - Régime méditerranéen strict (réduction lipides saturés)
                   - Activité physique modérée: 30 min/jour, 5 jours/semaine
                   - Techniques de gestion du stress (relaxation, cohérence cardiaque)
                   - Sevrage tabagique impératif et limitation alcool
                
                4. **Traitement Pharmacologique**
                   - Respect strict de l'ordonnance médicale
                   - Statines, antihypertenseurs selon prescription
                   - Suivi des effets secondaires
                   - Aucune modification thérapeutique sans avis médical
                
                ⚠️ **Clause de non-responsabilité:** Cette analyse prédictive est un outil d'aide à la décision 
                et ne remplace en aucun cas un diagnostic médical établi par un professionnel de santé qualifié.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='color: #ffffff;'>
                <h2>✅ Profil Cardiovasculaire dans les Normes</h2>
                
                <p><b>Score de santé cardiovasculaire: {probability[0]:.1%}</b></p>
                
                <p>L'analyse des paramètres cliniques indique un profil cardiovasculaire satisfaisant 
                avec un risque faible de pathologie cardiaque à court terme.</p>
                
                <h3>🌟 Recommandations Préventives:</h3>
                
                <p><b>1. Prévention Primaire</b></p>
                <ul>
                <li>Maintien du poids optimal (IMC 18.5-24.9)</li>
                <li>Activité physique régulière et progressive</li>
                <li>Alimentation équilibrée: fruits, légumes, poissons gras</li>
                <li>Hydratation adéquate (1.5-2L/jour)</li>
                </ul>
                
                <p><b>2. Surveillance Préventive</b></p>
                <ul>
                <li>Bilan de santé annuel systématique</li>
                <li>Contrôle tensionnel semestriel</li>
                <li>Bilan lipidique annuel après 40 ans</li>
                <li>Suivi glycémique si facteurs de risque</li>
                </ul>
                
                <p><b>3. Hygiène de Vie Optimale</b></p>
                <ul>
                <li>Sommeil réparateur: 7-8 heures/nuit</li>
                <li>Gestion active du stress quotidien</li>
                <li>Évitement tabac et modération alcool</li>
                <li>Limitation exposition pollution et toxiques</li>
                </ul>
                
                <p><b>4. Vigilance Continue</b></p>
                <ul>
                <li>Attention aux signaux d'alarme (douleur thoracique, dyspnée)</li>
                <li>Consultation rapide si symptômes nouveaux</li>
                <li>Information du médecin traitant en cas de changement</li>
                <li>Mise à jour régulière du dossier médical</li>
                </ul>
                
                <p>✨ <b>Félicitations!</b> Votre profil cardiovasculaire est favorable. 
                Maintenez ces bonnes habitudes de vie pour une santé optimale à long terme.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Graphique de jauge
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📈 Échelle d'Évaluation du Risque")
            
            risk_level = probability[1] * 100
            if risk_level < 30:
                gauge_color = "📍 Risque Faible"
                gauge_emoji = "✅"
                bar_color = "#3b82f6"
            elif risk_level < 60:
                gauge_color = "📍 Risque Modéré"
                gauge_emoji = "⚠️"
                bar_color = "#60a5fa"
            else:
                gauge_color = "📍 Risque Élevé"
                gauge_emoji = "🚨"
                bar_color = "#93c5fd"
            
            st.markdown(f"""
                <div style='background: #1e3a8a; padding: 30px; border-radius: 20px; 
                            text-align: center; border: 2px solid #3b82f6;'>
                    <h2 style='color: #ffffff; font-size: 3rem;'>{gauge_emoji}</h2>
                    <h3 style='color: #ffffff;'>Catégorie: {gauge_color}</h3>
                    <div style='background: #3b82f6; height: 40px; border-radius: 20px; 
                                margin: 20px 0; overflow: hidden;'>
                        <div style='background: #93c5fd; 
                                    width: {risk_level}%; height: 100%; border-radius: 20px;
                                    transition: width 1s ease;'></div>
                    </div>
                    <p style='color: #ffffff; font-size: 1.5rem; font-weight: 700;'>{risk_level:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Données d'entrée
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.expander("📋 Détail des Paramètres Analysés"):
                st.dataframe(input_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

# Sidebar professionnel
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 3rem;'>🏥</h1>
            <h2 style='color: white;'>Informations Système</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: #1e3a8a; padding: 20px; border-radius: 15px; 
                    border: 1px solid #3b82f6;'>
            <p style='color: white; line-height: 1.8;'>
                Système expert utilisant des <b>algorithmes de Machine Learning</b> pour 
                l'évaluation du risque cardiovasculaire.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Paramètres Cliniques")
    
    parameters = {
        "🫀 SBP": "Pression artérielle systolique",
        "🧪 LDL": "Cholestérol LDL (Low-Density Lipoprotein)",
        "📊 Adiposité": "Indice de composition corporelle",
        "⚖️ BMI": "Indice de masse corporelle",
        "📅 Âge": "Âge du patient",
        "🧬 Antécédents": "Historique familial cardiovasculaire"
    }
    
    for param, desc in parameters.items():
        st.markdown(f"""
            <div style='background: #1e3a8a; padding: 10px; margin: 10px 0; 
                        border-radius: 10px; border: 1px solid #3b82f6;'>
                <p style='color: white; margin: 0;'><b>{param}</b></p>
                <p style='color: #ffffff; font-size: 0.9rem; margin: 5px 0 0 0;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: #1e3a8a; padding: 20px; border-radius: 15px; 
                    border: 1px solid #3b82f6; text-align: center;'>
            <h3 style='color: #ffffff;'>⚠️ Avertissement Légal</h3>
            <p style='color: white; line-height: 1.6;'>
                Ce système est un <b>outil d'aide à la décision</b> uniquement. 
                Il ne remplace en aucun cas un diagnostic médical établi par un 
                professionnel de santé qualifié. Toute décision thérapeutique doit 
                être prise en consultation avec un médecin.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <p style='color: white; font-weight: 600;'>💻 Développé par</p>
            <h3 style='color: #ffffff;'>Josias DJAGBARE</h3>
            <p style='color: #ffffff;'>Élève Ingénieur</p>
            <p style='color: #ffffff;'>Modélisation et Informatique Scientifique</p>
            <p style='color: white; margin-top: 15px;'>🎓 EMI 2025-2026</p>
        </div>
    """, unsafe_allow_html=True)

# Section avec expander pour les explications
st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("📚 Guide des Paramètres Analysés"):
    st.markdown("""
        <div style='color: #ffffff;'>
        <p><b>🫀 Pression Artérielle Systolique (SBP)</b></p>
        <p>La pression artérielle systolique est la pression maximale du sang dans les artères 
        au moment de la contraction cardiaque. Une valeur normale est entre 90-120 mmHg. 
        Une pression élevée (hypertension) augmente le risque cardiovasculaire.</p>
        
        <p><b>🧪 Cholestérol LDL</b></p>
        <p>Le LDL (Low-Density Lipoprotein) est le "mauvais cholestérol" qui s'accumule dans 
        les artères. Un niveau optimal est inférieur à 100 mg/dL. Des niveaux élevés 
        favorisent l'athérosclérose et augmentent le risque de maladie cardiaque.</p>
        
        <p><b>⚖️ Indice de Masse Corporelle (BMI)</b></p>
        <p>L'IMC mesure le rapport entre le poids et la taille. Les catégories sont:</p>
        <ul>
        <li>Normal: 18.5-24.9</li>
        <li>Surpoids: 25-29.9</li>
        <li>Obésité: ≥30</li>
        </ul>
        <p>Un IMC élevé est associé à un risque cardiovasculaire accru.</p>
        
        <p><b>📊 Indice d'Adiposité</b></p>
        <p>L'adiposité mesure la proportion de tissu adipeux dans le corps. 
        Elle est souvent utilisée pour évaluer la composition corporelle de manière plus 
        précise que l'IMC seul et contribue à l'évaluation du risque métabolique.</p>
        
        <p><b>📅 Âge</b></p>
        <p>L'âge est un facteur de risque non modifiable pour les maladies cardiovasculaires. 
        Le risque augmente généralement avec l'âge, particulièrement après 40-50 ans pour les hommes 
        et après 55 ans pour les femmes.</p>
        
        <p><b>🧬 Antécédents Familiaux</b></p>
        <p>La présence d'antécédents familiaux de maladies cardiaques augmente significativement 
        le risque personnel. Si un parent proche a eu une maladie cardiaque, vous êtes à plus haut risque 
        et devez être plus vigilant.</p>
        </div>
    """, unsafe_allow_html=True)

# Avertissement final important
with st.expander("⚠️ AVERTISSEMENT IMPORTANT"):
    st.markdown("""
        <div style='color: #ffffff;'>
        <p><b>⚠️ Cette application NE REMPLACE EN AUCUN CAS UN DIAGNOSTIC MÉDICAL</b></p>
        
        <p>Ce système utilise des <b>algorithmes de Machine Learning</b> pour fournir une évaluation 
        prédictive basée sur les données que vous fournissez. Les résultats sont informatifs et 
        destinés à vous aider à prendre conscience de vos facteurs de risque cardiovasculaire.</p>
        
        <p><b>⚕️ Responsabilités médicales:</b></p>
        <ul>
        <li>Seul un médecin qualifié peut diagnostiquer une maladie cardiaque</li>
        <li>Toute décision thérapeutique doit être prise en consultation avec un professionnel de santé</li>
        <li>Ne pas utiliser cette application comme substitut à des examens médicaux professionnels</li>
        <li>Consultez immédiatement un médecin en cas de symptômes cardiovasculaires (douleur thoracique, essoufflement, etc.)</li>
        </ul>
        
        <p><b>💡 Recommandation:</b></p>
        <p>Utilisez cette application comme point de départ pour une conversation avec votre médecin 
        sur votre santé cardiovasculaire, notamment si vous avez des facteurs de risque identifiés ici.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer professionnel
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 20px; background: #1e3a8a; 
                border-radius: 15px; border: 2px solid #3b82f6;'>
        <p style='color: #ffffff; margin: 0; font-weight: 600;'>
            🔬 Système d'Intelligence Artificielle Médicale
        </p>
        <p style='color: #ffffff; font-size: 0.9rem; margin-top: 10px;'>
            Propulsé par Streamlit & scikit-learn | © 2025 Josias DJAGBARE
        </p>
    </div>
""", unsafe_allow_html=True)