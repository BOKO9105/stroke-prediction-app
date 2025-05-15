import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split


# MUST BE THE FIRST STREAMLIT COMMAND (and only one)
st.set_page_config(page_title="Prédiction d'AVC", layout="wide")

@st.cache_resource
def load_model_and_data():
    # Load model and data
    model = joblib.load("assets/modele_logreg_5var_avc.pkl")
    scaler = joblib.load("assets/scaler_5var.pkl")
    data = pd.read_csv('assets/healthcare-dataset-stroke-data.csv')
    
    # Preprocessing
    median_bmi = data['bmi'].median()
    data = data.assign(
        bmi=data['bmi'].fillna(median_bmi),
        smoking_status=data['smoking_status'].fillna('unknown')
    )
    
    # Separate features and target
    X = data.drop('stroke', axis=1)
    y = data['stroke']
    
    # Get the feature names the model expects
    feature_names = model.feature_names_in_
    
    return scaler, model, X, y, feature_names

# Load data
scaler, model, X, y, feature_names = load_model_and_data()

# Title
st.title("📊 Modèle de Prédiction d'AVC")

# Navigation
page = st.sidebar.radio("Navigation", [
    "📈 Statistiques du Modèle", 
    "🔮 Simulation de Prédiction",
    "👤 À propos"
    ])

if page == "📈 Statistiques du Modèle":
    st.header("Performances du Modèle Logistique")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Prepare test data with same features as model expects
    X_test_processed = pd.get_dummies(X_test)
    missing_cols = set(feature_names) - set(X_test_processed.columns)
    for col in missing_cols:
        X_test_processed[col] = 0
    X_test_processed = X_test_processed[feature_names]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", "0.839", "Bonne discrimination")
    col2.metric("Sensibilité", "78.7%", "Détection des AVC")
    col3.metric("Spécificité", "73.2%", "Exclusion des non-AVC")
    
    # ROC Curve
    st.subheader("Courbe ROC")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test_processed, y_test, ax=ax)
    st.pyplot(fig)
    
    # Model coefficients
    st.subheader("Variables Influentes")
    coefficients = pd.DataFrame({
        'Variable': ['Âge', 'Hypertension', 'Maladie cardiaque', 'Glucose', 'IMC'],
        'Coefficient': [1.902, 0.529, 0.289, 0.018, 0.042],
        'Odds Ratio': [6.70, 1.70, 1.34, 1.02, 1.04]
    })
    st.dataframe(coefficients.style.format({'Odds Ratio': '{:.2f}'}))
    
    # Clinical interpretation
    st.subheader("Points Clés Cliniques")
    st.markdown("""
    - **Âge** : Facteur de risque majeur (OR=6.70 par année)
    - **Hypertension** : Triple le risque d'AVC
    - **Hyperglycémie** : Chaque augmentation de 50mg/dL → +27% de risque
    """)

elif page == "🔮 Simulation de Prédiction":

    st.header("Simulateur de Risque d'AVC")
    
    with st.form("formulaire_5var"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge", 1, 100, 50)
            hypertension = st.checkbox("Hypertension artérielle")
            heart_disease = st.checkbox("Maladie cardiaque")
        with col2:
            glucose = st.number_input("Taux de glucose moyen (mg/dL)", 50, 300, 100)
            bmi = st.number_input("IMC", 10.0, 60.0, 25.0)

        submitted = st.form_submit_button("Calculer le risque")

    if submitted:
        # Préparation des données utilisateur
        input_dict = {
            'age': age,
            'hypertension': int(hypertension),
            'heart_disease': int(heart_disease),
            'avg_glucose_level': glucose,
            'bmi': bmi
        }
        input_df = pd.DataFrame([input_dict])
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)


        # Prédiction
        proba = model.predict_proba(input_scaled)[0][1]

        # Affichage
        st.subheader("🧮 Résultat de la Prédiction")
        if proba > 0.7:
            couleur = "red"
            message = "🔴 Risque élevé – consultez un professionnel rapidement"
        elif proba > 0.3:
            couleur = "orange"
            message = "🟠 Risque modéré – une surveillance médicale est conseillée"
        else:
            couleur = "green"
            message = "🟢 Risque faible – maintien des bonnes pratiques recommandé"

        st.metric("Probabilité d'AVC", f"{proba*100:.1f} %")
        st.progress(int(proba * 100))
        st.markdown(f"<p style='color:{couleur}; font-size:18px'>{message}</p>", unsafe_allow_html=True)

        # Analyse des facteurs
        st.subheader("🔍 Facteurs de Risque Détectés")
        facteurs = []
        if age > 60: facteurs.append(f"Âge élevé ({age})")
        if hypertension: facteurs.append("Hypertension")
        if heart_disease: facteurs.append("Maladie cardiaque")
        if glucose > 140: facteurs.append(f"Hyperglycémie ({glucose} mg/dL)")
        if bmi > 30: facteurs.append(f"IMC élevé (obésité, {bmi})")

        if facteurs:
            st.markdown("Ce patient présente :")
            for f in facteurs:
                st.write(f"- {f}")
        else:
            st.info("Aucun facteur de risque majeur détecté.")
else:
    st.header("👤 À propos de ce projet")

    st.markdown("""
    ### Auteur : Bidossessi BOKO  
    📧 **Email :** boko.rodrigue@yahoo.fr  
    📍 **Localisation :** Rennes, France  

    ---
    ### 🎓 Projet : Prédiction du risque d’AVC  
    Ce projet s'inscrit dans une démarche de valorisation des données de santé à des fins préventives. Il utilise une régression logistique basée sur 5 variables cliniques simples pour prédire la probabilité d’AVC.

    ---
    ### 🎯 Objectif professionnel  
    Passionné par la data science appliquée à la santé publique, je vise à :
    - Poursuivre un **Master 2 en Data Science en Santé Publique**
    - Développer un projet de **doctorat en Intelligence Artificielle pour la Santé**
    - Contribuer à des outils de **prévention, de dépistage et de décision clinique** au service des patients et des systèmes de santé.

    ---
    ### 🛠️ Technologies utilisées
    - Python (scikit-learn, pandas, matplotlib)
    - Streamlit
    - Modélisation supervisée
    - Interface utilisateur interactive

    ---
    *Merci de votre intérêt pour ce projet !*
    """)


# Sidebar instructions
with st.sidebar:
    st.markdown("""
    **Instructions:**
    1. Remplissez le formulaire
    2. Cliquez sur "Calculer le risque"
    3. Consultez les résultats
    
    **Seuils de risque:**
    - 🔴 > 70%: Risque élevé
    - 🟠 30-70%: Risque modéré
    - 🟢 < 30%: Risque faible
    """)