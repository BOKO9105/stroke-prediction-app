import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, recall_score, precision_score
import seaborn as sns
import numpy as np

# Configuration de la page (DOIT ÊTRE LA PREMIÈRE COMMANDE)
st.set_page_config(page_title="Prédiction d'AVC", layout="wide")

@st.cache_resource
def load_model_and_data():
    try:
        # Chemin relatif pour les assets
        base_path = os.path.dirname(__file__)
        
        # Charger le modèle et le scaler
        model_path = os.path.join(base_path, 'assets', 'modele_logreg_5var_avc.pkl')
        scaler_path = os.path.join(base_path, 'assets', 'scaler_5var.pkl')
        data_path = os.path.join(base_path, 'assets', 'healthcare-dataset-stroke-data.csv')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        data = pd.read_csv(data_path)
        
        # Prétraitement des données
        median_bmi = data['bmi'].median()
        data = data.assign(
            bmi=data['bmi'].fillna(median_bmi),
            smoking_status=data['smoking_status'].fillna('unknown')
        )
        
        # Séparation des features et target
        X = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
        y = data['stroke']
        
        return scaler, model, X, y
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des ressources : {str(e)}")
        return None, None, None, None

# Chargement des données
scaler, model, X, y = load_model_and_data()

if scaler is None or model is None:
    st.error("L'application ne peut pas charger les ressources nécessaires. Veuillez vérifier les fichiers dans le dossier 'assets'.")
    st.stop()

# Fonction de calcul d'IMC
def calculer_imc(taille_cm, poids_kg):
    taille_m = taille_cm / 100
    return poids_kg / (taille_m ** 2)


# Titre principal
st.title("📊 Modèle de Prédiction d'AVC")

# Navigation
page = st.sidebar.radio("Navigation", [
    "📈 Statistiques du Modèle", 
    "🔮 Simulation de Prédiction",
    "👤 À propos"
])

if page == "📈 Statistiques du Modèle":
    st.header("Analyse Statistique des Facteurs d'AVC")
    
    # Section 1: Statistiques descriptives bivariées
    st.subheader("1. Analyse Bivariée")
    
    # Charger les données complètes
    data = pd.read_csv('assets/healthcare-dataset-stroke-data.csv')
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())
    
    # A. Variables quantitatives
    st.markdown("**Variables Quantitatives**")
    quant_vars = ['age', 'avg_glucose_level', 'bmi']
    
    # Tableau de comparaison
    stats_df = data.groupby('stroke')[quant_vars].agg(['mean', 'std'])
    st.dataframe(stats_df.style.format("{:.2f}"))
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, var in enumerate(quant_vars):
        sns.boxplot(x='stroke', y=var, data=data, ax=axes[i])
        axes[i].set_title(f'Distribution de {var}')
    st.pyplot(fig)
    
    # B. Variables catégorielles
    st.markdown("**Variables Catégorielles**")
    cat_vars = ['hypertension', 'heart_disease', 'gender', 'smoking_status']
    
    for var in cat_vars:
        st.write(f"#### {var}")
        ct = pd.crosstab(data[var], data['stroke'], normalize='index')*100
        st.dataframe(ct.style.format("{:.1f}%"))
        
        # Test du Chi2
        chi2, p, _, _ = chi2_contingency(pd.crosstab(data[var], data['stroke']))
        st.write(f"Test du Chi²: p-value = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    
    # Section 2: Performance du modèle
    st.subheader("2. Performance du Modèle")
    
    # Préparation des données pour le modèle
    X = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']]
    y = data['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    y_pred = model.predict(X_test)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Sensibilité", f"{recall_score(y_test, y_pred):.2%}")
    col3.metric("Spécificité", f"{precision_score(y_test, y_pred):.2%}")
    
    # Courbe ROC
    st.markdown("**Courbe ROC**")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    st.pyplot(fig)
    
    # Coefficients
    st.markdown("**Variables Influentes**")
    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model.coef_[0],
        'Odds Ratio': np.exp(model.coef_[0])
    }).sort_values('Odds Ratio', ascending=False)
    st.dataframe(coef_df.style.format({'Odds Ratio': '{:.2f}'}))

elif page == "🔮 Simulation de Prédiction":
    st.header("Simulateur de Risque d'AVC")
    
    with st.form("formulaire_avc"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", 1, 100, 50)
            taille_cm = st.number_input("Taille (cm)", 100, 250, 170)
            glucose = st.number_input("Taux de glucose moyen (mg/dL)", 50, 300, 100)
        with col2:
            poids_kg = st.number_input("Poids (kg)", 30, 200, 70)
            hypertension = st.checkbox("Hypertension artérielle")
            heart_disease = st.checkbox("Maladie cardiaque")
        
        submitted = st.form_submit_button("Calculer le risque")

    if submitted:
        # Calcul automatique de l'IMC
        bmi = calculer_imc(taille_cm, poids_kg)
        # Préparation des données
        input_data = {
            'age': [age],
            'hypertension': [int(hypertension)],
            'heart_disease': [int(heart_disease)],
            'avg_glucose_level': [glucose],
            'bmi': [bmi]
        }
        
        input_df = pd.DataFrame(input_data)
        input_scaled = scaler.transform(input_df)
        
        # Prédiction
        proba = model.predict_proba(input_scaled)[0][1]

        # Affichage des résultats
        st.subheader("🧮 Résultat de la Prédiction")
        if proba > 0.7:
            couleur = "red"
            message = "🔴 Risque élevé - Consultation urgente recommandée"
        elif proba > 0.3:
            couleur = "orange"
            message = "🟠 Risque modéré - Surveillance médicale conseillée"
        else:
            couleur = "green"
            message = "🟢 Risque faible - Continuez vos bonnes habitudes"

        st.metric("Probabilité d'AVC", f"{proba*100:.1f}%")
        st.progress(int(proba * 100))
        st.markdown(f"<p style='color:{couleur}; font-size:18px'>{message}</p>", unsafe_allow_html=True)

        # Analyse des facteurs
        st.subheader("🔍 Facteurs de Risque Détectés")
        facteurs = []
        if age > 60: facteurs.append(f"Âge élevé ({age} ans)")
        if hypertension: facteurs.append("Hypertension artérielle")
        if heart_disease: facteurs.append("Maladie cardiaque")
        if glucose > 140: facteurs.append(f"Hyperglycémie ({glucose} mg/dL)")
        if bmi > 30: facteurs.append(f"IMC élevé ({bmi})")

        if facteurs:
            st.write("Facteurs identifiés :")
            for f in facteurs:
                st.write(f"- {f}")
        else:
            st.info("Aucun facteur de risque majeur détecté")

else:
    st.header("👤 À propos de ce projet")
    st.markdown("""
    ### Auteur : Bidossessi BOKO  
    📧 **Email :** boko.rodrigue@yahoo.fr  
    📍 **Localisation :** Rennes, France  

    ---
    ### 🎓 Projet : Prédiction du risque d'AVC  
    Application développée avec Streamlit utilisant un modèle de régression logistique pour évaluer le risque d'accident vasculaire cérébral.

    ---
    ### 🛠️ Technologies utilisées
    - Python (scikit-learn, pandas)
    - Streamlit pour l'interface
    - Modélisation prédictive

    ---
    *Pour toute question ou collaboration, n'hésitez pas à me contacter.*
    """)

# Instructions dans la sidebar
with st.sidebar:
    st.info("""
    **Instructions :**
    1. Remplissez le formulaire de prédiction
    2. Cliquez sur "Calculer le risque"
    3. Consultez les résultats et recommandations
    
    **Seuils de risque :**
    - 🔴 > 70% : Risque élevé
    - 🟠 30-70% : Risque modéré
    - 🟢 < 30% : Risque faible
    """)