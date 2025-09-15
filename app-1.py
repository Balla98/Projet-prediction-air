import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
from tensorflow import keras

# 1. Chargement des modèles

deep_model = keras.models.load_model("deep_model.keras")
causal_forest = joblib.load("causal_forest.pkl")
scaler = joblib.load("scaler.pkl")

# Charger les noms des features 
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)  

# 2. Interface utilisateur

st.set_page_config(page_title="Prédicteur de la qualité de l’air", layout="centered")

st.title("Application de prédiction de la qualité de l’air")
st.write("""
Cette application utilise une **forêt causale** combinée à un **modèle d'apprentissage profond** 
pour prédire les niveaux de pollution à partir des conditions environnementales.
""")

# Récupère l'ordre exact utilisé par le scaler
base_features = list(getattr(scaler, "feature_names_in_", feature_names))

st.sidebar.header("Paramètres d'entrée")

# Entrée des variables explicatives
user_inputs = {}
for feature in base_features:
    user_inputs[feature] = st.sidebar.number_input(f"Enter {feature}", min_value=0.0, max_value=5000.0, value=0.0)

# Champ pour T (si ton DL en a besoin)
T_val = st.sidebar.number_input("Enter T (traitement, ex: température)", value=20.0)

# 3. Préparation des données

# DataFrame avec SEULEMENT les features du scaler, dans le bon ordre
X_base_df = pd.DataFrame([[user_inputs[f] for f in base_features]], columns=base_features)

# Mise à l'échelle uniquement sur les features vues au fit
Xs = scaler.transform(X_base_df)
# CATE à partir de la forêt causale (sur les features scalées)
cate = float(causal_forest.effect(Xs)[0])

# Construction de l'entrée du modèle DL
X_dl_input = np.hstack([Xs])#[[T_val]], [[cate]]

# Choisir automatiquement celle qui matche la dimension d'entrée du modèle
n_expected = deep_model.input_shape[1]
if X_dl_input.shape[1] != n_expected:
    st.error(f"Erreur : le modèle attend {n_expected} features, mais {X_dl_input.shape[1]} ont été fournis.")
    st.stop()

# 4. Prédictions

if st.button("Prédire"):
    # Prédiction avec le modèle deep learning
    prediction = deep_model.predict(X_dl_input,verbose=0)[0][0]
    
    # Affichage des résultats
    st.subheader("Résultats de la prédiction")
    st.metric(label="Prédiction de la pollution: ", value=f"{prediction:.4f}")
    st.write(f"**Effet de traitement (CATE)** : `{cate:.4f}`")#{cate[0][0]:.4f}
    st.write(f"**Effet moyen (ATE)** :{float(causal_forest.ate(Xs)):.4f}")

    st.success("Prédiction terminée avec succès.")

    st.markdown("---")
    st.caption("Modèle basé sur : Forêt Causale (EconML) + Réseau de Neurones (Keras)")
