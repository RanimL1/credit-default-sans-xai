import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.api_client import url_api
from core.api_embarquee import demarrer_api_si_besoin

st.set_page_config(page_title="Classification du défaut de paiement", page_icon="📊", layout="wide")

st.title("📊 Classification du défaut de paiement")
st.markdown(
    """
Application interactive de **scoring de crédit** (Data Mining, ENSI).
Elle charge un jeu de données, entraîne et compare cinq modèles de classification, puis évalue
leurs performances.

**Parcours (menu de gauche) :**
1. **Données** : téléchargement automatique (UCI) ou import CSV, exploration
2. **Préparation** : variables, valeurs manquantes, encodage, normalisation, split
3. **Modèles** : régression logistique, arbre, kNN, Random Forest, XGBoost (+ optimisation)
4. **Évaluation** : métriques, courbes ROC/PR, matrice de confusion, curseur de seuil
"""
)

with st.sidebar:
    st.subheader("API de prédiction")
    with st.spinner("Démarrage de l'API (quelques secondes)..."):
        api_prete, erreur_api = demarrer_api_si_besoin()
    if api_prete:
        st.success("API connectée")
    else:
        st.warning("API injoignable : la prédiction par API est indisponible, "
                   "mais les pages de l'application restent utilisables.")
        if erreur_api:
            st.caption(erreur_api)
    st.caption(url_api())

if "df" in st.session_state:
    st.info(f"Jeu de données chargé : {st.session_state['df'].shape[0]} lignes, "
            f"{st.session_state['df'].shape[1]} colonnes.")
