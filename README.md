# Prédiction du défaut de paiement : application de classification et API

Projet **Challenge 1** du cours de Data Mining.
Cas métier choisi : **scoring de crédit** (cas n°1 du catalogue) sur le jeu *Default of Credit Card Clients* (Taiwan, UCI).

L'objectif est de prédire si un client de carte bancaire fera défaut le mois suivant, de comparer plusieurs modèles de classification sur des métriques adaptées au risque de crédit, et de choisir un seuil de décision orienté vers la détection des défauts.

## Données

- Source : jeu *Default of Credit Card Clients* de l'UCI (identifiant 350), téléchargé automatiquement avec `ucimlrepo`. L'import d'un CSV personnel est aussi possible dans l'application.
- 29 965 clients après suppression des doublons, 23 variables explicatives (limite de crédit, sexe, niveau d'études, situation familiale, âge, statuts de remboursement, montants de factures et de paiements sur six mois).
- Cible : défaut de paiement le mois suivant. Environ **22 %** de défauts : les classes sont déséquilibrées.
- Nettoyage : regroupement des codes non documentés de `EDUCATION` (0, 5, 6 vers « autre ») et de `MARRIAGE` (0 vers « autre »), suppression des doublons, renommage de `PAY_0` en `PAY_1`.

## Choix méthodologiques

- **Split stratifié** 80 / 20 (graine 42) : le test garde la proportion réelle de défauts, il n'est jamais rééquilibré.
- **Pas de fuite de données** : imputation, encodage et normalisation sont dans un `Pipeline` scikit-learn, ajustés sur le train uniquement.
- **Déséquilibre des classes** traité par des poids de classes (`class_weight`, `scale_pos_weight`), sans sur-échantillonnage.
- **Optimisation** des hyperparamètres par `RandomizedSearchCV`, validation croisée stratifiée sur le train, critère AUC.
- **Métriques** : AUC-ROC, PR-AUC, précision, rappel, F1 sur la classe « défaut », plus l'indice de Gini et la statistique de Kolmogorov-Smirnov (spécifiques au scoring de crédit).
- **Seuil de décision** : un défaut non détecté coûte plus cher qu'un bon client refusé à tort. Le projet privilégie donc le **rappel** sous contrainte d'une précision minimale. Ce choix repose sur cette hypothèse de coûts, que le curseur de seuil de l'application permet de faire varier. Le seuil est choisi en validation croisée sur le train, puis vérifié sur le test.

## Résultats

Modèles optimisés, évalués sur le jeu de test (seuil 0,5) :

| Modèle | AUC-ROC | PR-AUC |
|---|---|---|
| XGBoost | 0,775 | 0,557 |
| Random Forest | 0,774 | 0,554 |
| Arbre de décision | 0,754 | 0,516 |
| k plus proches voisins | 0,747 | 0,511 |
| Régression logistique | 0,719 | 0,503 |

Avec XGBoost (Gini 0,55, KS 0,42), le rappel est de 0,62 et la précision de 0,46 au seuil 0,5. En abaissant le seuil à 0,40, le rappel passe à 0,72 et la précision à 0,38 (43 % des clients signalés) : détecter davantage de défauts coûte plus de fausses alertes. Les modèles les plus performants sont les méthodes d'ensemble, mais l'écart avec la régression logistique (interprétable) reste modeste, ce qui nourrit la discussion performance contre interprétabilité.

## Structure du dépôt

```
api/                   API FastAPI (main.py, modeles.py, schemas.py)
app/                   Application Streamlit (Home.py et pages/)
core/                  Logique partagée (données, préparation, modèles, métriques, client de l'API)
models/                Modèles entraînés et découpage train/test (joblib)
notebooks/             Notebook d'analyse (données, préparation, modèles, évaluation)
requirements.txt       Dépendances de l'application et de l'API
test_api.py            Test rapide de l'API
```

## Installation et lancement

Prérequis : Python 3.10 ou plus récent.

```bash
git clone https://github.com/RanimL1/credit-default-sans-xai.git
cd credit-default-sans-xai
python -m venv .venv
# Windows : .venv\Scripts\Activate.ps1      Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

### Application Streamlit

```bash
python -m streamlit run app/Home.py
```

L'application s'ouvre sur http://localhost:8501. Parcours : Données, Préparation, Modèles, Évaluation.

L'application démarre l'API toute seule et affiche son état dans la barre latérale. Par défaut, elle cherche l'API sur `http://localhost:8000`. Pour utiliser une API hébergée ailleurs, définir la variable d'environnement `API_URL` (ou la clé `API_URL` dans `.streamlit/secrets.toml`). Si l'API n'est pas disponible, les pages de l'application restent utilisables.

### API FastAPI seule

Il est aussi possible de lancer l'API sans l'application, dans un terminal avec l'environnement activé :

```bash
python -m uvicorn api.main:app --reload --port 8000
```

La documentation interactive est sur http://localhost:8000/docs. Pour tester, avec l'API lancée : `python test_api.py`.

| Route | Rôle |
|---|---|
| `GET /health` | Vérifie que le service tourne |
| `GET /models` | Liste les modèles et les variables attendues |
| `POST /predict` | Probabilité de défaut et décision au seuil choisi |

Les modèles de `models/` ont été sauvegardés avec scikit-learn 1.6.1 et XGBoost 3.2.0 : `requirements.txt` fixe ces versions. Charger les modèles avec d'autres versions peut échouer ou donner des résultats différents.

## Limites

- Le jeu de données est ancien (Taïwan, 2005) et propre à un contexte bancaire : les résultats ne se transposent pas tels quels à un autre établissement.
- Le plafond de performance sur ce jeu est autour de 0,78 d'AUC dans la littérature : les gains supplémentaires viennent surtout de l'ingénierie de variables et du choix du seuil.
- Les variables sensibles (sexe, âge, situation familiale) font partie des données : leur usage dans un vrai système de crédit soulève des questions d'équité et de réglementation.

## Référence

Yeh, I-C., & Lien, C-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2).

## Auteur

Ranim Latrous, étudiante en 3e année, spécialité intelligence artificielle, ENSI.