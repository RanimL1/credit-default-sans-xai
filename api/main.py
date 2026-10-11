from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from api import modeles
from api.schemas import Prediction, Requete

app = FastAPI(
    title="API de prédiction - Défaut de paiement (Taiwan)",
    description=(
        "Prédit la probabilité de défaut de paiement d'un client de carte bancaire "
        "avec l'un des modèles entraînés."
    ),
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def racine():
    return RedirectResponse(url="/docs")


def _verifier_modele(nom):
    if nom not in modeles.modeles_disponibles():
        raise HTTPException(
            status_code=404,
            detail=f"Modèle inconnu : '{nom}'. Disponibles : {modeles.modeles_disponibles()}",
        )


@app.get("/health", tags=["Service"])
def health():
    return {"status": "ok"}


@app.get("/models", tags=["Service"])
def models():
    return {
        "modeles": modeles.modeles_disponibles(),
        "variables": modeles.COLONNES,
    }


@app.post("/predict", response_model=Prediction, tags=["Prédiction"])
def predict(req: Requete):
    _verifier_modele(req.modele)
    p = modeles.probabilite_defaut(req.modele, req.client.model_dump())
    return Prediction(
        modele=req.modele,
        probabilite_defaut=p,
        seuil=req.seuil,
        decision="defaut_probable" if p >= req.seuil else "pas_de_defaut",
    )
