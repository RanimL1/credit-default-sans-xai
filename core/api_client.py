import os

import requests


def url_api():
    try:
        import streamlit as st
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def api_disponible():
    try:
        return requests.get(f"{url_api()}/health", timeout=8).json().get("status") == "ok"
    except Exception:
        return False


def liste_modeles():
    r = requests.get(f"{url_api()}/models", timeout=30)
    r.raise_for_status()
    return r.json()
