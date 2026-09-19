"""Inferencia de sentimiento financiero con dependencias inyectables."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


VALID_LABELS = {"positive", "negative", "neutral"}
SIGNED_LABEL = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def load_finbert(model_name: str = "ProsusAI/finbert", device: int = -1) -> Any:
    """Carga FinBERT bajo demanda para evitar descargas durante los tests."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depende del entorno del usuario
        raise RuntimeError(
            "Falta transformers. Instala las dependencias con "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return pipeline("text-classification", model=model_name, device=device)


def make_google_translator() -> Callable[[str], str]:
    """Crea un traductor externo opcional; no se usa por defecto."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError as exc:  # pragma: no cover - depende del entorno del usuario
        raise RuntimeError(
            "Falta deep-translator. Instala las dependencias completas."
        ) from exc
    translator = GoogleTranslator(source="auto", target="en")
    return translator.translate


def _run_classifier(classifier: Any, texts: list[str]) -> list[dict[str, Any]]:
    try:
        return classifier(texts, truncation=True, max_length=512)
    except TypeError:
        # Facilita pruebas con clasificadores simples inyectados.
        return classifier(texts)


def analyze_headlines(
    frame: pd.DataFrame,
    classifier: Any | None = None,
    translator: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Añade etiqueta y score del modelo sin tratarlos como verdad observada."""
    if "title" not in frame.columns:
        raise ValueError("El DataFrame debe contener una columna `title`.")
    if frame.empty:
        return frame.assign(
            analyzed_text=pd.Series(dtype="object"),
            translated=pd.Series(dtype="bool"),
            model_label=pd.Series(dtype="object"),
            model_score=pd.Series(dtype="float64"),
            signed_score=pd.Series(dtype="float64"),
        )

    titles = frame["title"].fillna("").astype(str).tolist()
    analyzed = [translator(text) for text in titles] if translator else titles
    model = classifier or load_finbert()
    predictions = _run_classifier(model, analyzed)
    if len(predictions) != len(frame):
        raise ValueError("El clasificador no devolvió una predicción por titular.")

    labels: list[str] = []
    scores: list[float] = []
    for prediction in predictions:
        label = str(prediction["label"]).lower()
        if label not in VALID_LABELS:
            raise ValueError(f"Etiqueta no reconocida: {label}")
        labels.append(label)
        scores.append(float(prediction["score"]))

    result = frame.copy()
    result["analyzed_text"] = analyzed
    result["translated"] = translator is not None
    result["model_label"] = labels
    result["model_score"] = scores
    result["signed_score"] = [SIGNED_LABEL[label] * score for label, score in zip(labels, scores)]
    return result


def sentiment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume conteos y scores sin inferir dirección futura del precio."""
    required = {"model_label", "model_score"}
    if not required.issubset(frame.columns):
        raise ValueError("Primero ejecuta `analyze_headlines`.")
    if frame.empty:
        return pd.DataFrame(columns=["model_label", "headlines", "mean_model_score"])
    return (
        frame.groupby("model_label", as_index=False)
        .agg(headlines=("model_label", "size"), mean_model_score=("model_score", "mean"))
        .sort_values("headlines", ascending=False)
        .reset_index(drop=True)
    )

