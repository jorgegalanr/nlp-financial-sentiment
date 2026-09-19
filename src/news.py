"""Extracción y normalización de noticias financieras."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd


OUTPUT_COLUMNS = [
    "ticker",
    "published_at",
    "title",
    "summary",
    "publisher",
    "url",
    "relevance",
]


def _first_url(content: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    """Obtiene una URL tanto del formato nuevo como del antiguo de Yahoo."""
    for key in ("clickThroughUrl", "canonicalUrl"):
        candidate = content.get(key)
        if isinstance(candidate, Mapping) and candidate.get("url"):
            return str(candidate["url"])
        if isinstance(candidate, str) and candidate:
            return candidate
    return str(content.get("link") or record.get("link") or "")


def _published_at(content: Mapping[str, Any], record: Mapping[str, Any]) -> Any:
    value = content.get("pubDate") or record.get("providerPublishTime")
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _publisher(content: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    provider = content.get("provider")
    if isinstance(provider, Mapping):
        return str(provider.get("displayName") or provider.get("name") or "")
    return str(provider or record.get("publisher") or "")


def classify_relevance(
    title: str,
    ticker: str,
    company_terms: Sequence[str] = (),
) -> str:
    """Distingue menciones directas de noticias de contexto relacionadas."""
    normalized_title = title.casefold()
    terms = [ticker, *company_terms]
    return (
        "direct"
        if any(term and term.casefold() in normalized_title for term in terms)
        else "context"
    )


def normalize_yahoo_news(
    records: Iterable[Mapping[str, Any]],
    ticker: str,
    company_terms: Sequence[str] = (),
) -> pd.DataFrame:
    """Normaliza respuestas anidadas o planas de ``yfinance.Ticker.news``."""
    rows: list[dict[str, Any]] = []
    for record in records:
        nested = record.get("content")
        content = nested if isinstance(nested, Mapping) else record
        title = str(content.get("title") or "").strip()
        if not title:
            continue
        rows.append(
            {
                "ticker": ticker.upper(),
                "published_at": _published_at(content, record),
                "title": title,
                "summary": str(content.get("summary") or "").strip(),
                "publisher": _publisher(content, record),
                "url": _first_url(content, record),
                "relevance": classify_relevance(title, ticker, company_terms),
            }
        )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return (
        frame.drop_duplicates(subset="title")
        .sort_values("published_at", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def fetch_yahoo_news(
    ticker: str,
    company_terms: Sequence[str] = (),
) -> pd.DataFrame:
    """Descarga noticias asociadas a un ticker y devuelve un esquema estable."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depende del entorno del usuario
        raise RuntimeError(
            "Falta yfinance. Instala las dependencias con "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    records = yf.Ticker(ticker).news or []
    return normalize_yahoo_news(records, ticker=ticker, company_terms=company_terms)

