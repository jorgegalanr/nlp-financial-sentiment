"""Ejecuta el pipeline sobre una muestra local o noticias actuales de Yahoo."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.news import fetch_yahoo_news
from src.sentiment import analyze_headlines, make_google_translator, sentiment_summary


COLORS = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}


def save_plot(results: pd.DataFrame, output: Path) -> None:
    counts = results["model_label"].value_counts().reindex(
        ["positive", "neutral", "negative"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, color=[COLORS[label] for label in counts.index])
    ax.set_title("Distribución de etiquetas generadas por FinBERT")
    ax.set_xlabel("Etiqueta del modelo")
    ax.set_ylabel("Número de titulares")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="CSV local con una columna `title`.")
    source.add_argument("--ticker", help="Ticker del que descargar noticias actuales.")
    parser.add_argument(
        "--company-term",
        action="append",
        default=[],
        help="Término que identifica una noticia directa; se puede repetir.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Traduce todos los titulares al inglés mediante un servicio externo.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/sentiment_results.csv"))
    parser.add_argument("--plot", type=Path, default=Path("reports/sentiment_distribution.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        news = pd.read_csv(args.input)
    else:
        ticker = args.ticker or "TSLA"
        terms = args.company_term or (["Tesla"] if ticker.upper() == "TSLA" else [])
        news = fetch_yahoo_news(ticker, company_terms=terms)

    if news.empty:
        raise SystemExit("No se encontraron titulares para analizar.")

    translator = make_google_translator() if args.translate else None
    results = analyze_headlines(news, translator=translator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    save_plot(results, args.plot)

    print(sentiment_summary(results).to_string(index=False))
    print(f"\nResultados: {args.output}")
    print(f"Gráfico: {args.plot}")


if __name__ == "__main__":
    main()

