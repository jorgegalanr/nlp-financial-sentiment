import pandas as pd

from src.sentiment import analyze_headlines, sentiment_summary


class FakeClassifier:
    def __call__(self, texts, **kwargs):
        labels = ["positive", "negative", "neutral"]
        return [
            {"label": labels[index], "score": 0.9 - index * 0.1}
            for index, _ in enumerate(texts)
        ]


def test_analysis_keeps_model_score_separate_from_observed_truth():
    frame = pd.DataFrame({"title": ["Profit rises", "Liquidity falls", "Meeting held"]})

    result = analyze_headlines(frame, classifier=FakeClassifier())

    assert list(result["model_label"]) == ["positive", "negative", "neutral"]
    assert list(result["signed_score"]) == [0.9, -0.8, 0.0]
    assert not result["translated"].any()

    summary = sentiment_summary(result)
    assert summary["headlines"].sum() == 3


def test_translation_is_explicit_and_injectable():
    frame = pd.DataFrame({"title": ["Beneficio récord"]})
    translator = lambda text: "Record profit"

    result = analyze_headlines(
        frame,
        classifier=lambda texts, **kwargs: [{"label": "positive", "score": 0.95}],
        translator=translator,
    )

    assert result.loc[0, "analyzed_text"] == "Record profit"
    assert bool(result.loc[0, "translated"])

