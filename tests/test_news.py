import pandas as pd

from src.news import normalize_yahoo_news


def test_normalizes_nested_and_flat_yahoo_records():
    records = [
        {
            "content": {
                "title": "Tesla raises its delivery forecast",
                "summary": "Guidance update",
                "pubDate": "2026-03-01T10:00:00Z",
                "provider": {"displayName": "Example News"},
                "clickThroughUrl": {"url": "https://example.com/nested"},
            }
        },
        {
            "title": "Markets close unchanged",
            "providerPublishTime": 1772362800,
            "publisher": "Legacy Feed",
            "link": "https://example.com/flat",
        },
    ]

    result = normalize_yahoo_news(records, "TSLA", company_terms=["Tesla"])

    relevance_by_title = result.set_index("title")["relevance"].to_dict()
    assert relevance_by_title == {
        "Tesla raises its delivery forecast": "direct",
        "Markets close unchanged": "context",
    }
    assert set(result["publisher"]) == {"Example News", "Legacy Feed"}
    assert str(result["published_at"].dtype) == "datetime64[ns, UTC]"


def test_skips_empty_titles_and_deduplicates():
    records = [
        {"content": {"title": "Repeated headline"}},
        {"content": {"title": "Repeated headline"}},
        {"content": {"summary": "No title"}},
    ]

    result = normalize_yahoo_news(records, "TEST")

    assert len(result) == 1
    assert result.iloc[0]["title"] == "Repeated headline"
    assert pd.isna(result.iloc[0]["published_at"])
