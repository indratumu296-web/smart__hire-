from src.features.text_features import tokenize


def test_tokenize_returns_lowercase_tokens():
    assert tokenize("Python SQL") == ["python", "sql"]
