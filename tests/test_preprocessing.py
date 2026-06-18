from src.features.text_preprocessing import preprocess_text


def test_lowercase():
    result = preprocess_text("Hello World")
    assert result == "hello world"


def test_html_tag_removal():
    result = preprocess_text("<p>How are you</p>")

    assert "<p>" not in result
    assert "</p>" not in result


def test_remove_non_letters():
    result = preprocess_text("what are2 you do3ing")

    assert "2" or "3" not in result


def test_lematization():
    result = preprocess_text("Cars")

    assert result == "car"
