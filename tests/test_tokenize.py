import unicodedata

from findex1.tokenizer import tokenize


def test_tokenize_empty_string():

    assert list(tokenize("")) == []


def test_tokenize_mixed_case():

    text = "Hello World PyThOn"
    assert list(tokenize(text)) == ["hello", "world", "python"]


def test_tokenize_cyrillic_text():

    text = "Тестовий корпус ТЕКСТІВ для Лабораторної"
    assert list(tokenize(text)) == [
        "тестовий",
        "корпус",
        "текстів",
        "для",
        "лабораторної",
    ]


test_tokenize_combining_mark_accent_data = (
    "e" + unicodedata.lookup("COMBINING ACUTE ACCENT")
)


def test_tokenize_combining_mark_accent():

    text = f"caf{test_tokenize_combining_mark_accent_data}"
    tokens = list(tokenize(text))
    assert tokens == ["café"]
    assert unicodedata.is_normalized("NFC", tokens[0])


def test_tokenize_punctuation_and_digits():

    text = "Hello, world! Price is 100$ (test_ver 2.0)."
    assert list(tokenize(text)) == ["hello", "world", "price", "is", "test_ver"]


def test_tokenize_apostrophes_and_hyphens():

    text = "об'єм don't word-level будь-ласка"
    assert list(tokenize(text)) == [
        "об'єм",
        "don't",
        "word-level",
        "будь-ласка",
    ]


def test_tokenize_whitespaces_and_newlines():

    text = "Line1 \n\t Line2   \r\n Line3"
    assert list(tokenize(text)) == ["line1", "line2", "line3"]