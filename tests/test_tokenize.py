from src.data.preprocess import encode_chars, encode_words, normalize_label


def test_encode_words_shape():
    vocab = {"<PAD>": 0, "<OOV>": 1, "hello": 2, "world": 3}
    out = encode_words("hello world", vocab, max_words=3)
    assert len(out) == 3
    assert out[0] == 2
    assert out[1] == 3
    assert out[2] == 0


def test_encode_chars_shape():
    vocab = {"<PAD>": 0, "<OOV>": 1, "h": 2, "i": 3}
    out = encode_chars("h i", vocab, max_chars=4)
    assert len(out) == 4
    assert out[0] == 2
    assert out[1] == 3
    assert out[2] == 0


def test_normalize_label_filters_missing_strings():
    assert normalize_label("None") is None
    assert normalize_label("") is None
    assert normalize_label("St. Louis") == "St Louis"
