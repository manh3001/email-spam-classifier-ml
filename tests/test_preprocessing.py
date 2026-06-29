from spam_classifier.preprocessing import clean_text


def test_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_removes_punctuation_keeps_spaces():
    assert clean_text("Win!!! Free $$$ money") == "win free  money"


def test_keeps_alphanumeric():
    assert clean_text("Call 0800 now") == "call 0800 now"


def test_idempotent():
    once = clean_text("Hey, THERE!!")
    assert clean_text(once) == once
