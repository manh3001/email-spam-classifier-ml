"""Text cleaning used before vectorization."""

import re

_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def clean_text(text: str) -> str:
    """Lowercase and strip characters that are not letters, digits, or spaces."""
    text = str(text).lower()
    return _NON_ALNUM.sub("", text)
