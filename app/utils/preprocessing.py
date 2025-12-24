# preprocessing.py
import re
from typing import Optional, List

# Simple text cleaning utilities used before vectorization / modeling.


URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#(\w+)")
NON_ALPHANUMERIC = re.compile(r"[^0-9a-zA-Z\s\u00C0-\u017F]")  # keep accented letters (basic)


def clean_text(text: str, lower: bool = True, remove_urls: bool = True,
               remove_mentions: bool = True, remove_hashtags_symbol: bool = True,
               remove_punct: bool = True, collapse_spaces: bool = True) -> str:
    """
    Basic text cleaner:
      - optionally lowercase
      - remove URLs
      - remove mentions @user
      - optionally remove '#' symbol but keep the word (useful for hashtags)
      - remove punctuation (non alphanumeric)
      - collapse multiple spaces

    Args:
        text: input string
        lower: convert to lowercase
        remove_urls: strip URLs
        remove_mentions: strip @mentions
        remove_hashtags_symbol: remove '#' but keep tag text
        remove_punct: strip non-alphanumeric chars (keeps accented letters)
        collapse_spaces: collapse repeated whitespace to single space

    Returns:
        cleaned string
    """
    if not isinstance(text, str):
        text = str(text)

    s = text

    if remove_urls:
        s = URL_REGEX.sub("", s)

    if remove_mentions:
        s = MENTION_REGEX.sub("", s)

    if remove_hashtags_symbol:
        # convert "#tag" -> "tag"
        s = HASHTAG_REGEX.sub(r"\1", s)

    if lower:
        s = s.lower()

    if remove_punct:
        s = NON_ALPHANUMERIC.sub(" ", s)

    if collapse_spaces:
        s = re.sub(r"\s+", " ", s).strip()

    return s


def simple_tokenize(text: str, min_len: int = 1) -> List[str]:
    """
    Very simple whitespace tokenizer. Returns list of tokens with length >= min_len.
    """
    if not isinstance(text, str):
        text = str(text)
    tokens = text.split()
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    return tokens
