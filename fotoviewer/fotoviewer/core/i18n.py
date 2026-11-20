import json
from functools import lru_cache
from pathlib import Path

from markupsafe import Markup

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
SUPPORTED_LANGUAGES = ("en", "de")
FALLBACK_LANGUAGE = "en"
JS_PREFIX = "js."


@lru_cache
def load_messages(language: str) -> dict[str, str]:
    with open(LOCALES_DIR / f"{language}.json", encoding="utf-8") as f:
        return json.load(f)


class Translator:
    """Looks up UI texts by key for one language, falling back to English and then to
    the key itself. Placeholders use str.format syntax ("{count} photo(s)").

    Keys ending in "_html" contain trusted markup from the locale files and are
    returned as Markup (placeholder values are still escaped); all other texts are
    plain strings that Jinja autoescapes as usual."""

    def __init__(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}, expected one of {SUPPORTED_LANGUAGES}")
        self.language = language
        self.messages = {**load_messages(FALLBACK_LANGUAGE), **load_messages(language)}

    def __call__(self, key: str, **params) -> str:
        text = self.messages.get(key, key)
        if key.endswith("_html"):
            return Markup(text).format(**params)
        return text.format(**params) if params else text

    def js_messages(self) -> dict[str, str]:
        """The subset of texts used by client-side JavaScript (keys prefixed "js.")."""
        return {key: text for key, text in self.messages.items() if key.startswith(JS_PREFIX)}
