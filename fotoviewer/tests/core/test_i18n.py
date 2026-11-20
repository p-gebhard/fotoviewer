import re
from pathlib import Path

import pytest
from markupsafe import Markup
from pydantic import ValidationError

from fotoviewer.core.config import Settings
from fotoviewer.core.i18n import SUPPORTED_LANGUAGES, Translator, load_messages

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "fotoviewer" / "templates"


def test_all_languages_define_the_same_keys():
    reference = set(load_messages("en"))
    for language in SUPPORTED_LANGUAGES:
        assert set(load_messages(language)) == reference, language


def test_all_languages_use_the_same_placeholders():
    english = load_messages("en")
    for language in SUPPORTED_LANGUAGES:
        for key, text in load_messages(language).items():
            assert set(re.findall(r"\{(\w+)\}", text)) == set(re.findall(r"\{(\w+)\}", english[key])), (language, key)


def test_every_key_used_in_templates_exists():
    known = set(load_messages("en"))
    used = set()
    for template in TEMPLATES_DIR.glob("*.html"):
        used |= set(re.findall(r"""\bt\(\s*["']([\w.]+)["']""", template.read_text()))
    # JS builds phase keys dynamically ("js.phase." + status.phase)
    used = {key for key in used if not key.endswith(".")}
    assert used, "no translation keys found in templates"
    assert used <= known, used - known


def test_translates_and_fills_placeholders():
    assert Translator("de")("filter.count", count=3) == "3 Foto(s) gefunden."
    assert Translator("en")("filter.count", count=3) == "Found 3 photo(s)."


def test_unknown_key_falls_back_to_key():
    assert Translator("de")("does.not.exist") == "does.not.exist"


def test_html_keys_return_markup_and_escape_params():
    result = Translator("en")("filter.needs_detection_index_html")
    assert isinstance(result, Markup)
    assert "<code>" in result


def test_js_messages_only_contain_js_keys():
    messages = Translator("de").js_messages()
    assert messages
    assert all(key.startswith("js.") for key in messages)


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError):
        Translator("fr")


def test_settings_reject_unsupported_language():
    with pytest.raises(ValidationError):
        Settings(images_path="/tmp", language="fr")
