# -*- coding: utf-8 -*-
"""Unified EN → RU translation entry point and provider catalogue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from translation.get_translate_apis import get_translate_apis

_TRANSLATOR_CLASSES = {
    "google": ("translation.translator.google_tanslator", "GoogleTranslator"),
    "microsoft": ("translation.translator.microsoft_translator", "MicrosoftTranslator"),
    "amazon": ("translation.translator.amazon_translator", "AmazonTranslator"),
    "deepl": ("translation.translator.deepl_translator", "DeeplTranslator"),
    "yandex": ("translation.translator.yandex_translator", "YandexTranslator"),
    "baidu": ("translation.translator.baidu_translator", "BaiduTranslator"),
    "alibaba": ("translation.translator.alibaba_translator", "AlibabaTranslator"),
    "youdao": ("translation.translator.youdao_translator", "YoudaoTranslator"),
    "tencent": ("translation.translator.tencent_translator", "TencentTranslator"),
    "niutrans": ("translation.translator.niutrans_translator", "NiutransTranslator"),
    "volcengine": ("translation.translator.volcengine_translator", "VolcengineTranslator"),
    "iflytekV1": ("translation.translator.iflytekV1_translator", "IflytekV1Translator"),
    "iflytekV2": ("translation.translator.iflytekV2_translator", "IflytekV2Translator"),
    "mbart50": ("translation.translator.mbart50_translator", "MBart50Translator"),
    "myMemory_free": ("translation.translator.mymemory_translator", "MyMemoryTranslator"),
    "openai": ("translation.translator.openai_compatible_translator", "OpenAICompatibleTranslator"),
}


def get_translation_providers() -> list[dict[str, Any]]:
    """Return the donor provider definitions supported for the fixed EN → RU workflow."""
    result = []
    for group in get_translate_apis().get("apis", []):
        for item in group.get("children", []):
            result.append({**item, "group": group.get("type", "")})
    return result


def get_translation_provider(provider_key: str) -> dict[str, Any] | None:
    return next((p for p in get_translation_providers() if p.get("key") == provider_key), None)


def _translate_google_free(text: str, config: dict[str, Any]) -> str:
    import requests

    host = str(config.get("host") or "https://translate.googleapis.com").strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        raise ValueError("Адрес Google Translate должен начинаться с http:// или https://.")
    response = requests.get(
        host + "/translate_a/single",
        params={"client": "gtx", "sl": "en", "tl": "ru", "dt": "t", "q": text},
        headers={"User-Agent": "Mozilla/5.0 Dataset-Curator/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        translated = "".join(part[0] for part in payload[0] if part and part[0])
    except (IndexError, TypeError, KeyError):
        translated = ""
    if not translated.strip():
        raise RuntimeError("Google Translate не вернул текст перевода.")
    return translated.strip()


def translate_prompt(
    text: str,
    provider: str = "google_free",
    provider_config: dict[str, Any] | None = None,
) -> str:
    """Translate one prompt from English to Russian using the selected provider."""
    source_text = (text or "").strip()
    if not source_text:
        return ""
    config = dict(provider_config or {})

    # Backward compatibility with the old translate_en_ru(text, email) call.
    if provider not in {p["key"] for p in get_translation_providers()}:
        config.setdefault("email", str(provider or ""))
        provider = "myMemory_free"

    if provider == "google_free":
        return _translate_google_free(source_text, config)

    provider_item = get_translation_provider(provider)
    if provider_item is None:
        raise ValueError(f"Неизвестный провайдер перевода: {provider}")

    if provider == "mbart50":
        from translation.mbart50 import initialize
        initialize()

    import importlib
    class_info = _TRANSLATOR_CLASSES.get(provider)
    if class_info is None:
        raise RuntimeError(f"Адаптер для провайдера {provider} не подключён.")
    module = importlib.import_module(class_info[0])
    translator_class = getattr(module, class_info[1])
    translator = translator_class()
    translator.set_from_lang("en_US")
    translator.set_to_lang("ru_RU")
    translator.set_api_config(config)

    result = translator.translate(source_text)
    if isinstance(result, list):
        result = " ".join(str(item) for item in result)
    translated = str(result or "").strip()
    if not translated:
        raise RuntimeError(f"{provider_item.get('name', provider)} вернул пустой перевод.")
    return translated
