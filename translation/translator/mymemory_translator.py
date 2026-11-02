# -*- coding: utf-8 -*-
"""MyMemory translator adapter with optional email and API key."""

from translation.translator.base_tanslator import BaseTranslator
import html
import requests
from translation.get_lang import get_lang

class MyMemoryTranslator(BaseTranslator):
    def __init__(self):
        super().__init__("myMemory_free")

    def translate(self, text):
        if not text:
            return ""
        params = {
            "q": text,
            "langpair": f"{self.from_lang}|{self.to_lang}",
        }
        email = str(self.api_config.get("email", "") or "").strip()
        api_key = str(self.api_config.get("api_key", "") or "").strip()
        if email:
            params["de"] = email
        if api_key:
            params["key"] = api_key
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params=params,
            timeout=30,
            headers={"User-Agent": "Dataset-Curator/1.0"},
        )
        response.raise_for_status()
        result = response.json()
        if int(result.get("responseStatus", 0)) != 200:
            raise RuntimeError(str(result.get("responseDetails") or "MyMemory returned an error"))
        translated = html.unescape(str(result.get("responseData", {}).get("translatedText", ""))).strip()
        if not translated:
            raise RuntimeError(get_lang("no_response_from", {"0":"MyMemory"}))
        return translated
