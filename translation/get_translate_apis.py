# -*- coding: utf-8 -*-
"""Загрузка каталога провайдеров перевода."""

import json
from pathlib import Path

_CACHE = None
_CONFIG_PATH = Path(__file__).with_name("translation_providers.json")

def get_translate_apis(reload=False):
    global _CACHE
    if reload or _CACHE is None:
        with _CONFIG_PATH.open("r", encoding="utf-8") as stream:
            _CACHE = json.load(stream)
    return _CACHE
