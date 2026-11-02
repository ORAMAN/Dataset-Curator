# -*- coding: utf-8 -*-
"""Offline tests for the configurable translation layer."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dataset_catalog
import translation.translation_service as translation_service
from dataset_catalog import CatalogStore
from translation.translation_service import (
    _TRANSLATOR_CLASSES,
    get_translation_providers,
)


class TranslationProviderTests(unittest.TestCase):
    def test_every_listed_provider_has_an_en_ru_adapter(self):
        providers = get_translation_providers()
        keys = {item["key"] for item in providers}
        expected = {
            "google_free", "myMemory_free", "google", "openai", "microsoft",
            "amazon", "deepl", "yandex", "myMemory", "baidu", "alibaba",
            "youdao", "tencent", "niutrans", "volcengine", "iflytekV1",
            "iflytekV2", "mbart50",
        }
        self.assertEqual(keys, expected)

        for item in providers:
            support = item.get("support", {})
            self.assertTrue(support.get("en_US"), item["key"])
            self.assertTrue(support.get("ru_RU"), item["key"])
        self.assertEqual(keys - {"google_free"}, set(_TRANSLATOR_CLASSES))

    def test_cache_isolated_by_provider_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = CatalogStore(Path(tmp))
            calls = []

            def fake_translate(text, provider, provider_config):
                calls.append((text, provider, dict(provider_config)))
                return f"{provider}:{provider_config.get('variant', '')}"

            with patch.object(translation_service, "translate_prompt", fake_translate):
                first = catalog.translate_en_ru("same caption", "google_free", {"variant": "a"})
                repeated = catalog.translate_en_ru("same caption", "google_free", {"variant": "a"})
                second_provider = catalog.translate_en_ru("same caption", "deepl", {"variant": "a"})
                second_config = catalog.translate_en_ru("same caption", "google_free", {"variant": "b"})

            self.assertEqual(first, "google_free:a")
            self.assertEqual(repeated, first)
            self.assertEqual(second_provider, "deepl:a")
            self.assertEqual(second_config, "google_free:b")
            self.assertEqual(len(calls), 3)

    def test_legacy_mymemory_email_call_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = CatalogStore(Path(tmp))
            calls = []

            def fake_legacy_translate(text, email=""):
                calls.append((text, email))
                return "legacy translation"

            with patch.object(dataset_catalog, "translate_en_ru_mymemory", fake_legacy_translate):
                first = catalog.translate_en_ru("legacy caption", "test@example.com")
                repeated = catalog.translate_en_ru("legacy caption", "test@example.com")

            self.assertEqual(first, "legacy translation")
            self.assertEqual(repeated, first)
            self.assertEqual(calls, [("legacy caption", "test@example.com")])

    def test_old_translation_cache_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "dataset_curator.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.execute("""
                    CREATE TABLE translation_cache (
                        source_text TEXT PRIMARY KEY,
                        translated_text TEXT NOT NULL,
                        provider TEXT NOT NULL DEFAULT 'mymemory',
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                connection.execute(
                    "INSERT INTO translation_cache(source_text, translated_text, provider) VALUES (?, ?, ?)",
                    ("old caption", "old cached translation", "mymemory"),
                )

            catalog = CatalogStore(root)
            with patch.object(
                translation_service,
                "translate_prompt",
                side_effect=AssertionError("The migrated cache entry should be reused"),
            ):
                translated = catalog.translate_en_ru("old caption", "myMemory_free", {})

            self.assertEqual(translated, "old cached translation")


if __name__ == "__main__":
    unittest.main()
