# -*- coding: utf-8 -*-
"""Tests for bulk caption generation and CLIP metadata persistence."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from caption_utils import append_clip_category
from dataset_catalog import CatalogStore


class CaptionGenerationTests(unittest.TestCase):
    def test_clip_category_is_appended_once_when_enabled(self):
        self.assertEqual(
            append_clip_category("woman portrait, soft lighting", "portrait photo", True),
            "woman portrait, soft lighting, portrait photo",
        )
        self.assertEqual(
            append_clip_category("woman portrait, portrait photo", "portrait photo", True),
            "woman portrait, portrait photo",
        )
        self.assertEqual(
            append_clip_category("woman portrait", "portrait photo", False),
            "woman portrait",
        )
        self.assertEqual(append_clip_category("woman portrait", "", True), "woman portrait")

    def test_clip_metadata_is_saved_without_changing_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = CatalogStore(Path(tmp))
            image = Path(tmp) / "selected.jpg"
            catalog.set_status(image, "selected", "")
            catalog.set_descriptions(
                image,
                "woman portrait, portrait photo",
                "Портрет женщины",
                clip_category="portrait photo",
                clip_relative_score=0.87321,
            )

            row = catalog.get_image(image)
            self.assertEqual(row["status"], "selected")
            self.assertEqual(row["description_en"], "woman portrait, portrait photo")
            self.assertEqual(row["clip_category"], "portrait photo")
            self.assertAlmostEqual(row["clip_relative_score"], 0.87321)

    def test_updating_text_without_clip_result_preserves_previous_clip_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = CatalogStore(Path(tmp))
            image = Path(tmp) / "selected.jpg"
            catalog.set_status(image, "selected", "")
            catalog.set_descriptions(
                image, "first caption", "", clip_category="studio photo", clip_relative_score=0.91
            )
            catalog.set_descriptions(image, "second caption", "Перевод")

            row = catalog.get_image(image)
            self.assertEqual(row["status"], "selected")
            self.assertEqual(row["description_en"], "second caption")
            self.assertEqual(row["clip_category"], "studio photo")
            self.assertAlmostEqual(row["clip_relative_score"], 0.91)

    def test_list_all_images_includes_every_review_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = CatalogStore(Path(tmp))
            paths = {
                "selected": Path(tmp) / "selected.jpg",
                "rejected": Path(tmp) / "rejected.jpg",
                "pending": Path(tmp) / "pending.jpg",
            }
            catalog.set_status(paths["selected"], "selected", "")
            catalog.set_status(paths["rejected"], "rejected", "manual")
            catalog.set_descriptions(paths["pending"], "caption", "")

            rows = catalog.list_all_images()
            self.assertEqual({Path(row["path"]).name for row in rows}, {
                "selected.jpg", "rejected.jpg", "pending.jpg"
            })

    def test_bulk_ui_has_both_actions_and_runs_clip(self):
        source = Path("dataset_curator_gui.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="dataset_curator_gui.py")
        app = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DatasetCuratorApp"
        )
        methods = {
            node.name: node
            for node in app.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("_generate_missing_selected_captions", methods)
        self.assertIn("_recreate_all_descriptions", methods)
        self.assertIn("_run_bulk_caption_generation", methods)

        bulk_source = ast.get_source_segment(source, methods["_run_bulk_caption_generation"])
        self.assertIn("catalog.list_all_images()", bulk_source)
        self.assertIn("services.classify(image)", bulk_source)
        self.assertIn("clip_category=clip_category", bulk_source)
        self.assertIn("append_clip_category(", bulk_source)
        self.assertIn("text=\"♻ Пересоздать все описания\"", source)


if __name__ == "__main__":
    unittest.main()
