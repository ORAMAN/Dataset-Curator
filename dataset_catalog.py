# -*- coding: utf-8 -*-
"""SQLite catalog and final exporter for Dataset Curator.

The catalog stores metadata and descriptions without moving or rewriting source images.
The only file-writing operation is the explicit final export requested by the user.
"""
from __future__ import annotations

import html
import json
import shutil
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

REJECTED_FOLDER_STATUS = {
    "rejected": "rejected",
    "bed_size": "small_resolution",
    "blure": "blurry",
    "no_or_small_faces": "no_face",
    "moree_faces": "multiple_faces",
    "duplicate": "duplicate",
}
SELECTED_STATUSES = {"selected", "maybe"}
REJECTED_STATUSES = {"rejected", "small_resolution", "blurry", "no_face", "multiple_faces", "duplicate", "error"}


class CatalogStore:
    """Small transaction-safe SQLite repository, one database per source folder."""

    def __init__(self, source_dir: str | Path):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.db_path = self.source_dir / "dataset_curator.sqlite3"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    path TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    reason TEXT NOT NULL DEFAULT '',
                    width INTEGER,
                    height INTEGER,
                    aesthetic_score REAL,
                    suitable_faces INTEGER,
                    max_face_height_percent REAL,
                    blur_variance REAL,
                    clip_category TEXT NOT NULL DEFAULT '',
                    clip_relative_score REAL,
                    description_en TEXT NOT NULL DEFAULT '',
                    description_ru TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_images_status ON images(status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename)")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            # A tiny migration path for databases created by earlier iterations.
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(images)")}
            required = {
                "description_en": "TEXT NOT NULL DEFAULT ''",
                "description_ru": "TEXT NOT NULL DEFAULT ''",
                "clip_category": "TEXT NOT NULL DEFAULT ''",
                "clip_relative_score": "REAL",
                "reason": "TEXT NOT NULL DEFAULT ''",
                "error": "TEXT NOT NULL DEFAULT ''",
                "updated_at": "TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'",
            }
            for name, declaration in required.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE images ADD COLUMN {name} {declaration}")

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    def sync_existing_files(self) -> int:
        """Register root images and old result-folder images without moving anything.

        Existing rows are never overwritten. Legacy .txt sidecars are read only when
        a new database row is inserted, as a one-time migration convenience.
        """
        candidates: list[tuple[Path, str]] = []
        if self.source_dir.is_dir():
            candidates.extend(
                (p, "pending")
                for p in self.source_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for folder_name, status in REJECTED_FOLDER_STATUS.items():
                folder = self.source_dir / folder_name
                if folder.is_dir() and not folder.is_symlink():
                    candidates.extend(
                        (p, status)
                        for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    )
            folder = self.source_dir / "selected"
            if folder.is_dir() and not folder.is_symlink():
                for p in folder.iterdir():
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        status = "maybe" if p.name.startswith("!_") else "selected"
                        candidates.append((p, status))

        added = 0
        with self._connect() as connection:
            for path, status in candidates:
                normalized = self._normalize_path(path)
                sidecar = path.with_suffix(".txt")
                description_en = ""
                try:
                    if sidecar.is_file():
                        description_en = sidecar.read_text(encoding="utf-8-sig").strip()
                except (OSError, UnicodeError):
                    pass
                cursor = connection.execute("""
                    INSERT OR IGNORE INTO images(path, filename, status, description_en)
                    VALUES (?, ?, ?, ?)
                """, (normalized, path.name, status, description_en))
                added += max(cursor.rowcount, 0)
        return added

    def upsert_analysis(self, record: dict[str, Any], source_path: str | Path) -> None:
        """Persist one analysis result, preserving existing descriptions when no new caption exists."""
        path = self._normalize_path(source_path)
        values = (
            path,
            Path(path).name,
            str(record.get("status") or "pending"),
            str(record.get("reason") or ""),
            self._optional_int(record.get("width")),
            self._optional_int(record.get("height")),
            self._optional_float(record.get("aesthetic_score")),
            self._optional_int(record.get("suitable_faces")),
            self._optional_float(record.get("max_face_height_percent")),
            self._optional_float(record.get("blur_variance")),
            str(record.get("clip_category") or ""),
            self._optional_float(record.get("clip_relative_score")),
            str(record.get("caption") or ""),
            str(record.get("description_ru") or ""),
            str(record.get("error") or ""),
        )
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO images (
                    path, filename, status, reason, width, height, aesthetic_score,
                    suitable_faces, max_face_height_percent, blur_variance,
                    clip_category, clip_relative_score, description_en, description_ru, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename=excluded.filename,
                    status=excluded.status,
                    reason=excluded.reason,
                    width=COALESCE(excluded.width, images.width),
                    height=COALESCE(excluded.height, images.height),
                    aesthetic_score=COALESCE(excluded.aesthetic_score, images.aesthetic_score),
                    suitable_faces=COALESCE(excluded.suitable_faces, images.suitable_faces),
                    max_face_height_percent=COALESCE(excluded.max_face_height_percent, images.max_face_height_percent),
                    blur_variance=COALESCE(excluded.blur_variance, images.blur_variance),
                    clip_category=CASE WHEN excluded.clip_category != '' THEN excluded.clip_category ELSE images.clip_category END,
                    clip_relative_score=COALESCE(excluded.clip_relative_score, images.clip_relative_score),
                    description_en=CASE WHEN excluded.description_en != '' THEN excluded.description_en ELSE images.description_en END,
                    description_ru=CASE
                        WHEN excluded.description_en != '' THEN excluded.description_ru
                        ELSE images.description_ru
                    END,
                    error=excluded.error,
                    updated_at=CURRENT_TIMESTAMP
            """, values)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def get_image(self, path: str | Path) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM images WHERE path=?", (self._normalize_path(path),)).fetchone()
        return dict(row) if row else None

    def list_images(self, group: str) -> list[dict[str, Any]]:
        if group == "selected":
            values = sorted(SELECTED_STATUSES)
        else:
            values = sorted(REJECTED_STATUSES)
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM images WHERE status IN ({placeholders}) ORDER BY filename COLLATE NOCASE, path COLLATE NOCASE",
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def set_status(self, path: str | Path, status: str, reason: str = "") -> None:
        normalized = self._normalize_path(path)
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO images(path, filename, status, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    status=excluded.status, reason=excluded.reason, updated_at=CURRENT_TIMESTAMP
            """, (normalized, Path(normalized).name, status, reason))

    def set_descriptions(self, path: str | Path, description_en: str, description_ru: str) -> None:
        normalized = self._normalize_path(path)
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO images(path, filename, status, description_en, description_ru)
                VALUES (?, ?, 'pending', ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    description_en=excluded.description_en,
                    description_ru=excluded.description_ru,
                    updated_at=CURRENT_TIMESTAMP
            """, (normalized, Path(normalized).name, description_en.strip(), description_ru.strip()))

    def set_metadata(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO catalog_meta(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, value))

    def get_metadata(self, key: str, default: str = "") -> str:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM catalog_meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def export_selected(self, output_dir: str | Path, include_russian_csv: bool = True) -> tuple[int, int, Path]:
        """Copy accepted images and export English .txt sidecars only on explicit user request."""
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        rows = self.list_images("selected")
        exported = 0
        skipped = 0
        used_names: set[str] = set()
        russian_rows: list[tuple[str, str]] = []

        for row in rows:
            source = Path(row["path"])
            if not source.is_file():
                skipped += 1
                continue
            description_en = str(row.get("description_en") or "").strip()
            if not description_en:
                skipped += 1
                continue

            stem = source.stem
            suffix = source.suffix
            filename = source.name
            n = 1
            while filename.casefold() in used_names or (output / filename).exists() or (output / Path(filename).with_suffix(".txt")).exists():
                n += 1
                filename = f"{stem}_{n}{suffix}"
            used_names.add(filename.casefold())
            destination = output / filename
            shutil.copy2(source, destination)
            caption_path = destination.with_suffix(".txt")
            temp_caption = caption_path.with_suffix(caption_path.suffix + ".tmp")
            temp_caption.write_text(description_en + "\n", encoding="utf-8")
            temp_caption.replace(caption_path)
            exported += 1

            description_ru = str(row.get("description_ru") or "").strip()
            if description_ru:
                russian_rows.append((filename, description_ru))

        if include_russian_csv and russian_rows:
            csv_path = output / "captions_ru.csv"
            temp_csv = csv_path.with_suffix(".csv.tmp")
            with temp_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                import csv
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(["filename", "description_ru"])
                writer.writerows(russian_rows)
            temp_csv.replace(csv_path)
        return exported, skipped, output


def translate_en_ru_mymemory(text: str, email: str = "", timeout: int = 20) -> str:
    """Translate a short English caption to Russian with MyMemory's public REST endpoint.

    MyMemory currently documents a 5,000-character/day anonymous limit and up to
    50,000 characters/day when a valid contact email is supplied. Limits can change.
    """
    text = (text or "").strip()
    if not text:
        return ""
    params: dict[str, str] = {"q": text, "langpair": "en|ru"}
    if email.strip():
        params["de"] = email.strip()
    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "Dataset-Curator/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    status = int(payload.get("responseStatus", 0))
    if status != 200:
        detail = payload.get("responseDetails") or payload.get("responseData", {}).get("translatedText") or "Неизвестная ошибка MyMemory"
        raise RuntimeError(f"MyMemory API: HTTP-подобный статус {status}: {detail}")
    translated = payload.get("responseData", {}).get("translatedText", "")
    translated = html.unescape(str(translated)).strip()
    if not translated:
        raise RuntimeError("MyMemory вернул пустой перевод.")
    return translated
