# -*- coding: utf-8 -*-
"""Dataset Curator — Tkinter GUI для отбора изображений перед обучением LoRA/SDXL.

Базовые проверки: разрешение, лица через MediaPipe, размытие, эстетическая оценка.
Дополнительно: подписи BLIP, zero-shot категории CLIP, поиск near-duplicates.
Тяжёлые модели загружаются только при включении соответствующей функции.
"""
from __future__ import annotations

import csv
import json
import logging
import importlib.util
import sys
import os
import re
import queue
import shutil
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from dataset_catalog import CatalogStore

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "Dataset Curator — отбор изображений для LoRA"
APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / "dataset_curator_settings.json"
TOKEN_FILE = APP_DIR / "hf_token.txt"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_CLIP_LABELS = """portrait photo
close-up portrait
full-body photo
fashion photo
studio photo
outdoor photo
indoor photo
group photo
landscape photo
illustration
anime illustration
3D render"""

# HF_TOKEN не выводится в журнал. Сам файл с токеном не обязателен для публичных моделей.
if TOKEN_FILE.exists():
    _hf_token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if _hf_token:
        os.environ.setdefault("HF_TOKEN", _hf_token)


def load_config() -> dict[str, Any]:
    try:
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_config(data: dict[str, Any]) -> bool:
    """Атомарно сохраняет настройки, чтобы повреждение JSON при закрытии было маловероятно."""
    temp_path = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
    try:
        # Сначала временный файл, затем атомарная замена старой конфигурации.
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, CONFIG_FILE)
        return True
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        # Невозможность сохранить настройки не должна мешать обработке изображений.
        return False


def imread_unicode(path: Path):
    """Чтение через Pillow ниже; эта функция оставлена для OpenCV и Unicode-путей."""
    import cv2
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def get_blur_variance(image_rgb) -> float:
    """Variance of Laplacian: низкое значение часто указывает на смазанное изображение."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())




@dataclass
class Settings:
    input_dir: str
    clean_outputs: bool = False
    check_resolution: bool = True
    min_resolution: int = 1024
    resolution_mode: str = "Одной стороной"  # or "Обеими сторонами"
    check_faces: bool = True
    face_model_path: str = str(APP_DIR / "blaze_face_short_range.tflite")
    min_face_percent: float = 8.0
    reject_multiple_faces: bool = True
    check_blur: bool = False
    blur_threshold: float = 110.0
    check_aesthetic: bool = True
    aesthetic_keep: float = 5.0
    aesthetic_maybe: float = 4.0
    find_duplicates: bool = False
    duplicate_hash: str = "phash"
    duplicate_distance: int = 8
    duplicate_dry_run: bool = True
    generate_captions: bool = False
    caption_only: bool = False
    skip_existing_captions: bool = True
    translate_captions: bool = True
    translation_email: str = ""
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_prefix: str = ""
    classify_clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_labels: str = DEFAULT_CLIP_LABELS
    append_clip_label_to_caption: bool = True


class ModelServices:
    """Ленивая инициализация AI-моделей, каждая модель загружается не чаще одного раза."""
    def __init__(self, settings: Settings, log: Callable[[str], None]):
        self.settings = settings
        self.log = log
        self.predict_aesthetic = None
        self.face_detector = None
        self.caption_processor = None
        self.caption_model_obj = None
        self.clip_processor = None
        self.clip_model_obj = None
        self.clip_text_features = None
        self.clip_labels: list[str] = []
        self.torch = None
        self.device = "cpu"

    def _torch(self):
        if self.torch is None:
            import torch
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log(f"PyTorch: устройство {self.device.upper()}" + (f", CUDA {torch.version.cuda}" if self.device == "cuda" else ""))
        return self.torch

    def load_aesthetic(self):
        if self.predict_aesthetic is None:
            self.log("Загрузка модели эстетической оценки…")
            from aesthetic_predictor import predict_aesthetic
            self.predict_aesthetic = predict_aesthetic
        return self.predict_aesthetic

    def load_face_detector(self):
        if self.face_detector is not None:
            return self.face_detector
        model_path = Path(self.settings.face_model_path).expanduser()
        if not model_path.is_file():
            # Если модель отсутствует, пробуем загрузить официальный файл MediaPipe.
            # Пишем сначала во временный файл и атомарно переименовываем после загрузки.
            import urllib.request
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = model_path.with_suffix(model_path.suffix + ".download")
                self.log("Файл модели MediaPipe не найден. Загружаю официальный BlazeFace Short Range…")
                request = urllib.request.Request(FACE_MODEL_URL, headers={"User-Agent": "DatasetCurator/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
                if temp_path.stat().st_size < 100_000:
                    raise RuntimeError("скачанный файл подозрительно мал, вероятно это не модель")
                temp_path.replace(model_path)
                self.log(f"Модель скачана: {model_path}")
            except Exception as exc:
                try:
                    if 'temp_path' in locals() and temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                raise FileNotFoundError(
                    "Не удалось получить модель MediaPipe.\n"
                    f"Ожидаемый путь: {model_path}\n"
                    f"Ошибка загрузки: {exc}\n"
                    f"Официальная ссылка: {FACE_MODEL_URL}\n"
                    "Скачайте файл вручную или отключите проверку лиц."
                ) from exc
        self.log("Инициализация MediaPipe Face Detector…")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.55,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        return self.face_detector

    def detect_suitable_faces(self, image_rgb) -> tuple[int, float]:
        """Один проход детектора на изображение. Возвращает число лиц и максимальную долю высоты."""
        import mediapipe as mp
        import numpy as np
        detector = self.load_face_detector()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image_rgb, dtype=np.uint8))
        result = detector.detect(mp_image)
        image_height = mp_image.height
        min_fraction = self.settings.min_face_percent / 100.0
        rel_sizes = [d.bounding_box.height / image_height for d in (result.detections or [])]
        suitable = [value for value in rel_sizes if value >= min_fraction]
        return len(suitable), max(rel_sizes, default=0.0)

    def load_caption_model(self):
        if self.caption_model_obj is not None:
            return
        torch = self._torch()
        self.log(f"Загрузка модели описаний: {self.settings.caption_model} …")
        from transformers import AutoProcessor
        self.caption_processor = AutoProcessor.from_pretrained(self.settings.caption_model)
        # Transformers 4.x и 5.x используют разные публичные классы для этой модели.
        try:
            from transformers import BlipForConditionalGeneration
            model_class = BlipForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForMultimodalLM
            model_class = AutoModelForMultimodalLM
        self.caption_model_obj = model_class.from_pretrained(self.settings.caption_model)
        self.caption_model_obj.to(self.device)
        self.caption_model_obj.eval()
        self.log(f"Модель описаний загружена ({self.device.upper()}).")

    def caption(self, image_rgb) -> str:
        self.load_caption_model()
        torch = self._torch()
        inputs = self.caption_processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.caption_model_obj.generate(**inputs, max_new_tokens=40, num_beams=3)
        return self.caption_processor.batch_decode(output, skip_special_tokens=True)[0].strip()

    def load_clip_model(self):
        if self.clip_model_obj is not None:
            return
        torch = self._torch()
        labels = [line.strip() for line in self.settings.clip_labels.splitlines() if line.strip()]
        if len(labels) < 2:
            raise ValueError("Для CLIP укажите минимум две категории — по одной на строку.")
        self.clip_labels = labels
        self.log(f"Загрузка CLIP: {self.settings.clip_model} ({len(labels)} категорий)…")
        from transformers import CLIPModel, CLIPProcessor
        self.clip_processor = CLIPProcessor.from_pretrained(self.settings.clip_model)
        self.clip_model_obj = CLIPModel.from_pretrained(self.settings.clip_model)
        self.clip_model_obj.to(self.device)
        self.clip_model_obj.eval()
        text_inputs = self.clip_processor(text=labels, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            features = self.clip_model_obj.get_text_features(**text_inputs)
            self.clip_text_features = features / features.norm(dim=-1, keepdim=True)
        self.log("CLIP загружен; текстовые векторы категорий закэшированы.")

    def classify(self, image_rgb) -> tuple[str, float]:
        self.load_clip_model()
        torch = self._torch()
        inputs = self.clip_processor(images=image_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            image_features = self.clip_model_obj.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # Softmax по пользовательским категориям — относительная уверенность среди них,
            # а не абсолютная вероятность правильности.
            logits = 100.0 * image_features @ self.clip_text_features.T
            probs = logits.softmax(dim=-1)[0]
            best_idx = int(probs.argmax().item())
            probability = float(probs[best_idx].item())
        return self.clip_labels[best_idx], probability

    def close(self):
        try:
            if self.face_detector is not None:
                self.face_detector.close()
        except Exception:
            pass
        self.face_detector = None
        self.caption_model_obj = None
        self.clip_model_obj = None
        self.clip_text_features = None
        if self.torch is not None and self.device == "cuda":
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass


class BKNode:
    __slots__ = ("value", "indices", "children")

    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    """BK-tree для поиска хешей в пределах расстояния Хэмминга без полного O(n²) перебора."""
    def __init__(self):
        self.root: BKNode | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, index)
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, index)
                return
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= radius:
                matches.extend(node.indices)
            low, high = distance - radius, distance + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return matches


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def deduplicate_selected(
    settings: Settings,
    log: Callable[[str], None],
    stop_event: threading.Event,
    catalog: CatalogStore | None = None,
) -> tuple[int, dict[str, str]]:
    """Ищет near-duplicates среди выбранных записей и при подтверждённом режиме меняет только статус в SQLite."""
    import imagehash
    from PIL import Image, ImageOps

    catalog = catalog or CatalogStore(settings.input_dir)
    rows = catalog.list_images("selected")
    paths = [Path(row["path"]) for row in rows if Path(row["path"]).is_file()]
    if len(paths) < 2:
        log("Для поиска дубликатов в выбранном каталоге недостаточно изображений.")
        return 0, {}

    hash_func = getattr(imagehash, settings.duplicate_hash)
    quality: list[tuple[int, int]] = []
    valid_paths: list[Path] = []
    tree = BKTree()
    uf = UnionFind(len(paths))
    log(
        f"Ищу near-duplicates среди выбранных записей SQLite: {len(paths)} файлов, "
        f"{settings.duplicate_hash}, порог {settings.duplicate_distance}…"
    )

    for path in paths:
        if stop_event.is_set():
            log("Поиск дубликатов остановлен пользователем.")
            return 0, {}
        try:
            with Image.open(path) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = img.size
                hash_value = int(str(hash_func(img)), 16)
            idx = len(valid_paths)
            for other in tree.query(hash_value, settings.duplicate_distance):
                uf.union(idx, other)
            tree.add(hash_value, idx)
            valid_paths.append(path)
            quality.append((width * height, path.stat().st_size))
        except Exception as exc:
            log(f"Не удалось вычислить хеш {path.name}: {exc}")

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(valid_paths)):
        groups[uf.find(idx)].append(idx)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    if not duplicate_groups:
        log("Дубликаты / near-duplicates не найдены.")
        return 0, {}

    log(f"Найдено групп похожих изображений: {len(duplicate_groups)}")
    duplicate_count = 0
    duplicate_map: dict[str, str] = {}
    for group in duplicate_groups:
        ordered = sorted(group, key=lambda i: (quality[i][0], quality[i][1]), reverse=True)
        keeper = valid_paths[ordered[0]]
        log(f"Оставляю: {keeper.name} ({quality[ordered[0]][0]:,} пикс.)")
        for idx in ordered[1:]:
            duplicate = valid_paths[idx]
            duplicate_count += 1
            log(f"  Похожий: {duplicate.name}" + (" [тестовый режим]" if settings.duplicate_dry_run else ""))
            if not settings.duplicate_dry_run:
                try:
                    catalog.set_status(
                        duplicate, "duplicate",
                        reason=f"Near-duplicate: выбран более качественный оригинал {keeper.name}",
                    )
                    duplicate_map[str(duplicate)] = str(duplicate)
                    log(f"  → статус duplicate сохранён в SQLite; файл оставлен на месте")
                except Exception as exc:
                    log(f"  Ошибка обновления записи SQLite для {duplicate.name}: {exc}")

    if settings.duplicate_dry_run:
        log(f"Тестовый режим: найдено кандидатов — {duplicate_count}. Статусы и файлы не изменены.")
    else:
        log(f"В SQLite помечено как duplicate: {len(duplicate_map)}. Файлы не перемещались.")
    return duplicate_count, duplicate_map


class DatasetProcessor:
    def __init__(self, settings: Settings, stop_event: threading.Event, send: Callable[[tuple], None]):
        self.settings = settings
        self.stop_event = stop_event
        self.send = send
        self.stats = {
            "selected": 0, "maybe": 0, "rejected": 0, "small_resolution": 0,
            "no_face": 0, "multiple_faces": 0, "blurry": 0, "errors": 0,
            "duplicates": 0, "captions": 0, "captions_skipped": 0, "processed": 0,
        }
        self.report_rows: list[dict[str, Any]] = []
        self.services = ModelServices(settings, self.log)
        self.catalog = CatalogStore(settings.input_dir)

    def log(self, message: str) -> None:
        self.send(("log", message))

    def update_progress(self, current: int, total: int, filename: str) -> None:
        self.send(("progress", current, total, filename, dict(self.stats)))

    def _route(self, folder: str, source: Path, name: str | None = None) -> Path:
        """В режиме каталога назначает путь для отчёта, но никогда не копирует файл."""
        return source

    def _process_one(self, source: Path, index: int, total: int) -> None:
        from PIL import Image, ImageOps

        record: dict[str, Any] = {
            "filename": source.name, "output_file": str(source), "status": "error", "reason": "",
            "width": "", "height": "", "aesthetic_score": "", "suitable_faces": "",
            "max_face_height_percent": "", "blur_variance": "", "clip_category": "",
            "clip_relative_score": "", "caption": "", "description_ru": "", "error": "",
        }
        self.log(f"[{index}/{total}] {source.name}")
        try:
            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = image.size
            record["width"], record["height"] = width, height

            if self.settings.check_resolution:
                if self.settings.resolution_mode == "Обеими сторонами":
                    resolution_ok = min(width, height) >= self.settings.min_resolution
                else:
                    resolution_ok = max(width, height) >= self.settings.min_resolution
                if not resolution_ok:
                    dest = self._route("bed_size", source)
                    record.update(status="small_resolution", reason=f"Недостаточное разрешение {width}x{height}", output_file=str(dest))
                    self.stats["small_resolution"] += 1
                    self.log(f"  → разрешение {width}×{height} ниже порога → статус small_resolution в SQLite")
                    return

            if self.settings.check_faces:
                count, max_rel = self.services.detect_suitable_faces(image)
                record["suitable_faces"] = count
                record["max_face_height_percent"] = round(max_rel * 100, 2)
                if count < 1:
                    dest = self._route("no_or_small_faces", source)
                    record.update(status="no_face", reason="Нет достаточно крупного лица", output_file=str(dest))
                    self.stats["no_face"] += 1
                    self.log(f"  → нет подходящего лица (макс. высота {max_rel * 100:.1f}%) → статус no_face в SQLite")
                    return
                if self.settings.reject_multiple_faces and count > 1:
                    dest = self._route("moree_faces", source)
                    record.update(status="multiple_faces", reason=f"Подходящих лиц: {count}", output_file=str(dest))
                    self.stats["multiple_faces"] += 1
                    self.log(f"  → подходящих лиц: {count} → статус multiple_faces в SQLite")
                    return

            if self.settings.check_blur:
                variance = get_blur_variance(image)
                record["blur_variance"] = round(variance, 3)
                if variance < self.settings.blur_threshold:
                    dest = self._route("blure", source)
                    record.update(status="blurry", reason=f"Размыто: Laplacian variance {variance:.2f}", output_file=str(dest))
                    self.stats["blurry"] += 1
                    self.log(f"  → размыто ({variance:.1f} < {self.settings.blur_threshold:g}) → статус blurry в SQLite")
                    return

            score = None
            if self.settings.check_aesthetic:
                score = float(self.services.load_aesthetic()(image).item())
                record["aesthetic_score"] = round(score, 4)

            if score is not None and score < self.settings.aesthetic_maybe:
                dest = self._route("rejected", source)
                record.update(status="rejected", reason=f"Низкая эстетическая оценка {score:.2f}", output_file=str(dest))
                self.stats["rejected"] += 1
                self.log(f"  → эстетика {score:.2f} → статус rejected в SQLite")
                return

            if score is not None and score < self.settings.aesthetic_keep:
                dest = self._route("selected", source)
                status = "maybe"
                self.stats["maybe"] += 1
                self.log(f"  → эстетика {score:.2f} → пограничный статус maybe в SQLite")
            else:
                dest = self._route("selected", source)
                status = "selected"
                self.stats["selected"] += 1
                if score is None:
                    self.log("  → фильтры пройдены → selected")
                else:
                    self.log(f"  → эстетика {score:.2f} → selected")
            record.update(status=status, reason="", output_file=str(dest))

            caption = ""
            clip_category = ""
            clip_score = None
            # AI-метки выполняются только для изображений, попавших в selected.
            # Это существенно экономит время при больших наборах данных.
            if self.settings.classify_clip:
                try:
                    clip_category, clip_score = self.services.classify(image)
                    record["clip_category"] = clip_category
                    record["clip_relative_score"] = round(clip_score, 6)
                    self.log(f"  → CLIP: {clip_category} ({clip_score * 100:.1f}% среди заданных категорий)")
                except Exception as exc:
                    self.log(f"  ! ошибка CLIP: {exc}")
                    record["error"] = f"CLIP: {exc}"

            if self.settings.generate_captions:
                previous_description = self.catalog.get_image(source) or {}
                if self.settings.skip_existing_captions and str(previous_description.get("description_en") or "").strip():
                    self.stats["captions_skipped"] += 1
                    self.log("  → подпись пропущена: английское описание уже есть в SQLite")
                else:
                    try:
                        caption = self.services.caption(image)
                        prefix = self.settings.caption_prefix.strip().strip(",;")
                        if prefix and not caption.casefold().startswith(prefix.casefold()):
                            caption = f"{prefix}, {caption}".strip().strip(",")
                        if self.settings.append_clip_label_to_caption and clip_category:
                            if clip_category.casefold() not in caption.casefold():
                                caption = f"{caption.rstrip(' .,;')} , {clip_category}".replace(" ,", ",")
                        record["caption"] = caption
                        record["description_ru"] = ""
                        if self.settings.translate_captions:
                            try:
                                russian = self.catalog.translate_en_ru(caption, self.settings.translation_email)
                                record["description_ru"] = russian
                                self.log(f"  → перевод RU: {russian}")
                            except Exception as translate_exc:
                                self.log(f"  ! перевод MyMemory не выполнен: {type(translate_exc).__name__}: {translate_exc}")
                        self.stats["captions"] += 1
                        self.log(f"  → подпись EN сохранена в SQLite: {caption}")
                    except Exception as exc:
                        self.log(f"  ! ошибка генерации подписи: {exc}")
                        previous = record.get("error", "")
                        record["error"] = (previous + " | " if previous else "") + f"BLIP: {exc}"
        except Exception as exc:
            self.stats["errors"] += 1
            record.update(status="error", reason="Ошибка обработки", error=f"{type(exc).__name__}: {exc}")
            self.log(f"  ✖ ошибка: {type(exc).__name__}: {exc}")
        finally:
            try:
                self.catalog.upsert_analysis(record, source)
            except Exception as db_exc:
                self.log(f"ОШИБКА SQLite для {source.name}: {type(db_exc).__name__}: {db_exc}")
                self.stats["errors"] += 1
            self.stats["processed"] += 1
            self.report_rows.append(record)
            self.update_progress(index, total, source.name)

    def _process_caption_only(self, source: Path, index: int, total: int) -> None:
        """Генерирует описания в SQLite без создания sidecar-файлов и без фильтров."""
        from PIL import Image, ImageOps

        previous = self.catalog.get_image(source) or {}
        record: dict[str, Any] = {
            "filename": source.name, "output_file": str(source),
            "status": previous.get("status", "pending"), "reason": previous.get("reason", ""),
            "width": previous.get("width", ""), "height": previous.get("height", ""),
            "aesthetic_score": previous.get("aesthetic_score", ""),
            "suitable_faces": previous.get("suitable_faces", ""),
            "max_face_height_percent": previous.get("max_face_height_percent", ""),
            "blur_variance": previous.get("blur_variance", ""),
            "clip_category": previous.get("clip_category", ""),
            "clip_relative_score": previous.get("clip_relative_score", ""),
            "caption": "", "description_ru": "", "error": "",
        }
        self.log(f"[Подписи {index}/{total}] {source.name}")
        try:
            if self.settings.skip_existing_captions and str(previous.get("description_en", "")).strip():
                self.stats["captions_skipped"] += 1
                self.log("  → пропуск: английское описание уже есть в SQLite")
                return

            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = image.size
            record["width"], record["height"] = width, height

            clip_category = ""
            if self.settings.classify_clip:
                try:
                    clip_category, clip_score = self.services.classify(image)
                    record["clip_category"] = clip_category
                    record["clip_relative_score"] = round(clip_score, 6)
                    self.log(f"  → CLIP: {clip_category} ({clip_score * 100:.1f}% среди категорий)")
                except Exception as exc:
                    self.log(f"  ! CLIP недоступен для {source.name}: {type(exc).__name__}: {exc}")

            caption = self.services.caption(image)
            prefix = self.settings.caption_prefix.strip().strip(",;")
            if prefix and not caption.casefold().startswith(prefix.casefold()):
                caption = f"{prefix}, {caption}".strip().strip(",")
            if self.settings.append_clip_label_to_caption and clip_category:
                if clip_category.casefold() not in caption.casefold():
                    caption = f"{caption.rstrip(' .,;')} , {clip_category}".replace(" ,", ",")

            record.update(reason="", caption=caption, description_ru="")
            if self.settings.translate_captions:
                try:
                    record["description_ru"] = self.catalog.translate_en_ru(caption, self.settings.translation_email)
                    self.log(f"  → перевод RU: {record['description_ru']}")
                except Exception as translate_exc:
                    self.log(f"  ! перевод MyMemory не выполнен: {type(translate_exc).__name__}: {translate_exc}")
            self.stats["captions"] += 1
            self.log(f"  → описание EN подготовлено для SQLite: {caption}")
        except Exception as exc:
            self.stats["errors"] += 1
            record.update(reason="Ошибка генерации подписи", error=f"{type(exc).__name__}: {exc}")
            self.log(f"  ✖ ошибка подписи: {type(exc).__name__}: {exc}")
        finally:
            try:
                self.catalog.upsert_analysis(record, source)
            except Exception as db_exc:
                self.log(f"ОШИБКА SQLite для {source.name}: {type(db_exc).__name__}: {db_exc}")
                self.stats["errors"] += 1
            self.stats["processed"] += 1
            self.report_rows.append(record)
            self.update_progress(index, total, source.name)

    def save_report(self) -> Path | None:
        """Вместо CSV возвращает путь к базе SQLite; никаких итоговых файлов не создаём."""
        return self.catalog.db_path

    def run(self) -> None:
        started = time.monotonic()
        input_dir = Path(self.settings.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Папка не существует: {input_dir}")

        # Синхронизируем только записи каталога. Исходные фото остаются на местах;
        # существующие старые папки selected/rejected импортируются в БД без изменений.
        imported = self.catalog.sync_existing_files()
        paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            # В базе могут остаться ранее импортированные фото только из папок результата.
            if not self.catalog.list_images("selected") and not self.catalog.list_images("rejected"):
                raise FileNotFoundError("В выбранной папке и каталоге SQLite не найдено изображений.")
        self.log(f"Найдено изображений в корне папки: {len(paths)}")
        self.log(f"SQLite: {self.catalog.db_path}")
        if imported:
            self.log(f"Добавлено в каталог записей о существующих изображениях: {imported}")
        self.log("Исходные изображения не перемещаются и не копируются. Статусы и описания хранятся в SQLite.")
        if self.settings.caption_only:
            self.log("РЕЖИМ «ТОЛЬКО ПОДПИСИ»: проверки разрешения, лиц, резкости, эстетики и дубликатов пропускаются.")
            self.log("Подписи EN/RU записываются в базу данных, а не в файлы .txt.")
            if self.settings.skip_existing_captions:
                self.log("Изображения с непустым description_en в SQLite будут пропущены.")

        failed = False
        try:
            if not self.settings.caption_only and self.settings.check_faces:
                self.services.load_face_detector()
            if not self.settings.caption_only and self.settings.check_aesthetic:
                self.services.load_aesthetic()
            # Загружаем включённые модели заранее, чтобы ошибка модели не возникала в середине набора.
            if self.settings.classify_clip:
                try:
                    self.services.load_clip_model()
                except Exception as exc:
                    self.settings.classify_clip = False
                    self.log(f"Предупреждение: CLIP не удалось загрузить ({type(exc).__name__}: {exc}). Продолжаю без классификации.")
            if self.settings.generate_captions:
                try:
                    self.services.load_caption_model()
                except Exception as exc:
                    if self.settings.caption_only:
                        raise RuntimeError(
                            f"Режим «Только подписи» не может продолжиться без модели BLIP: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    self.settings.generate_captions = False
                    self.log(f"Предупреждение: BLIP не удалось загрузить ({type(exc).__name__}: {exc}). Продолжаю без генерации подписей.")

            # Настройка clean_outputs намеренно не удаляет каталоги при работе с SQLite.
            if self.settings.clean_outputs:
                self.log("Очистка файлов отключена в режиме SQLite: никаких каталогов или изображений не удаляется.")

            for index, path in enumerate(paths, 1):
                if self.stop_event.is_set():
                    self.log("Остановка запрошена пользователем — новые изображения не запускаются.")
                    break
                if self.settings.caption_only:
                    self._process_caption_only(path, index, len(paths))
                else:
                    self._process_one(path, index, len(paths))

            if self.settings.find_duplicates and not self.settings.caption_only and not self.stop_event.is_set():
                try:
                    duplicate_count, duplicate_map = deduplicate_selected(
                        self.settings, self.log, self.stop_event, catalog=self.catalog
                    )
                    self.stats["duplicates"] = duplicate_count
                    if duplicate_map:
                        for row in self.report_rows:
                            path_text = str(row.get("output_file", ""))
                            if path_text in duplicate_map:
                                row["status"] = "duplicate"
                                row["reason"] = "Near-duplicate; помечен в SQLite, файл оставлен на месте"
                except Exception as exc:
                    self.log(f"Ошибка поиска дубликатов: {type(exc).__name__}: {exc}")
            elif self.settings.find_duplicates and not self.settings.caption_only and self.stop_event.is_set():
                self.log("Поиск дубликатов пропущен: обработка была остановлена.")
        except Exception:
            failed = True
            raise
        finally:
            report = self.save_report()
            self.services.close()
            elapsed = time.monotonic() - started
            self.log(f"База каталога: {report}" if report else "Не удалось определить путь к базе SQLite.")
            self.log(f"Время обработки: {elapsed:.1f} сек.")
            if not failed:
                self.send(("done", dict(self.stats), str(report) if report else "", self.stop_event.is_set()))


class DatasetCuratorApp(tk.Tk):
    # Палитры и логика переключения повторяют подход VK Downloader:
    # графитовая тёмная тема и контрастная светлая тема с синим акцентом.
    THEME_PALETTES = {
        "light": {
            "app": "#F4F5F7", "panel": "#FFFFFF", "white": "#FFFFFF",
            "text": "#202124", "muted": "#62666B", "border": "#C8CFD8",
            "accent": "#1479C9", "accent_active": "#0F6DB6", "accent_disabled": "#9AC4E5", "accent_text": "#FFFFFF",
            "selection": "#D6E8F8", "button": "#E5E7EB", "button_active": "#D1D5DB",
            "button_pressed": "#C5CBD4", "button_disabled": "#EFF0F2", "disabled_text": "#8A9098",
            "progress_track": "#E9EDF2", "scrollbar": "#D1D5DB",
            "danger_bg": "#FDECEC", "danger_text": "#B42318", "danger_border": "#F5C5C5",
            "danger_active": "#F9DEDE", "danger_pressed": "#F5D0D0", "danger_disabled": "#F7F2F2",
        },
        "dark": {
            "app": "#151515", "panel": "#242424", "white": "#1C1C1C",
            "text": "#EEEEEE", "muted": "#AAAAAA", "border": "#383838",
            "accent": "#167AC6", "accent_active": "#1989DC", "accent_disabled": "#38566D", "accent_text": "#FFFFFF",
            "selection": "#244A68", "button": "#333333", "button_active": "#444444",
            "button_pressed": "#292929", "button_disabled": "#292929", "disabled_text": "#777777",
            "progress_track": "#383838", "scrollbar": "#444444",
            "danger_bg": "#482D32", "danger_text": "#FFB4AB", "danger_border": "#704148",
            "danger_active": "#613B42", "danger_pressed": "#542F37", "danger_disabled": "#332E32",
        },
    }

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(920, 700)
        self.config_data = load_config()
        self.theme_name = self.config_data.get("theme", "dark")
        if self.theme_name not in self.THEME_PALETTES:
            self.theme_name = "dark"
        self.palette = dict(self.THEME_PALETTES[self.theme_name])
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        # Восстанавливаем нормальный размер и координаты окна из последней сессии.
        self._normal_geometry = self._validated_geometry(self.config_data.get("window_geometry", "1120x850+80+60"))
        self.geometry(self._normal_geometry)
        if self.config_data.get("window_state") == "zoomed":
            try:
                self.state("zoomed")
            except tk.TclError:
                pass

        self._config_save_after_id = None
        self._last_geometry = self._normal_geometry
        self.event_queue: queue.Queue[tuple] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.settings_snapshot: Settings | None = None
        self.last_stats: dict[str, int] = {}
        self.review_page_size = 60
        self.review_file_cache: dict[str, list[Path]] = {"selected": [], "rejected": []}
        self.review_record_cache: dict[str, dict[str, dict[str, Any]]] = {"selected": {}, "rejected": {}}
        self.review_visible_count: dict[str, int] = {"selected": 0, "rejected": 0}
        self.review_card_widgets: dict[tuple[str, str], dict[str, Any]] = {}
        self.review_caption_drafts: dict[str, str] = {}
        self.review_caption_ru_drafts: dict[str, str] = {}
        self.review_photo_refs: dict[str, list[Any]] = {"selected": [], "rejected": []}
        self.review_workers: dict[tuple[str, str], threading.Thread] = {}
        self.review_count_vars: dict[str, tk.StringVar] = {}
        self.review_canvases: dict[str, tk.Canvas] = {}
        self.review_list_frames: dict[str, ttk.Frame] = {}
        self.review_load_more_buttons: dict[str, ttk.Button] = {}
        # Асинхронная загрузка каталога и поэтапное создание карточек, чтобы Tk не зависал.
        self.review_loaded: dict[str, bool] = {"selected": False, "rejected": False}
        self.review_dirty: dict[str, bool] = {"selected": True, "rejected": True}
        self.review_base_paths: dict[str, str] = {"selected": "", "rejected": ""}
        self.review_loading: dict[str, bool] = {"selected": False, "rejected": False}
        self.review_load_generation: dict[str, int] = {"selected": 0, "rejected": 0}
        self.review_render_generation: dict[str, int] = {"selected": 0, "rejected": 0}
        self.review_render_target: dict[str, int] = {"selected": 0, "rejected": 0}
        self.review_rendering: dict[str, bool] = {"selected": False, "rejected": False}
        self.review_catalog_request_id = 0
        self.review_load_workers: dict[int, threading.Thread] = {}
        self.export_worker: threading.Thread | None = None
        self._make_vars()
        self._build_ui()
        self._apply_theme(self.theme_name, persist=False)
        self._attach_auto_save()
        self.bind("<Configure>", self._on_root_configure, add="+")
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _validated_geometry(self, value: Any) -> str:
        """Проверяет geometry и не восстанавливает окно полностью за границами экрана."""
        value = str(value or "")
        match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)?([+-]\d+)?", value)
        if not match:
            return "1120x850+80+60"
        width, height = int(match.group(1)), int(match.group(2))
        width = max(920, min(width, max(920, self.winfo_screenwidth())))
        height = max(700, min(height, max(700, self.winfo_screenheight())))
        x = int(match.group(3) or "+80")
        y = int(match.group(4) or "+60")
        # Отрицательные координаты допустимы, если окно было на мониторе слева/сверху
        # от основного. Ограничиваем только явно экстремальные координаты.
        x = max(-2 * self.winfo_screenwidth(), min(x, max(0, self.winfo_screenwidth() - 120)))
        y = max(-2 * self.winfo_screenheight(), min(y, max(0, self.winfo_screenheight() - 120)))
        return f"{width}x{height}+{x}+{y}"

    def _make_vars(self):
        c = self.config_data
        defaults = asdict(Settings(input_dir=""))
        merged = {**defaults, **c}
        self.v = {
            "input_dir": tk.StringVar(value=merged.get("input_dir", "")),
            "clean_outputs": tk.BooleanVar(value=merged.get("clean_outputs", False)),
            "check_resolution": tk.BooleanVar(value=merged.get("check_resolution", True)),
            "min_resolution": tk.StringVar(value=str(merged.get("min_resolution", 1024))),
            "resolution_mode": tk.StringVar(value=merged.get("resolution_mode", "Одной стороной")),
            "check_faces": tk.BooleanVar(value=merged.get("check_faces", True)),
            "face_model_path": tk.StringVar(value=merged.get("face_model_path", str(APP_DIR / "blaze_face_short_range.tflite"))),
            "min_face_percent": tk.StringVar(value=str(merged.get("min_face_percent", 8.0))),
            "reject_multiple_faces": tk.BooleanVar(value=merged.get("reject_multiple_faces", True)),
            "check_blur": tk.BooleanVar(value=merged.get("check_blur", False)),
            "blur_threshold": tk.StringVar(value=str(merged.get("blur_threshold", 110.0))),
            "check_aesthetic": tk.BooleanVar(value=merged.get("check_aesthetic", True)),
            "aesthetic_keep": tk.StringVar(value=str(merged.get("aesthetic_keep", 5.0))),
            "aesthetic_maybe": tk.StringVar(value=str(merged.get("aesthetic_maybe", 4.0))),
            "find_duplicates": tk.BooleanVar(value=merged.get("find_duplicates", False)),
            "duplicate_hash": tk.StringVar(value=merged.get("duplicate_hash", "phash")),
            "duplicate_distance": tk.StringVar(value=str(merged.get("duplicate_distance", 8))),
            "duplicate_dry_run": tk.BooleanVar(value=merged.get("duplicate_dry_run", True)),
            "generate_captions": tk.BooleanVar(value=merged.get("generate_captions", False)),
            "caption_only": tk.BooleanVar(value=merged.get("caption_only", False)),
            "skip_existing_captions": tk.BooleanVar(value=merged.get("skip_existing_captions", True)),
            "translate_captions": tk.BooleanVar(value=merged.get("translate_captions", True)),
            "translation_email": tk.StringVar(value=merged.get("translation_email", "")),
            "caption_model": tk.StringVar(value=merged.get("caption_model", "Salesforce/blip-image-captioning-base")),
            "caption_prefix": tk.StringVar(value=merged.get("caption_prefix", "")),
            "classify_clip": tk.BooleanVar(value=merged.get("classify_clip", False)),
            "clip_model": tk.StringVar(value=merged.get("clip_model", "openai/clip-vit-base-patch32")),
            "append_clip_label_to_caption": tk.BooleanVar(value=merged.get("append_clip_label_to_caption", True)),
        }
        self.clip_labels_text: tk.Text | None = None
        self.v_status = tk.StringVar(value="Готов к работе")
        self.v_progress = tk.StringVar(value="Обработано: 0 / 0")
        self.v_summary = tk.StringVar(value="selected 0 · пограничные 0 · отклонено 0 · ошибки 0")
        self.v_model_note = tk.StringVar(value="AI-модели не загружаются, пока вы не включите соответствующие функции.")

    def _build_ui(self):
        style = self.style
        self._configure_theme_styles()

        top = ttk.Frame(self, padding=(12, 10, 12, 6))
        top.pack(fill="x")
        heading = ttk.Frame(top)
        heading.pack(fill="x")
        heading_text = ttk.Frame(heading)
        heading_text.pack(side="left", fill="x", expand=True)
        ttk.Label(heading_text, text="Dataset Curator", style="Header.TLabel").pack(anchor="w")
        ttk.Label(heading_text, text="Фильтрация изображений • оценка эстетики • лица • дубликаты • AI-подписи", style="Muted.TLabel").pack(anchor="w", pady=(2, 7))
        self.theme_button = ttk.Button(heading, text="☀️ Дневная тема" if self.theme_name == "dark" else "🌙 Ночная тема", command=self._toggle_theme)
        self.theme_button.pack(side="right", padx=(12, 0), pady=(2, 8))

        folder_frame = ttk.LabelFrame(self, text="Исходная папка", padding=8)
        folder_frame.pack(fill="x", padx=12, pady=(0, 8))
        folder_entry = ttk.Entry(folder_frame, textvariable=self.v["input_dir"])
        folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(folder_frame, text="Обзор…", command=self._browse_folder).pack(side="left")
        ttk.Button(folder_frame, text="Обновить каталог", command=self._refresh_all_review_tabs).pack(side="left", padx=(8, 0))

        # Основные действия размещены над вкладками, чтобы кнопка запуска
        # не терялась внизу окна на экранах с небольшим разрешением.
        action_bar = ttk.Frame(self, padding=(12, 0, 12, 8))
        action_bar.pack(fill="x")
        style.configure("Start.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        self.start_button = ttk.Button(
            action_bar,
            text="▶ СТАРТ — обработать изображения",
            command=self._start,
            style="Accent.TButton",
        )
        self.start_button.pack(side="left")
        self.caption_only_button = ttk.Button(
            action_bar,
            text="📝 Только подписи",
            command=self._start_caption_only,
        )
        self.caption_only_button.pack(side="left", padx=(8, 0))
        self.export_button = ttk.Button(action_bar, text="📦 Экспортировать набор", command=self._export_catalog)
        self.export_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(action_bar, text="■ Остановить", command=self._stop, state="disabled", style="Danger.TButton")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Папка базы SQLite", command=self._open_report).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Сохранить настройки", command=self._save_current_config).pack(side="right")
        ttk.Label(action_bar, text="Выбери папку, настрой фильтры и нажми СТАРТ", style="Muted.TLabel").pack(side="left", padx=(14, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.tab_filters = ttk.Frame(self.notebook, padding=12)
        self.tab_models = ttk.Frame(self.notebook, padding=12)
        self.tab_selected = ttk.Frame(self.notebook, padding=8)
        self.tab_rejected = ttk.Frame(self.notebook, padding=8)
        self.tab_log = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_filters, text="Фильтры")
        self.notebook.add(self.tab_models, text="AI-модели")
        self.notebook.add(self.tab_selected, text="Выбранные")
        self.notebook.add(self.tab_rejected, text="Отбракованные")
        self.notebook.add(self.tab_log, text="Журнал и отчёт")
        self._build_filters_tab()
        self._build_models_tab()
        self._build_review_tab(self.tab_selected, "selected")
        self._build_review_tab(self.tab_rejected, "rejected")
        self._build_log_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_review_tab_changed, add="+")

        footer = ttk.Frame(self, padding=(12, 0, 12, 12))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.v_status, style="Status.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(4, 3))
        row = ttk.Frame(footer)
        row.pack(fill="x")
        ttk.Label(row, textvariable=self.v_progress).pack(side="left")
        ttk.Label(row, textvariable=self.v_summary).pack(side="right")

    def _build_filters_tab(self):
        tab = self.tab_filters
        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        resolution = ttk.LabelFrame(left, text="Разрешение", padding=10)
        resolution.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(resolution, text="Проверять разрешение", variable=self.v["check_resolution"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(resolution, text="Минимум, px:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(resolution, textvariable=self.v["min_resolution"], width=10).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(resolution, text="Условие:").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Combobox(resolution, textvariable=self.v["resolution_mode"], values=["Одной стороной", "Обеими сторонами"], state="readonly", width=23).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(resolution, text="Одной стороной: max(ширина, высота) ≥ порога.\nОбеими: min(ширина, высота) ≥ порога.", style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        faces = ttk.LabelFrame(left, text="Лица", padding=10)
        faces.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(faces, text="Проверять наличие подходящего лица", variable=self.v["check_faces"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(faces, text="Мин. высота лица, % высоты кадра:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(faces, textvariable=self.v["min_face_percent"], width=8).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Checkbutton(faces, text="Отклонять фото с несколькими подходящими лицами", variable=self.v["reject_multiple_faces"]).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))

        blur = ttk.LabelFrame(left, text="Резкость", padding=10)
        blur.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(blur, text="Проверять размытие (Laplacian variance)", variable=self.v["check_blur"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(blur, text="Порог (ниже = размыто):").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(blur, textvariable=self.v["blur_threshold"], width=8).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))

        aesthetic = ttk.LabelFrame(right, text="Эстетическая оценка", padding=10)
        aesthetic.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(aesthetic, text="Использовать aesthetic-predictor", variable=self.v["check_aesthetic"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(aesthetic, text="Автоматически выбирать от:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(aesthetic, textvariable=self.v["aesthetic_keep"], width=8).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(aesthetic, text="Пограничная зона от:").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(aesthetic, textvariable=self.v["aesthetic_maybe"], width=8).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(aesthetic, text="Ниже пограничного порога статус станет «отклонено»; файлы останутся на месте.", style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        duplicates = ttk.LabelFrame(right, text="Похожие изображения", padding=10)
        duplicates.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(duplicates, text="После фильтрации искать дубликаты среди выбранных в SQLite", variable=self.v["find_duplicates"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(duplicates, text="Метод хеша:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Combobox(duplicates, textvariable=self.v["duplicate_hash"], values=["phash", "dhash", "whash", "ahash"], state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(duplicates, text="Расстояние Хэмминга:").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(duplicates, textvariable=self.v["duplicate_distance"], width=8).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Checkbutton(duplicates, text="Только показать, не менять статус", variable=self.v["duplicate_dry_run"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(duplicates, text="Порог 8 — умеренно строгий для 64-битного хеша. При отключённом тестовом режиме меняется только статус в SQLite.", style="Muted.TLabel", wraplength=390).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        warning = ttk.LabelFrame(right, text="Безопасность", padding=10)
        warning.pack(fill="x")
        ttk.Label(warning, text="Фото не перемещаются и не переименовываются. Статусы и оба описания хранятся в dataset_curator.sqlite3. Экспорт в отдельную папку выполняется только по кнопке «Экспортировать набор».", wraplength=410).pack(anchor="w")

    def _on_caption_only_toggle(self):
        # Отдельный режим сам включает генерацию подписей при построении Settings,
        # не меняя настройку создания подписей в обычном режиме фильтрации.
        self._schedule_save_config()

    def _build_models_tab(self):
        tab = self.tab_models
        captions = ttk.LabelFrame(tab, text="Автоматические подписи для датасета", padding=10)
        captions.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(captions, text="Генерировать описание EN/RU для выбранных изображений при фильтрации и сохранять в SQLite", variable=self.v["generate_captions"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            captions,
            text="ТОЛЬКО ПОДПИСИ — без фильтрации, оценки эстетики, детектора лиц и поиска дубликатов",
            variable=self.v["caption_only"],
            command=self._on_caption_only_toggle,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Checkbutton(
            captions,
            text="Пропускать изображения, у которых уже заполнено английское описание в SQLite",
            variable=self.v["skip_existing_captions"],
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(captions, text="Модель:").grid(row=3, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(captions, textvariable=self.v["caption_model"]).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))
        ttk.Label(captions, text="Префикс / trigger token:").grid(row=4, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(captions, textvariable=self.v["caption_prefix"]).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))
        ttk.Label(
            captions,
            text="Все описания хранятся в SQLite. Файлы .txt создаются только при явном экспорте выбранного набора. BLIP обычно пишет подписи на английском.",
            style="Muted.TLabel", wraplength=850,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            captions,
            text="Автоматически переводить английское описание на русский (MyMemory)",
            variable=self.v["translate_captions"],
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(captions, text="Email для бесплатного лимита MyMemory (необязательно):").grid(row=7, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(captions, textvariable=self.v["translation_email"]).grid(row=7, column=1, sticky="ew", padx=(8, 0), pady=(5, 0))
        ttk.Label(
            captions,
            text="MyMemory: бесплатный лимит около 5 000 символов/день без email или 50 000 с действительным email. Текст отправляется внешнему сервису для перевода.",
            style="Muted.TLabel", wraplength=850,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(5, 0))
        captions.columnconfigure(1, weight=1)

        clip = ttk.LabelFrame(tab, text="CLIP: классификация по заданным текстовым категориям", padding=10)
        clip.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Checkbutton(clip, text="Для выбранных изображений определять наиболее подходящую категорию", variable=self.v["classify_clip"]).pack(anchor="w")
        model_row = ttk.Frame(clip)
        model_row.pack(fill="x", pady=(7, 4))
        ttk.Label(model_row, text="Модель CLIP:").pack(side="left")
        ttk.Entry(model_row, textvariable=self.v["clip_model"]).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Label(clip, text="Категории (по одной на строку; минимум две):", padding=(0, 4, 0, 3)).pack(anchor="w")
        self.clip_labels_text = tk.Text(clip, height=8, wrap="word", undo=True, font=("Consolas", 9))
        self.clip_labels_text.pack(fill="both", expand=True)
        self.clip_labels_text.insert("1.0", self.config_data.get("clip_labels", DEFAULT_CLIP_LABELS))
        ttk.Checkbutton(clip, text="Добавлять выбранную CLIP-категорию в конец английского описания BLIP", variable=self.v["append_clip_label_to_caption"]).pack(anchor="w", pady=(7, 0))
        ttk.Label(clip, text="Важно: CLIP не генерирует описания — он сравнивает изображение с текстовыми категориями. Для генерации подписи используется BLIP. Уверенность CLIP относительная и зависит от набора категорий.", style="Muted.TLabel", wraplength=850).pack(anchor="w", pady=(7, 0))

        face_box = ttk.LabelFrame(tab, text="Модель MediaPipe для поиска лиц", padding=10)
        face_box.pack(fill="x")
        ttk.Label(face_box, text="Файл .tflite:").grid(row=0, column=0, sticky="w")
        ttk.Entry(face_box, textvariable=self.v["face_model_path"]).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(face_box, text="Обзор…", command=self._browse_face_model).grid(row=0, column=2)
        face_box.columnconfigure(1, weight=1)
        self.v_model_note.set("BLIP и CLIP опциональны. При включении потребуется transformers и PyTorch. Первое скачивание модели требует доступа к Hugging Face.")
        ttk.Label(tab, textvariable=self.v_model_note, style="Muted.TLabel", wraplength=900).pack(anchor="w", pady=(8, 0))

    def _build_review_tab(self, tab: ttk.Frame, group: str):
        """Прокручиваемый просмотр изображений с EN/RU-описаниями из SQLite."""
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, 6))

        ttk.Label(toolbar, textvariable=self._make_review_count_var(group), style="Header.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Обновить список", command=lambda g=group: self._refresh_review_tab(g, reset=True, sync_existing=True)).pack(side="right")
        ttk.Button(
            toolbar,
            text="Открыть папку",
            command=lambda g=group: self._open_review_folder(g),
        ).pack(side="right", padx=(0, 8))

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)

        canvas = tk.Canvas(
            body,
            highlightthickness=0,
            background=self.palette["app"],
            yscrollincrement=24,
        )
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        list_frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=list_frame, anchor="nw")
        list_frame.bind(
            "<Configure>",
            lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")),
            add="+",
        )
        canvas.bind(
            "<Configure>",
            lambda event, c=canvas, win=window_id: c.itemconfigure(win, width=event.width),
            add="+",
        )
        canvas.bind("<Enter>", lambda _event, c=canvas: c.bind_all("<MouseWheel>", lambda e: c.yview_scroll(int(-e.delta / 120), "units")), add="+")
        canvas.bind("<Leave>", lambda _event: self.unbind_all("<MouseWheel>"), add="+")

        footer = ttk.Frame(tab)
        footer.pack(fill="x", pady=(6, 0))
        load_more = ttk.Button(
            footer,
            text="Показать ещё",
            command=lambda g=group: self._refresh_review_tab(g, reset=False),
        )
        load_more.pack(side="left")
        self.review_load_more_buttons[group] = load_more
        self.review_canvases[group] = canvas
        self.review_list_frames[group] = list_frame

    def _make_review_count_var(self, group: str) -> tk.StringVar:
        var = tk.StringVar(value="Выбранные изображения" if group == "selected" else "Отбракованные изображения")
        self.review_count_vars[group] = var
        return var

    def _review_input_dir(self) -> Path | None:
        raw = self.v["input_dir"].get().strip()
        if not raw:
            return None
        try:
            return Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None

    def _open_review_folder(self, group: str):
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showinfo(APP_TITLE, "Сначала выберите существующую исходную папку.")
            return
        # В SQLite-режиме нет папок selected/rejected: оригиналы остаются в исходных местах.
        self._open_path(base)

    def _on_review_tab_changed(self, _event=None):
        """Не перечитывает базу и не пересоздаёт карточки при каждом возврате на вкладку."""
        try:
            selected_tab = self.notebook.select()
            if selected_tab == str(self.tab_selected):
                group = "selected"
            elif selected_tab == str(self.tab_rejected):
                group = "rejected"
            else:
                return

            base = self._review_input_dir()
            if base is None or not base.is_dir():
                self._clear_review_group(group, "Выберите существующую исходную папку выше.")
                return
            base_text = str(base)
            known_bases = [value for value in self.review_base_paths.values() if value]
            if known_bases and any(value != base_text for value in known_bases):
                self._refresh_all_review_tabs(sync_existing=True)
            elif not any(self.review_loaded.values()):
                if not any(self.review_loading.values()):
                    self._refresh_all_review_tabs(sync_existing=True)
            elif (
                not self.review_loaded.get(group, False)
                or self.review_dirty.get(group, True)
                or self.review_base_paths.get(group) != base_text
            ):
                if not self.review_loading.get(group, False):
                    self._refresh_review_tab(group, reset=True, sync_existing=False)
        except tk.TclError:
            pass

    def _catalog_for_current_folder(self, sync_existing: bool = False) -> CatalogStore | None:
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            return None
        catalog = CatalogStore(base)
        if sync_existing:
            added = catalog.sync_existing_files()
            if added:
                self._log(f"SQLite: добавлено записей об изображениях — {added}")
        return catalog

    def _review_files(self, group: str) -> list[Path]:
        """Синхронный вспомогательный метод для тестов; GUI загружает списки в фоне."""
        catalog = self._catalog_for_current_folder(sync_existing=True)
        if catalog is None:
            self.review_record_cache[group] = {}
            return []
        rows = catalog.list_images(group)
        self.review_record_cache[group] = {str(row["path"]): row for row in rows}
        return [Path(row["path"]) for row in rows]

    def _refresh_all_review_tabs(self, sync_existing: bool = True):
        """Один фоновый проход по папке и базе для обеих вкладок."""
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            self.review_catalog_request_id += 1
            for group in ("selected", "rejected"):
                self._clear_review_group(group, "Выберите существующую исходную папку выше.")
            return
        self._launch_review_catalog_load(("selected", "rejected"), base, sync_existing)

    def _refresh_review_tab(self, group: str, reset: bool = True, sync_existing: bool = False):
        if group not in self.review_list_frames:
            return
        if not reset:
            if not self.review_loaded.get(group) or self.review_dirty.get(group):
                self._refresh_review_tab(group, reset=True, sync_existing=False)
                return
            if self.review_loading.get(group) or self.review_rendering.get(group):
                return
            files = self.review_file_cache.get(group, [])
            self.review_render_target[group] = min(
                len(files), max(self.review_visible_count.get(group, 0), self.review_render_target.get(group, 0))
                + self.review_page_size,
            )
            if self.review_visible_count.get(group, 0) < self.review_render_target[group]:
                self.review_render_generation[group] += 1
                generation = self.review_render_generation[group]
                self.review_rendering[group] = True
                self._update_review_controls(group)
                self._render_review_batch(group, generation)
            return

        base = self._review_input_dir()
        if base is None or not base.is_dir():
            self._clear_review_group(group, "Выберите существующую исходную папку выше.")
            return
        if self.review_loading.get(group) and not sync_existing:
            return
        self._launch_review_catalog_load((group,), base, sync_existing)

    def _launch_review_catalog_load(
        self,
        groups: tuple[str, ...],
        base: Path,
        sync_existing: bool,
    ):
        self.review_catalog_request_id += 1
        request_id = self.review_catalog_request_id
        base_text = str(base)
        for group in groups:
            self.review_load_generation[group] = request_id
            self.review_render_generation[group] += 1  # отменяем отложенное построение старого списка
            self.review_rendering[group] = False
            self.review_loading[group] = True
            self.review_dirty[group] = True
            if not self.review_list_frames[group].winfo_children():
                ttk.Label(
                    self.review_list_frames[group],
                    text="Загрузка каталога в фоновом режиме…",
                    style="Muted.TLabel",
                    padding=20,
                ).pack(anchor="w")
            if group in self.review_count_vars:
                self.review_count_vars[group].set("Загрузка каталога…")
            self._update_review_controls(group)

        def load_catalog():
            rows_by_group: dict[str, list[dict[str, Any]]] = {}
            added = 0
            error = ""
            try:
                catalog = CatalogStore(base)
                if sync_existing:
                    added = catalog.sync_existing_files()
                for group_name in groups:
                    rows_by_group[group_name] = catalog.list_images(group_name)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self.event_queue.put((
                "review_catalog_result", request_id, base_text, groups,
                rows_by_group, added, error,
            ))

        worker = threading.Thread(
            target=load_catalog,
            daemon=True,
            name=f"DatasetCuratorReviewLoad-{request_id}",
        )
        self.review_load_workers[request_id] = worker
        worker.start()

    def _handle_review_catalog_result(self, event: tuple):
        _, request_id, base_text, groups, rows_by_group, added, error = event
        self.review_load_workers.pop(request_id, None)
        current_base = self._review_input_dir()
        current_base_text = str(current_base) if current_base is not None else ""
        for group in groups:
            if self.review_load_generation.get(group) != request_id:
                continue
            self.review_loading[group] = False
            if current_base_text != base_text:
                self.review_dirty[group] = True
                self._update_review_controls(group)
                continue
            if error:
                self.review_dirty[group] = True
                self.v_status.set(f"Не удалось загрузить каталог ({group})")
                self._log(f"Ошибка фоновой загрузки каталога {base_text}: {error}")
                self._update_review_controls(group)
                continue
            self._apply_review_rows(group, rows_by_group.get(group, []), base_text)
        if added:
            for group in ("selected", "rejected"):
                if group not in groups:
                    self.review_dirty[group] = True
            self._log(f"SQLite: добавлено записей об изображениях — {added}")

    def _apply_review_rows(self, group: str, rows: list[dict[str, Any]], base_text: str):
        """Обновляет только запрошенную вкладку, затем создаёт карточки небольшими порциями."""
        for key in [key for key in self.review_card_widgets if key[0] == group]:
            self.review_card_widgets.pop(key, None)
        list_frame = self.review_list_frames[group]
        for child in list_frame.winfo_children():
            child.destroy()

        self.review_record_cache[group] = {
            str(row["path"]): row for row in rows
        }
        self.review_file_cache[group] = [Path(row["path"]) for row in rows]
        self.review_base_paths[group] = base_text
        self.review_loaded[group] = True
        self.review_dirty[group] = False
        self.review_visible_count[group] = 0
        self.review_render_target[group] = min(self.review_page_size, len(rows))
        self.review_photo_refs[group] = []
        self.review_render_generation[group] += 1
        generation = self.review_render_generation[group]
        self.review_rendering[group] = bool(self.review_render_target[group])

        if not rows:
            base = self._review_input_dir()
            if base is None or not base.is_dir():
                text = "Выберите существующую исходную папку выше."
            elif group == "selected":
                text = "Выбранных изображений пока нет."
            else:
                text = "Пока нет отклонённых изображений."
            ttk.Label(list_frame, text=text, style="Muted.TLabel", padding=20).pack(anchor="w")

        self._update_review_controls(group)
        canvas = self.review_canvases.get(group)
        if canvas is not None:
            canvas.configure(scrollregion=canvas.bbox("all"))
        if self.review_rendering[group]:
            self._render_review_batch(group, generation)

    def _clear_review_group(self, group: str, message: str):
        self.review_load_generation[group] = self.review_catalog_request_id + 1
        self.review_loading[group] = False
        self.review_loaded[group] = False
        self.review_dirty[group] = True
        self.review_base_paths[group] = ""
        self.review_file_cache[group] = []
        self.review_record_cache[group] = {}
        self.review_visible_count[group] = 0
        self.review_render_target[group] = 0
        self.review_rendering[group] = False
        self.review_render_generation[group] += 1
        for key in [key for key in self.review_card_widgets if key[0] == group]:
            self.review_card_widgets.pop(key, None)
        self.review_photo_refs[group] = []
        list_frame = self.review_list_frames.get(group)
        if list_frame is not None:
            for child in list_frame.winfo_children():
                child.destroy()
            ttk.Label(list_frame, text=message, style="Muted.TLabel", padding=20).pack(anchor="w")
        self._update_review_controls(group)

    def _render_review_batch(self, group: str, generation: int):
        """Создаёт несколько карточек за проход, оставляя Tk event loop отзывчивым."""
        if generation != self.review_render_generation.get(group):
            return
        files = self.review_file_cache.get(group, [])
        list_frame = self.review_list_frames[group]
        started = time.perf_counter()
        created = 0
        target = min(self.review_render_target.get(group, 0), len(files))
        while (
            self.review_visible_count.get(group, 0) < target
            and created < 4
            and time.perf_counter() - started < 0.012
        ):
            index = self.review_visible_count[group]
            self._build_review_card(list_frame, group, files[index])
            self.review_visible_count[group] = index + 1
            created += 1

        canvas = self.review_canvases.get(group)
        if canvas is not None:
            canvas.configure(scrollregion=canvas.bbox("all"))
        more = self.review_visible_count.get(group, 0) < target
        self.review_rendering[group] = more
        self._update_review_controls(group)
        if more:
            self.after(1, lambda g=group, gen=generation: self._render_review_batch(g, gen))

    def _update_review_controls(self, group: str):
        total = len(self.review_file_cache.get(group, []))
        shown = self.review_visible_count.get(group, 0)
        label = "Выбранные" if group == "selected" else "Отбракованные"
        if group in self.review_count_vars:
            suffix = " · загрузка…" if self.review_loading.get(group) else ""
            self.review_count_vars[group].set(
                f"{label} — {total} файлов (показано {shown}){suffix}"
            )
        button = self.review_load_more_buttons.get(group)
        if button is not None:
            busy = self.review_loading.get(group, False) or self.review_rendering.get(group, False)
            if busy:
                button.configure(state="disabled", text="Подготовка списка…")
            elif shown < total:
                button.configure(
                    state="normal",
                    text=f"Показать ещё ({min(self.review_page_size, total - shown)})",
                )
            else:
                button.configure(
                    state="disabled",
                    text="Все изображения показаны" if total else "Нет изображений",
                )

    def _remove_review_card_from_cache(self, group: str, source: Path):
        """Удаляет одну карточку из текущей вкладки без пересоздания остальных."""
        source_text = str(source)
        files = self.review_file_cache.get(group, [])
        try:
            removed_index = files.index(source)
            files.pop(removed_index)
        except ValueError:
            removed_index = -1

        old_visible = self.review_visible_count.get(group, 0)
        card = self.review_card_widgets.pop((group, source_text), None)
        if card:
            frame = card.get("frame")
            if frame is not None:
                try:
                    frame.destroy()
                except tk.TclError:
                    pass
            photo = card.get("photo")
            if photo is not None:
                try:
                    self.review_photo_refs[group].remove(photo)
                except (ValueError, KeyError):
                    pass

        self.review_record_cache.get(group, {}).pop(source_text, None)
        if removed_index >= 0 and removed_index < old_visible:
            self.review_visible_count[group] = max(0, old_visible - 1)

        self.review_render_generation[group] += 1
        generation = self.review_render_generation[group]
        self.review_rendering[group] = False
        self.review_render_target[group] = min(
            len(files), max(self.review_visible_count[group], self.review_render_target.get(group, 0))
        )
        if self.review_visible_count[group] < self.review_render_target[group]:
            self.review_rendering[group] = True
            self.after(1, lambda g=group, gen=generation: self._render_review_batch(g, gen))

        list_frame = self.review_list_frames.get(group)
        if list_frame is not None and not files:
            for child in list_frame.winfo_children():
                child.destroy()
            ttk.Label(
                list_frame,
                text="В этой вкладке пока нет изображений.",
                style="Muted.TLabel",
                padding=20,
            ).pack(anchor="w")
        canvas = self.review_canvases.get(group)
        if canvas is not None:
            canvas.configure(scrollregion=canvas.bbox("all"))
        self._update_review_controls(group)

    def _build_review_card(self, list_frame: ttk.Frame, group: str, source: Path):
        card = ttk.LabelFrame(list_frame, text=f"{source.name}  ·  {source.parent.name}", padding=8)
        card.pack(fill="x", padx=5, pady=5)

        record = self.review_record_cache.get(group, {}).get(str(source)) or {}
        thumbnail = None

        preview_frame = ttk.Frame(card, width=166, height=184)
        preview_frame.pack(side="left", fill="y", padx=(0, 10))
        preview_frame.pack_propagate(False)

        stored_width = record.get("width")
        stored_height = record.get("height")
        try:
            stored_width, stored_height = int(stored_width or 0), int(stored_height or 0)
        except (TypeError, ValueError):
            stored_width = stored_height = 0
        size_text = (
            f"{stored_width:,} × {stored_height:,} px".replace(",", " ")
            if stored_width > 0 and stored_height > 0
            else "Размер неизвестен"
        )
        preview_size_var = tk.StringVar(value=size_text)
        size_label = ttk.Label(
            preview_frame, textvariable=preview_size_var, style="Muted.TLabel", anchor="center"
        )
        size_label.pack(side="bottom", fill="x", pady=(4, 0))

        try:
            from PIL import Image, ImageOps, ImageTk
            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image_width, image_height = image.size
                preview_size_var.set(
                    f"{image_width:,} × {image_height:,} px".replace(",", " ")
                )
                image.thumbnail((144, 144))
                thumbnail = ImageTk.PhotoImage(image.copy())
            self.review_photo_refs[group].append(thumbnail)
            image_label = ttk.Label(preview_frame, image=thumbnail, anchor="center", cursor="hand2")
            image_label.image = thumbnail
            image_label.pack(fill="both", expand=True)
            image_label.bind(
                "<Button-1>", lambda _event, p=source: self._show_review_image(p), add="+"
            )
            image_label.bind(
                "<Enter>", lambda _event, widget=image_label: widget.configure(cursor="hand2"), add="+"
            )
        except Exception as exc:
            ttk.Label(
                preview_frame, text=f"Не удалось\nзагрузить фото\n{exc}", wraplength=145
            ).pack(fill="both", expand=True)

        content = ttk.Frame(card)
        content.pack(side="left", fill="both", expand=True)
        path_key = str(source)
        caption_value = self.review_caption_drafts.get(path_key, str(record.get("description_en") or ""))
        caption_ru_value = self.review_caption_ru_drafts.get(path_key, str(record.get("description_ru") or ""))

        caption_header = ttk.Frame(content)
        caption_header.pack(fill="x")
        ttk.Label(caption_header, text="Описание на английском (для LoRA)", style="Muted.TLabel").pack(side="left")
        caption_state = ttk.Label(
            caption_header,
            text="Сохранено в SQLite" if (record.get("description_en") or record.get("description_ru")) else "Описание отсутствует",
            style="Muted.TLabel",
        )
        caption_state.pack(side="right")

        caption_text = tk.Text(content, height=3, wrap="word", undo=True, font=("Segoe UI", 9))
        caption_text.pack(fill="x", expand=True, pady=(3, 5))
        if caption_value:
            caption_text.insert("1.0", caption_value.rstrip("\r\n"))

        ttk.Label(content, text="Перевод на русский (для просмотра и редактирования)", style="Muted.TLabel").pack(anchor="w")
        caption_ru_text = tk.Text(content, height=3, wrap="word", undo=True, font=("Segoe UI", 9))
        caption_ru_text.pack(fill="x", expand=True, pady=(3, 6))
        if caption_ru_value:
            caption_ru_text.insert("1.0", caption_ru_value.rstrip("\r\n"))

        def remember_draft(_event=None, p=path_key, widget=caption_text, ru=False):
            try:
                target = self.review_caption_ru_drafts if ru else self.review_caption_drafts
                self.after_idle(lambda: target.__setitem__(p, widget.get("1.0", "end-1c")))
            except tk.TclError:
                pass

        for sequence in ("<KeyRelease>", "<<Paste>>", "<<Cut>>", "<<Undo>>", "<<Redo>>"):
            caption_text.bind(sequence, lambda event, p=path_key, t=caption_text: remember_draft(event, p, t, False), add="+")
            caption_ru_text.bind(sequence, lambda event, p=path_key, t=caption_ru_text: remember_draft(event, p, t, True), add="+")

        buttons = ttk.Frame(content)
        buttons.pack(fill="x")
        save_button = ttk.Button(
            buttons,
            text="Сохранить оба описания",
            command=lambda p=source, en=caption_text, ru=caption_ru_text: self._save_review_caption(p, en, ru),
        )
        save_button.pack(side="left")
        translate_button = ttk.Button(
            buttons,
            text="Перевести EN → RU",
            command=lambda p=source, en=caption_text, ru=caption_ru_text, g=group: self._translate_review_caption(p, en, ru, g),
        )
        translate_button.pack(side="left", padx=(6, 0))
        generate_button = ttk.Button(
            buttons,
            text="Сгенерировать EN + RU",
            command=lambda p=source, en=caption_text, ru=caption_ru_text, g=group: self._generate_review_caption(p, en, ru, g),
        )
        generate_button.pack(side="left", padx=(6, 0))
        move_text = "Отклонить (только статус)" if group == "selected" else "Вернуть в выбранные"
        move_style = "Danger.TButton" if group == "selected" else "Accent.TButton"
        move_button = ttk.Button(
            buttons,
            text=move_text,
            style=move_style,
            command=lambda p=source, g=group, en=caption_text, ru=caption_ru_text: self._move_review_image(p, g, en, ru),
        )
        move_button.pack(side="left", padx=(6, 0))
        ttk.Button(
            buttons,
            text="Открыть фото",
            command=lambda p=source: self._open_path(p),
        ).pack(side="right")

        self.review_card_widgets[(group, str(source))] = {
            "caption": caption_text,
            "caption_ru": caption_ru_text,
            "generate": generate_button,
            "translate": translate_button,
            "save": save_button,
            "move": move_button,
            "caption_state": caption_state,
            "group": group,
            "path": source,
            "frame": card,
            "photo": thumbnail,
        }

    def _show_review_image(self, source: Path):
        """Показывает исходное изображение в отдельном окне с масштабированием."""
        if not source.is_file():
            messagebox.showerror(APP_TITLE, f"Файл изображения не найден:\n{source}")
            return

        try:
            from PIL import Image, ImageOps, ImageTk
            with Image.open(source) as raw:
                full_image = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось открыть изображение:\n{source}\n\n{exc}")
            return

        image_width, image_height = full_image.size
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        max_image_width = max(320, int(screen_width * 0.88))
        max_image_height = max(240, int(screen_height * 0.74))
        fit_zoom = min(max_image_width / image_width, max_image_height / image_height)
        # Небольшие изображения можно увеличить до 200%; крупные сразу помещаются на экран.
        initial_zoom = max(0.05, min(2.0, fit_zoom))

        window_width = min(screen_width - 30, max(640, int(image_width * initial_zoom) + 48))
        window_height = min(screen_height - 50, max(480, int(image_height * initial_zoom) + 112))
        dialog = tk.Toplevel(self)
        dialog.title(
            f"{source.name} — {image_width:,} × {image_height:,} px".replace(",", " ")
        )
        dialog.transient(self)
        dialog.geometry(f"{window_width}x{window_height}")
        dialog.minsize(min(520, screen_width - 30), min(360, screen_height - 50))

        container = ttk.Frame(dialog, padding=8)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container)
        toolbar.pack(fill="x")
        ttk.Label(
            toolbar,
            text=f"Исходный размер: {image_width:,} × {image_height:,} px".replace(",", " "),
            style="Muted.TLabel",
        ).pack(side="left")

        zoom_var = tk.StringVar(value="")
        state = {"zoom": initial_zoom}
        photo_ref = {"image": None}

        ttk.Button(
            toolbar, text="−", width=3,
            command=lambda: redraw(state["zoom"] / 1.25),
        ).pack(side="right", padx=(4, 0))
        ttk.Button(
            toolbar, text="+", width=3,
            command=lambda: redraw(state["zoom"] * 1.25),
        ).pack(side="right", padx=(4, 0))
        ttk.Label(toolbar, textvariable=zoom_var, width=6, anchor="center").pack(side="right")
        ttk.Button(
            toolbar, text="100%",
            command=lambda: redraw(1.0),
        ).pack(side="right", padx=(8, 0))
        ttk.Button(
            toolbar, text="Уместить",
            command=lambda: redraw(min(1.0, max_image_width / image_width, max_image_height / image_height)),
        ).pack(side="right", padx=(8, 0))
        ttk.Button(toolbar, text="Закрыть", command=dialog.destroy).pack(side="right", padx=(8, 0))

        image_frame = ttk.Frame(container)
        image_frame.pack(fill="both", expand=True, pady=(8, 0))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(
            image_frame,
            highlightthickness=0,
            background=self.palette["app"],
            xscrollincrement=32,
            yscrollincrement=32,
        )
        xscroll = ttk.Scrollbar(image_frame, orient="horizontal", command=canvas.xview)
        yscroll = ttk.Scrollbar(image_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        image_item = canvas.create_image(0, 0, anchor="nw")

        def redraw(zoom: float):
            zoom = max(0.1, min(8.0, float(zoom)))
            resized_size = (
                max(1, int(round(image_width * zoom))),
                max(1, int(round(image_height * zoom))),
            )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized = full_image.resize(resized_size, resampling)
            photo = ImageTk.PhotoImage(resized, master=dialog)
            canvas.itemconfigure(image_item, image=photo)
            canvas.configure(scrollregion=(0, 0, resized_size[0], resized_size[1]))
            photo_ref["image"] = photo
            state["zoom"] = zoom
            zoom_var.set(f"{zoom * 100:.0f}%")

        def zoom_with_wheel(event):
            factor = 1.15 if event.delta > 0 else 1 / 1.15
            redraw(state["zoom"] * factor)
            return "break"

        canvas.bind("<Control-MouseWheel>", zoom_with_wheel, add="+")
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"), add="+")
        dialog.bind("<Escape>", lambda _event: dialog.destroy(), add="+")
        redraw(initial_zoom)

    def _review_is_busy(self) -> bool:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(APP_TITLE, "Основная обработка ещё выполняется. Дождитесь её окончания.")
            return True
        if self.export_worker and self.export_worker.is_alive():
            messagebox.showwarning(APP_TITLE, "Экспорт набора ещё выполняется. Дождитесь его завершения.")
            return True
        if self.review_workers:
            messagebox.showwarning(APP_TITLE, "Дождитесь завершения текущей генерации подписи в карточке изображения.")
            return True
        return False

    def _review_group_for_path(self, source: Path) -> str:
        for group, path_text in self.review_card_widgets:
            if path_text == str(source):
                return group
        return "selected"

    def _save_review_caption(
        self,
        source: Path,
        caption_widget: tk.Text,
        russian_widget: tk.Text,
        show_message: bool = True,
    ) -> bool:
        """Сохраняет английское описание и русский перевод только в SQLite."""
        if self._review_is_busy():
            return False
        description_en = caption_widget.get("1.0", "end-1c").strip()
        description_ru = russian_widget.get("1.0", "end-1c").strip()
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showerror(APP_TITLE, "Выберите существующую исходную папку.")
            return False
        try:
            source.resolve().relative_to(base.resolve())
        except (OSError, RuntimeError, ValueError):
            messagebox.showerror(APP_TITLE, "Изображение не относится к текущему каталогу. Обновите список.")
            return False

        catalog = CatalogStore(base)
        previous = catalog.get_image(source) or {}
        if not description_en and not description_ru and (
            str(previous.get("description_en") or "").strip() or str(previous.get("description_ru") or "").strip()
        ):
            if not messagebox.askyesno(
                APP_TITLE,
                "Оба поля описания пусты. Очистить английское описание и перевод в SQLite?",
                icon="warning",
            ):
                return False

        try:
            catalog.set_descriptions(source, description_en, description_ru)
            self.review_caption_drafts.pop(str(source), None)
            self.review_caption_ru_drafts.pop(str(source), None)
            for key, card in self.review_card_widgets.items():
                if key[1] == str(source):
                    card["caption_state"].configure(text="Оба описания сохранены в SQLite")
            for group in ("selected", "rejected"):
                row = self.review_record_cache.get(group, {}).get(str(source))
                if row is not None:
                    row["description_en"] = description_en
                    row["description_ru"] = description_ru
            self._log(f"Описания EN/RU сохранены в SQLite: {source}")
            if show_message:
                self.v_status.set(f"Описания сохранены: {source.name}")
            return True
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось сохранить описания в SQLite:\n{type(exc).__name__}: {exc}")
            return False

    def _move_review_image(
        self,
        source: Path,
        group: str,
        caption_widget: tk.Text,
        russian_widget: tk.Text,
    ):
        """Меняет только status в SQLite; изображение и любые текстовые файлы не перемещаются."""
        if self._review_is_busy():
            return
        if not source.is_file():
            messagebox.showwarning(APP_TITLE, f"Исходное изображение не найдено:\n{source}")
            self._refresh_review_tab(group)
            return

        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showerror(APP_TITLE, "Выберите существующую исходную папку.")
            return
        try:
            source.resolve().relative_to(base.resolve())
        except (OSError, RuntimeError, ValueError):
            messagebox.showwarning(
                APP_TITLE,
                "Карточка относится к другой исходной папке. Обновите список и повторите действие.",
            )
            self._refresh_all_review_tabs()
            return

        # Сначала сохраняем текущие правки в БД. Файловая система не меняется.
        description_en = caption_widget.get("1.0", "end-1c").strip()
        description_ru = russian_widget.get("1.0", "end-1c").strip()
        draft_key = str(source)
        if description_en or description_ru:
            if not self._save_review_caption(source, caption_widget, russian_widget, show_message=False):
                return

        new_status = "rejected" if group == "selected" else "selected"
        try:
            CatalogStore(base).set_status(
                source,
                new_status,
                reason="Статус изменён вручную в интерфейсе Dataset Curator",
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось изменить статус в SQLite:\n{type(exc).__name__}: {exc}")
            return

        self.review_caption_drafts.pop(draft_key, None)
        self.review_caption_ru_drafts.pop(draft_key, None)
        self._log(f"Статус изменён в SQLite: {source} → {new_status}; файл не перемещался")
        self.v_status.set(f"Статус изменён: {source.name}")
        target_group = "rejected" if group == "selected" else "selected"
        self._remove_review_card_from_cache(group, source)
        # Вторую вкладку обновим в фоне только при переходе на неё.
        self.review_dirty[target_group] = True


    def _translate_review_caption(
        self,
        source: Path,
        english_widget: tk.Text,
        russian_widget: tk.Text,
        group: str,
    ):
        """Переводит текущий текст EN в редакторе; запись в SQLite выполняется кнопкой сохранения."""
        if self._review_is_busy():
            return
        english_text = english_widget.get("1.0", "end-1c").strip()
        if not english_text:
            messagebox.showinfo(APP_TITLE, "Сначала заполните английское описание или сгенерируйте его через BLIP.")
            return
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showerror(APP_TITLE, "Выберите существующую исходную папку.")
            return
        try:
            source.resolve().relative_to(base.resolve())
        except (OSError, RuntimeError, ValueError):
            messagebox.showerror(APP_TITLE, "Изображение относится к другому каталогу. Обновите список.")
            return

        email = self.v["translation_email"].get().strip()
        key = (group, str(source))
        card = self.review_card_widgets.get(key)
        if card:
            for widget_name in ("generate", "translate", "caption", "caption_ru", "save", "move"):
                card[widget_name].configure(state="disabled")
        self.v_status.set(f"Перевод EN → RU: {source.name}")
        self._log(f"Перевод существующего описания: {source.name}")

        def work():
            translated = ""
            error = ""
            try:
                translated = CatalogStore(base).translate_en_ru(english_text, email)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self.event_queue.put(("review_translation_result", group, str(source), translated, error))

        worker = threading.Thread(target=work, daemon=True, name="DatasetCuratorReviewTranslation")
        self.review_workers[key] = worker
        worker.start()

    def _generate_review_caption(
        self,
        source: Path,
        caption_widget: tk.Text,
        russian_widget: tk.Text,
        group: str,
    ):
        if self._review_is_busy():
            return
        if not source.is_file():
            messagebox.showwarning(APP_TITLE, f"Исходное изображение не найдено:\n{source}")
            self._refresh_review_tab(group)
            return

        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showerror(APP_TITLE, "Выберите существующую исходную папку.")
            return
        try:
            source.resolve().relative_to(base.resolve())
        except (OSError, RuntimeError, ValueError):
            messagebox.showerror(APP_TITLE, "Изображение относится к другому каталогу. Обновите список.")
            return

        current_en = caption_widget.get("1.0", "end-1c").strip()
        current_ru = russian_widget.get("1.0", "end-1c").strip()
        catalog = CatalogStore(base)
        saved = catalog.get_image(source) or {}
        saved_en = str(saved.get("description_en") or "").strip()
        saved_ru = str(saved.get("description_ru") or "").strip()
        if (current_en != saved_en or current_ru != saved_ru) and (current_en or current_ru):
            if not messagebox.askyesno(
                APP_TITLE,
                "В редакторе есть несохранённые изменения EN/RU. Заменить оба поля новым результатом AI?\n"
                "Откажитесь, если сначала нужно сохранить ручные правки.",
                icon="warning",
            ):
                return

        try:
            settings = self._get_settings()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        try:
            if importlib.util.find_spec("transformers") is None:
                messagebox.showerror(
                    APP_TITLE,
                    "Для генерации подписи необходим пакет transformers.\n\n"
                    f"Установите его командой:\n\"{sys.executable}\" -m pip install \"transformers>=4.45,<5.0\" \"accelerate>=0.30\"\n\n"
                    "Либо запустите install_optional_ai.bat и повторите попытку.",
                )
                return
        except (ImportError, ValueError, ModuleNotFoundError):
            messagebox.showerror(APP_TITLE, "Не удалось найти пакет transformers в текущем окружении Python.")
            return

        settings.generate_captions = True
        settings.caption_only = False
        key = (group, str(source))
        card = self.review_card_widgets.get(key)
        if card:
            for widget_name in ("generate", "translate", "caption", "caption_ru", "save", "move"):
                card[widget_name].configure(state="disabled")
        self.v_status.set(f"Генерация описаний EN/RU: {source.name}")
        self._log(f"Генерация описаний для карточки: {source.name}")

        def work():
            services = ModelServices(settings, lambda line: self.event_queue.put(("log", f"[AI] {line}")))
            caption = ""
            caption_ru = ""
            error = ""
            try:
                from PIL import Image, ImageOps
                services.load_caption_model()
                with Image.open(source) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                caption = services.caption(image)
                prefix = settings.caption_prefix.strip().strip(",;")
                if prefix and not caption.casefold().startswith(prefix.casefold()):
                    caption = f"{prefix}, {caption}".strip().strip(",")
                if settings.classify_clip:
                    try:
                        services.load_clip_model()
                        category, score = services.classify(image)
                        self.event_queue.put(("log", f"CLIP: {category} ({score * 100:.1f}% среди категорий)"))
                        if settings.append_clip_label_to_caption and category and category.casefold() not in caption.casefold():
                            caption = f"{caption.rstrip(' .,;')} , {category}".replace(" ,", ",")
                    except Exception as exc:
                        self.event_queue.put(("log", f"CLIP недоступен, оставлена подпись BLIP: {exc}"))
                if settings.translate_captions:
                    try:
                        caption_ru = catalog.translate_en_ru(caption, settings.translation_email)
                    except Exception as exc:
                        self.event_queue.put(("log", f"Перевод MyMemory не выполнен: {type(exc).__name__}: {exc}"))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                services.close()
                self.event_queue.put(("review_caption_result", group, str(source), caption, caption_ru, error))

        worker = threading.Thread(target=work, daemon=True, name="DatasetCuratorReviewCaption")
        self.review_workers[key] = worker
        worker.start()

    def _build_log_tab(self):
        toolbar = ttk.Frame(self.tab_log)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="Очистить журнал", command=self._clear_log).pack(side="left")
        ttk.Button(toolbar, text="Открыть папку отчёта", command=self._open_input).pack(side="left", padx=(8, 0))
        self.log_text = tk.Text(self.tab_log, wrap="word", height=20, state="disabled", font=("Consolas", 9), undo=False)
        scroll = ttk.Scrollbar(self.tab_log, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Выберите папку с изображениями", initialdir=self.v["input_dir"].get() or str(Path.home()))
        if path:
            self.v["input_dir"].set(path)
            self.after(100, self._refresh_all_review_tabs)

    def _browse_face_model(self):
        path = filedialog.askopenfilename(title="Выберите модель MediaPipe", filetypes=[("MediaPipe TFLite", "*.tflite"), ("Все файлы", "*.*")], initialdir=str(APP_DIR))
        if path:
            self.v["face_model_path"].set(path)

    def _open_input(self):
        path = Path(self.v["input_dir"].get().strip())
        if path.is_dir():
            self._open_path(path)
        else:
            messagebox.showwarning(APP_TITLE, "Сначала выберите существующую папку.")

    def _open_output(self, folder: str):
        base = Path(self.v["input_dir"].get().strip())
        path = base / folder
        if path.is_dir():
            self._open_path(path)
        else:
            messagebox.showinfo(APP_TITLE, f"Папка пока не создана:\n{path}")

    @staticmethod
    def _open_path(path: Path):
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось открыть папку:\n{exc}")

    def _open_report(self):
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showinfo(APP_TITLE, "Сначала выберите существующую папку датасета.")
            return
        catalog = CatalogStore(base)
        catalog.sync_existing_files()
        self._open_path(base)
        self.v_status.set(f"База SQLite: {catalog.db_path}")

    def _export_catalog(self):
        if self._review_is_busy():
            return
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showerror(APP_TITLE, "Сначала выберите существующую папку датасета.")
            return
        chosen = filedialog.askdirectory(
            title="Выберите отдельную папку для экспорта готового набора",
            initialdir=str(base.parent),
            mustexist=True,
        )
        if not chosen:
            return
        output = Path(chosen).expanduser().resolve()
        try:
            output.relative_to(base.resolve())
        except ValueError:
            pass
        else:
            messagebox.showerror(
                APP_TITLE,
                "Папка экспорта должна быть отдельной от исходного каталога и не располагаться внутри него.",
            )
            return
        if output == base.resolve():
            messagebox.showerror(APP_TITLE, "Нельзя экспортировать поверх исходного каталога.")
            return
        if output.is_dir():
            try:
                folder_has_files = next(output.iterdir(), None) is not None
            except OSError as exc:
                messagebox.showerror(APP_TITLE, f"Не удалось проверить папку экспорта:\n{exc}")
                return
            if folder_has_files:
                messagebox.showerror(
                    APP_TITLE,
                    "Для безопасного экспорта выберите пустую отдельную папку. Существующие файлы не будут перезаписаны.",
                )
                return
        if not messagebox.askyesno(
            APP_TITLE,
            f"Экспортировать выбранные изображения из SQLite в папку:\n{output}\n\n"
            "Будут скопированы только выбранные/пограничные записи с непустым английским описанием. "
            "Для каждого фото создастся .txt с английским описанием; переводы будут добавлены в captions_ru.csv.\n\n"
            "Проверка каталога и копирование выполнятся в фоне. "
            "Исходные изображения и база данных останутся без изменений. Продолжить?",
            icon="question",
        ):
            return

        self.export_button.configure(state="disabled")
        self.v_status.set("Экспорт набора выполняется в фоновом режиме…")

        def export_work():
            exported = skipped = added = 0
            empty = False
            error = ""
            output_text = str(output)
            try:
                catalog = CatalogStore(base)
                added = catalog.sync_existing_files()
                if not catalog.list_images("selected"):
                    empty = True
                else:
                    exported, skipped, target = catalog.export_selected(
                        output, include_russian_csv=True
                    )
                    output_text = str(target)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self.event_queue.put((
                "export_result", str(base), output_text, exported, skipped, error, empty, added,
            ))

        self.export_worker = threading.Thread(
            target=export_work,
            daemon=True,
            name="DatasetCuratorExport",
        )
        self.export_worker.start()

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _get_settings(self) -> Settings:
        input_dir = self.v["input_dir"].get().strip()
        if not input_dir or not Path(input_dir).expanduser().is_dir():
            raise ValueError("Выберите существующую исходную папку.")
        try:
            min_resolution = int(self.v["min_resolution"].get())
            min_face_percent = float(self.v["min_face_percent"].get().replace(",", "."))
            blur_threshold = float(self.v["blur_threshold"].get().replace(",", "."))
            keep = float(self.v["aesthetic_keep"].get().replace(",", "."))
            maybe = float(self.v["aesthetic_maybe"].get().replace(",", "."))
            dup_distance = int(self.v["duplicate_distance"].get())
        except ValueError:
            raise ValueError("Проверьте числовые настройки: разрешение, проценты, пороги, расстояние хеша.")
        if min_resolution < 1:
            raise ValueError("Минимальное разрешение должно быть положительным числом.")
        if not 0 < min_face_percent <= 100:
            raise ValueError("Минимальная высота лица должна быть в диапазоне 0–100%.")
        if blur_threshold < 0:
            raise ValueError("Порог размытия не может быть отрицательным.")
        if not -100 <= maybe <= 100 or not -100 <= keep <= 100 or maybe > keep:
            raise ValueError("Порог «Пограничная зона» должен быть не выше порога «selected».")
        if not 0 <= dup_distance <= 64:
            raise ValueError("Расстояние Хэмминга должно быть в диапазоне 0–64 для 64-битного хеша.")
        clip_labels = self.clip_labels_text.get("1.0", "end").strip() if self.clip_labels_text else DEFAULT_CLIP_LABELS
        if self.v["classify_clip"].get() and len([x for x in clip_labels.splitlines() if x.strip()]) < 2:
            raise ValueError("Для CLIP нужно ввести минимум две категории, по одной на строку.")
        caption_only = self.v["caption_only"].get()
        return Settings(
            input_dir=str(Path(input_dir).expanduser().resolve()),
            clean_outputs=self.v["clean_outputs"].get(),
            check_resolution=self.v["check_resolution"].get(), min_resolution=min_resolution,
            resolution_mode=self.v["resolution_mode"].get(), check_faces=self.v["check_faces"].get(),
            face_model_path=self.v["face_model_path"].get().strip(), min_face_percent=min_face_percent,
            reject_multiple_faces=self.v["reject_multiple_faces"].get(), check_blur=self.v["check_blur"].get(),
            blur_threshold=blur_threshold, check_aesthetic=self.v["check_aesthetic"].get(),
            aesthetic_keep=keep, aesthetic_maybe=maybe, find_duplicates=self.v["find_duplicates"].get(),
            duplicate_hash=self.v["duplicate_hash"].get(), duplicate_distance=dup_distance,
            duplicate_dry_run=self.v["duplicate_dry_run"].get(),
            generate_captions=self.v["generate_captions"].get() or caption_only,
            caption_only=caption_only,
            skip_existing_captions=self.v["skip_existing_captions"].get(),
            translate_captions=self.v["translate_captions"].get(),
            translation_email=self.v["translation_email"].get().strip(),
            caption_model=self.v["caption_model"].get().strip() or "Salesforce/blip-image-captioning-base",
            caption_prefix=self.v["caption_prefix"].get().strip(),
            classify_clip=self.v["classify_clip"].get(), clip_model=self.v["clip_model"].get().strip() or "openai/clip-vit-base-patch32",
            clip_labels=clip_labels, append_clip_label_to_caption=self.v["append_clip_label_to_caption"].get(),
        )

    def _settings_as_config(self, settings: Settings) -> dict[str, Any]:
        return asdict(settings)

    def _collect_config(self) -> dict[str, Any]:
        """Снимает значения всех контролов даже тогда, когда форма заполнена не полностью."""
        data = {key: var.get() for key, var in self.v.items()}
        data["clip_labels"] = self.clip_labels_text.get("1.0", "end-1c") if self.clip_labels_text else DEFAULT_CLIP_LABELS
        data["theme"] = self.theme_name
        current_state = "normal"
        try:
            if self.state() == "zoomed":
                current_state = "zoomed"
        except tk.TclError:
            pass
        data["window_state"] = current_state
        data["window_geometry"] = self._normal_geometry
        if current_state == "normal":
            try:
                geometry = self.geometry()
                if re.fullmatch(r"\d+x\d+[+-]\d+[+-]\d+", geometry):
                    data["window_geometry"] = geometry
            except tk.TclError:
                pass
        return data

    def _save_config_now(self, log_result: bool = False):
        data = self._collect_config()
        success = save_config(data)
        if success:
            if data.get("window_state") == "normal":
                self._normal_geometry = data.get("window_geometry", self._normal_geometry)
            if log_result:
                self._log("Настройки и расположение окна сохранены.")
        elif log_result:
            self._log(f"Не удалось записать настройки в {CONFIG_FILE}.")

    def _schedule_save_config(self, *_args):
        """Автосохранение с небольшой задержкой, чтобы не писать файл на каждую букву."""
        if self._config_save_after_id is not None:
            try:
                self.after_cancel(self._config_save_after_id)
            except tk.TclError:
                pass
        try:
            self._config_save_after_id = self.after(650, self._save_config_now)
        except tk.TclError:
            self._config_save_after_id = None

    def _attach_auto_save(self):
        for variable in self.v.values():
            try:
                variable.trace_add("write", self._schedule_save_config)
            except (AttributeError, tk.TclError):
                pass
        if self.clip_labels_text is not None:
            self.clip_labels_text.bind("<<Modified>>", self._on_clip_labels_modified, add="+")
            self.clip_labels_text.edit_modified(False)

    def _on_clip_labels_modified(self, _event=None):
        if self.clip_labels_text is not None and self.clip_labels_text.edit_modified():
            self.clip_labels_text.edit_modified(False)
            self._schedule_save_config()

    def _on_root_configure(self, event):
        if event.widget is not self:
            return
        try:
            if self.state() == "normal":
                geometry = self.geometry()
                if re.fullmatch(r"\d+x\d+[+-]\d+[+-]\d+", geometry):
                    self._normal_geometry = geometry
        except tk.TclError:
            pass
        self._schedule_save_config()

    def _save_current_config(self):
        if self._config_save_after_id is not None:
            try:
                self.after_cancel(self._config_save_after_id)
            except tk.TclError:
                pass
            self._config_save_after_id = None
        self._save_config_now(log_result=True)

    def _configure_theme_styles(self):
        """Централизованно окрашивает ttk-элементы для активной палитры."""
        p = self.palette
        style = self.style
        style.configure(".", font=("Segoe UI", 9), foreground=p["text"], background=p["panel"])
        style.configure("TFrame", background=p["panel"])
        style.configure("TLabelframe", background=p["panel"], bordercolor=p["border"],
                        lightcolor=p["border"], darkcolor=p["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=p["panel"], foreground=p["accent"],
                        font=("Segoe UI", 9, "bold"), padding=(4, 0))
        style.configure("TLabel", background=p["panel"], foreground=p["text"])
        style.configure("Header.TLabel", background=p["panel"], foreground=p["text"],
                        font=("Segoe UI", 14, "bold"))
        style.configure("Muted.TLabel", background=p["panel"], foreground=p["muted"], font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=p["panel"], foreground=p["text"], font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=p["panel"], foreground=p["text"], padding=(3, 3))
        style.map("TCheckbutton", background=[("active", p["panel"])], foreground=[("disabled", p["disabled_text"])])
        style.configure("TRadiobutton", background=p["panel"], foreground=p["text"], padding=(3, 3))
        style.map("TRadiobutton", background=[("active", p["panel"])])
        style.configure("TEntry", fieldbackground=p["white"], foreground=p["text"], insertcolor=p["text"],
                        bordercolor=p["border"], lightcolor=p["border"], darkcolor=p["border"], padding=5)
        style.configure("TSpinbox", fieldbackground=p["white"], foreground=p["text"], bordercolor=p["border"], padding=4)
        style.configure("TCombobox", fieldbackground=p["white"], foreground=p["text"], background=p["button"],
                        arrowcolor=p["text"], bordercolor=p["border"])
        style.map("TCombobox", fieldbackground=[("readonly", p["white"])], foreground=[("disabled", p["disabled_text"])])
        style.configure("TButton", background=p["button"], foreground=p["text"], bordercolor=p["border"], padding=(10, 6), relief="flat")
        style.map("TButton", background=[("disabled", p["button_disabled"]), ("pressed", p["button_pressed"]), ("active", p["button_active"])],
                  foreground=[("disabled", p["disabled_text"])])
        style.configure("Accent.TButton", background=p["accent"], foreground=p["accent_text"], bordercolor=p["accent"],
                        padding=(12, 7), font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("Accent.TButton", background=[("disabled", p["accent_disabled"]), ("pressed", p["accent_active"]), ("active", p["accent_active"])],
                  foreground=[("disabled", p["accent_text"]), ("!disabled", p["accent_text"])])
        style.configure("Danger.TButton", background=p["danger_bg"], foreground=p["danger_text"],
                        bordercolor=p["danger_border"], padding=(10, 6), relief="flat")
        style.map("Danger.TButton", background=[("disabled", p["danger_disabled"]), ("pressed", p["danger_pressed"]), ("active", p["danger_active"])])
        style.configure("TNotebook", background=p["app"], bordercolor=p["border"])
        style.configure("TNotebook.Tab", background=p["button"], foreground=p["text"], padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", p["panel"]), ("active", p["button_active"])], foreground=[("selected", p["accent"])])
        style.configure("TProgressbar", troughcolor=p["progress_track"], background=p["accent"], bordercolor=p["progress_track"])
        style.configure("TScrollbar", background=p["scrollbar"], troughcolor=p["app"], bordercolor=p["app"], arrowsize=12, arrowcolor=p["text"])

    def _apply_theme(self, theme_name: str, persist: bool = True):
        selected = theme_name if theme_name in self.THEME_PALETTES else "dark"
        self.theme_name = selected
        self.palette = dict(self.THEME_PALETTES[selected])
        self.configure(bg=self.palette["app"])
        self._configure_theme_styles()

        def recolor(widget):
            try:
                if isinstance(widget, tk.Text):
                    widget.configure(
                        bg=self.palette["white"], fg=self.palette["text"],
                        insertbackground=self.palette["text"], selectbackground=self.palette["selection"],
                        selectforeground=self.palette["text"], highlightbackground=self.palette["border"],
                        highlightcolor=self.palette["accent"],
                    )
                elif isinstance(widget, tk.Canvas):
                    widget.configure(background=self.palette["app"], highlightbackground=self.palette["app"])
                elif isinstance(widget, (tk.Frame, tk.LabelFrame)):
                    widget.configure(bg=self.palette["panel"])
                for child in widget.winfo_children():
                    recolor(child)
            except (tk.TclError, AttributeError):
                pass

        recolor(self)
        if hasattr(self, "theme_button"):
            self.theme_button.configure(text="☀️ Дневная тема" if selected == "dark" else "🌙 Ночная тема")
        if persist:
            self._save_config_now()

    def _toggle_theme(self):
        self._apply_theme("light" if self.theme_name == "dark" else "dark", persist=True)

    def _start_caption_only(self):
        """Запускает отдельный режим подписей одним нажатием, не меняя сохранённый режим фильтрации."""
        self._start(caption_only_override=True)

    def _start(self, caption_only_override: bool = False):
        if self.worker and self.worker.is_alive():
            return
        if self.export_worker and self.export_worker.is_alive():
            messagebox.showwarning(APP_TITLE, "Дождитесь завершения экспорта набора.")
            return
        if self.review_workers:
            messagebox.showwarning(APP_TITLE, "Сначала дождитесь завершения генерации подписи в карточке изображения.")
            return
        try:
            settings = self._get_settings()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        if caption_only_override:
            settings.caption_only = True
            settings.generate_captions = True
        # Переключатель clean_outputs сохранён только для совместимости со старой конфигурацией.
        # В режиме SQLite ни одна папка результата автоматически не удаляется.
        if not settings.caption_only and settings.check_faces and not Path(settings.face_model_path).expanduser().is_file():
            if not messagebox.askyesno(APP_TITLE, "Файл модели MediaPipe не найден. Приложение попробует скачать официальный файл при старте.\n\nПродолжить?", icon="warning"):
                self.notebook.select(self.tab_models)
                return
        # Проверяем необязательные BLIP/CLIP-зависимости до запуска рабочего потока.
        # Отсутствие transformers не должно завершать всю задачу критической ошибкой.
        ai_features = []
        if settings.classify_clip:
            ai_features.append("CLIP (категоризация)")
        if settings.generate_captions:
            ai_features.append("BLIP (генерация подписей)")
        if ai_features:
            try:
                transformers_available = importlib.util.find_spec("transformers") is not None
            except (ImportError, ValueError, ModuleNotFoundError):
                transformers_available = False
            if not transformers_available:
                install_command = (
                    f'"{sys.executable}" -m pip install '
                    '"transformers>=4.45,<5.0" "accelerate>=0.30"'
                )
                if settings.caption_only:
                    messagebox.showerror(
                        APP_TITLE,
                        "Для режима «ТОЛЬКО ПОДПИСИ» необходим пакет transformers, иначе BLIP не сможет создать описания.\n\n"
                        "Установите зависимость в том же Python, которым запускается приложение:\n"
                        + install_command
                        + "\n\nЛибо запустите install_optional_ai.bat и повторите операцию."
                    )
                    return
                message = (
                    "Не найден пакет Python «transformers», необходимый для включённых функций:\n\n"
                    + "\n".join(f"• {feature}" for feature in ai_features)
                    + "\n\nУстановите зависимость в том же Python, которым запускается приложение:\n"
                    + install_command
                    + "\n\nВ комплект также входит install_optional_ai.bat. После установки перезапустите приложение.\n\n"
                    "Продолжить обработку БЕЗ перечисленных AI-функций?\n"
                    "«Да» — временно отключить их и продолжить обычную фильтрацию.\n"
                    "«Нет» — отменить запуск и установить зависимость."
                )
                if not messagebox.askyesno(APP_TITLE, message, icon="warning"):
                    return
                if settings.classify_clip:
                    self.v["classify_clip"].set(False)
                    settings.classify_clip = False
                if settings.generate_captions:
                    self.v["generate_captions"].set(False)
                    settings.generate_captions = False
                self._save_current_config()

        self._save_current_config()
        self.settings_snapshot = settings
        self.stop_event.clear()
        self.progress.configure(value=0, maximum=100)
        self.v_progress.set("Обработано: 0 / …")
        self.v_status.set("Подготовка моделей…")
        self.start_button.configure(state="disabled")
        self.caption_only_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.notebook.select(self.tab_log)
        self._log("=" * 68)
        self._log("Начало обработки")
        self.worker = threading.Thread(target=self._worker_main, args=(settings,), daemon=True, name="DatasetCuratorWorker")
        self.worker.start()

    def _worker_main(self, settings: Settings):
        try:
            DatasetProcessor(settings, self.stop_event, self.event_queue.put).run()
        except Exception as exc:
            self.event_queue.put(("fatal", f"{type(exc).__name__}: {exc}", traceback.format_exc()))

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.v_status.set("Запрошена остановка… текущий файл будет завершён")
            self.stop_button.configure(state="disabled")

    def _mark_review_tabs_dirty_and_refresh_active(self):
        """Помечает оба списка устаревшими, но обновляет только видимую вкладку."""
        self.review_dirty["selected"] = True
        self.review_dirty["rejected"] = True
        try:
            active = self.notebook.select()
            if active == str(self.tab_selected):
                self.after(25, lambda: self._refresh_review_tab("selected", reset=True, sync_existing=False))
            elif active == str(self.tab_rejected):
                self.after(25, lambda: self._refresh_review_tab("rejected", reset=True, sync_existing=False))
        except tk.TclError:
            pass

    def _apply_progress_event(self, event: tuple):
        _, current, total, filename, stats = event
        self.progress.configure(maximum=max(total, 1), value=current)
        self.v_progress.set(f"Обработано: {current} / {total} · {filename}")
        self.v_summary.set(
            f"Подписей: {stats['captions']} · пропущено: {stats['captions_skipped']} · ошибки: {stats['errors']}"
            if self.settings_snapshot and self.settings_snapshot.caption_only
            else f"selected {stats['selected']} · пограничные {stats['maybe']} · отклонено {stats['rejected']} · ошибки {stats['errors']}"
        )
        self.v_status.set("Обработка…")

    def _handle_export_result(self, event: tuple):
        _, base_text, output_text, exported, skipped, error, empty, added = event
        self.export_worker = None
        self.export_button.configure(state="normal")
        if added:
            self._log(f"SQLite: добавлено записей об изображениях — {added}")
        if error:
            self.v_status.set("Ошибка экспорта")
            self._log(f"ОШИБКА экспорта: {error}")
            messagebox.showerror(APP_TITLE, f"Не удалось экспортировать набор:\n{error}")
            return
        if empty:
            self.v_status.set("Экспорт: нет выбранных изображений")
            messagebox.showinfo(APP_TITLE, "В SQLite пока нет выбранных изображений для экспорта.")
            return
        message = (
            f"Экспорт завершён.\n\n"
            f"Изображений и EN .txt экспортировано: {exported}\n"
            f"Пропущено (файл отсутствует или нет английского описания): {skipped}\n"
            f"Папка: {output_text}\n"
        )
        target = Path(output_text)
        if (target / "captions_ru.csv").is_file():
            message += "Переводы RU: captions_ru.csv"
        self._log(message.replace("\n", " | "))
        self.v_status.set(f"Экспорт завершён: {exported} изображений")
        messagebox.showinfo(APP_TITLE, message)

    def _poll_queue(self):
        """Обрабатывает очередь порциями; прогресс объединяется, журнал записывается пачкой."""
        batch = []
        deadline = time.perf_counter() + 0.008
        while len(batch) < 400 and time.perf_counter() < deadline:
            try:
                batch.append(self.event_queue.get_nowait())
            except queue.Empty:
                break

        pending_logs: list[str] = []
        latest_progress = None

        def flush_pending():
            nonlocal latest_progress
            if pending_logs:
                self._append_log_lines(pending_logs)
                pending_logs.clear()
            if latest_progress is not None:
                self._apply_progress_event(latest_progress)
                latest_progress = None

        for event in batch:
            kind = event[0]
            if kind == "log":
                pending_logs.append(str(event[1]))
                continue
            if kind == "progress":
                # Десятки промежуточных значений заменяются последним в текущей порции.
                latest_progress = event
                continue

            # Сохраняем правильный порядок относительно конечных сообщений и диалогов.
            flush_pending()
            if kind == "review_catalog_result":
                self._handle_review_catalog_result(event)
            elif kind == "review_translation_result":
                _, group, path_text, translated, error = event
                key = (group, path_text)
                self.review_workers.pop(key, None)
                card = self.review_card_widgets.get(key)
                if card:
                    for widget_name in ("generate", "translate", "save", "move", "caption", "caption_ru"):
                        widget = card.get(widget_name)
                        if widget is not None:
                            widget.configure(state="normal")
                if error:
                    self.v_status.set(f"Ошибка перевода: {Path(path_text).name}")
                    self._log(f"Перевод EN → RU не выполнен для {path_text}: {error}")
                    messagebox.showerror(APP_TITLE, f"Не удалось перевести описание:\n{path_text}\n\n{error}")
                else:
                    self.review_caption_ru_drafts[path_text] = translated
                    if card:
                        card["caption_ru"].delete("1.0", "end")
                        card["caption_ru"].insert("1.0", translated)
                        card["caption_state"].configure(text="Перевод подготовлен — не сохранён")
                    self.v_status.set(f"Перевод подготовлен: {Path(path_text).name}. Нажми «Сохранить оба описания».")
                    self._log(f"Перевод подготовлен в редакторе: {path_text}")
            elif kind == "review_caption_result":
                _, group, path_text, caption, caption_ru, error = event
                key = (group, path_text)
                self.review_workers.pop(key, None)
                card = self.review_card_widgets.get(key)
                if not error:
                    self.review_caption_drafts[path_text] = caption
                    self.review_caption_ru_drafts[path_text] = caption_ru
                if card:
                    for widget_name in ("generate", "translate", "save", "move", "caption", "caption_ru"):
                        widget = card.get(widget_name)
                        if widget is not None:
                            widget.configure(state="normal")
                    if not error:
                        card["caption"].delete("1.0", "end")
                        card["caption"].insert("1.0", caption)
                        card["caption_ru"].delete("1.0", "end")
                        card["caption_ru"].insert("1.0", caption_ru)
                        card["caption_state"].configure(text="Новые описания не сохранены")
                if error:
                    self.v_status.set(f"Ошибка генерации: {Path(path_text).name}")
                    self._log(f"Ошибка генерации описания для {path_text}: {error}")
                    messagebox.showerror(APP_TITLE, f"Не удалось сгенерировать описание для:\n{path_text}\n\n{error}")
                else:
                    self.v_status.set(f"Описания EN/RU подготовлены: {Path(path_text).name}. Нажми «Сохранить оба описания».")
                    self._log(f"Описания подготовлены в редакторе: {path_text}")
            elif kind == "export_result":
                self._handle_export_result(event)
            elif kind == "done":
                _, stats, report_path, stopped = event
                self.last_stats = stats
                self.start_button.configure(state="normal")
                self.caption_only_button.configure(state="normal")
                self.export_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.v_status.set("Остановлено пользователем" if stopped else "Готово")
                self.v_summary.set(
                    f"Подписей: {stats['captions']} · пропущено: {stats['captions_skipped']} · ошибки: {stats['errors']}"
                    if self.settings_snapshot and self.settings_snapshot.caption_only
                    else f"selected {stats['selected']} · пограничные {stats['maybe']} · отклонено {stats['rejected']} · ошибки {stats['errors']}"
                )
                summary = (
                    f"Обработано: {stats['processed']}\n"
                    f"selected: {stats['selected']}\n"
                    f"Пограничные: {stats['maybe']}\n"
                    f"Отклонено по эстетике: {stats['rejected']}\n"
                    f"Малое разрешение: {stats['small_resolution']}\n"
                    f"Нет подходящего лица: {stats['no_face']}\n"
                    f"Несколько лиц: {stats['multiple_faces']}\n"
                    f"Размытые: {stats['blurry']}\n"
                    f"Ошибки: {stats['errors']}\n"
                    f"Создано подписей: {stats['captions']}\n"
                    f"Пропущено (подпись уже существует): {stats['captions_skipped']}\n"
                    f"Кандидатов в дубликаты: {stats['duplicates']}\n\n"
                    f"База SQLite: {report_path or 'путь не определён'}"
                )
                self._log("ИТОГ\n" + summary.replace("\n", " | "))
                messagebox.showinfo(APP_TITLE, summary)
                self._mark_review_tabs_dirty_and_refresh_active()
            elif kind == "fatal":
                _, message, details = event
                self._log("КРИТИЧЕСКАЯ ОШИБКА: " + message)
                self._log(details)
                self.start_button.configure(state="normal")
                self.caption_only_button.configure(state="normal")
                self.export_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.v_status.set("Ошибка запуска")
                messagebox.showerror(APP_TITLE, message + "\n\nПодробности записаны в журнал.")
                self._mark_review_tabs_dirty_and_refresh_active()

        flush_pending()
        # Частый короткий опрос не даёт очереди логов блокировать поток Tkinter.
        self.after(50, self._poll_queue)

    def _append_log_lines(self, lines: list[str]):
        if not hasattr(self, "log_text") or not lines:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "\n".join(lines) + "\n")
        try:
            count = int(self.log_text.index("end-1c").split(".")[0])
            if count > 7000:
                self.log_text.delete("1.0", f"{count - 6000}.0")
        except (tk.TclError, ValueError):
            pass
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log(self, line: str):
        self._append_log_lines([line])

    def _on_close(self):
        if self.export_worker and self.export_worker.is_alive():
            messagebox.showwarning(APP_TITLE, "Экспорт ещё выполняется. Дождитесь его завершения перед закрытием приложения.")
            return
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(APP_TITLE, "Обработка ещё выполняется. Запросить остановку и закрыть окно?", icon="warning"):
                return
            self.stop_event.set()
            # Worker daemon, поэтому окно не будет удерживать завершение процесса.
        if self._config_save_after_id is not None:
            try:
                self.after_cancel(self._config_save_after_id)
            except tk.TclError:
                pass
            self._config_save_after_id = None
        self._save_config_now()
        self.destroy()


def main():
    app = DatasetCuratorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
