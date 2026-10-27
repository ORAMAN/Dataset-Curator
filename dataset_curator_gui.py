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

from dataset_catalog import CatalogStore, translate_en_ru_mymemory

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


def safe_copy(src: Path, destination_dir: Path, output_name: str | None = None) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / (output_name or src.name)
    # Если целевой файл есть, обновляем его: это делает повторный запуск идемпотентным.
    shutil.copy2(src, target)
    return target


def safe_move(src: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / src.name
    if target.exists():
        stem, suffix = src.stem, src.suffix
        counter = 2
        while True:
            candidate = destination_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            counter += 1
    shutil.move(str(src), str(target))
    return target


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


def deduplicate_selected(settings: Settings, log: Callable[[str], None], stop_event: threading.Event) -> tuple[int, dict[str, str]]:
    """Группирует near-duplicates транзитивно; предпочтение — разрешению, затем размеру файла."""
    import imagehash
    from PIL import Image, ImageOps

    selected_dir = Path(settings.input_dir) / "selected"
    duplicate_dir = Path(settings.input_dir) / "duplicate"
    if not selected_dir.is_dir():
        log("Папки selected нет — поиск дубликатов пропущен.")
        return 0, {}
    paths = sorted(p for p in selected_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        log("Для поиска дубликатов недостаточно изображений.")
        return 0, {}

    hash_func = getattr(imagehash, settings.duplicate_hash)
    hashes: list[int] = []
    quality: list[tuple[int, int]] = []  # пиксели, байты
    valid_paths: list[Path] = []
    tree = BKTree()
    uf = UnionFind(len(paths))
    log(f"Ищу near-duplicates в selected: {len(paths)} файлов, {settings.duplicate_hash}, порог {settings.duplicate_distance}…")

    for original_index, path in enumerate(paths, 1):
        if stop_event.is_set():
            log("Поиск дубликатов остановлен пользователем.")
            return 0, {}
        try:
            with Image.open(path) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
                w, h = img.size
                hash_value = int(str(hash_func(img)), 16)
            # Индексы соответствуют valid_paths, а не исходному списку.
            idx = len(valid_paths)
            for other in tree.query(hash_value, settings.duplicate_distance):
                uf.union(idx, other)
            tree.add(hash_value, idx)
            valid_paths.append(path)
            hashes.append(hash_value)
            quality.append((w * h, path.stat().st_size))
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
    moved_map: dict[str, str] = {}
    for group in duplicate_groups:
        # Детерминированно сохраняем файл с наибольшим количеством пикселей, затем — размером.
        ordered = sorted(group, key=lambda i: (quality[i][0], quality[i][1]), reverse=True)
        keeper = valid_paths[ordered[0]]
        log(f"Оставляю: {keeper.name} ({quality[ordered[0]][0]:,} пикс.)")
        for idx in ordered[1:]:
            duplicate = valid_paths[idx]
            duplicate_count += 1
            log(f"  Похожий: {duplicate.name}" + (" [тестовый режим]" if settings.duplicate_dry_run else ""))
            if not settings.duplicate_dry_run:
                try:
                    old_path = str(duplicate)
                    moved_path = safe_move(duplicate, duplicate_dir)
                    moved_map[old_path] = str(moved_path)
                    # Перемещаем и подпись BLIP, если она существует рядом с исходным изображением.
                    sidecar = duplicate.with_suffix(".txt")
                    if sidecar.is_file():
                        safe_move(sidecar, duplicate_dir)
                    log(f"  Перемещён в: {moved_path}")
                except Exception as exc:
                    log(f"  Ошибка перемещения {duplicate.name}: {exc}")
    if settings.duplicate_dry_run:
        log(f"Тестовый режим: найдено кандидатов на перемещение — {duplicate_count}. Файлы не изменены.")
    else:
        log(f"Перемещение завершено. Обработано кандидатов: {duplicate_count}.")
    return duplicate_count, moved_map


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
            "filename": source.name, "output_file": "", "status": "error", "reason": "",
            "width": "", "height": "", "aesthetic_score": "", "suitable_faces": "",
            "max_face_height_percent": "", "blur_variance": "", "clip_category": "",
            "clip_relative_score": "", "caption": "", "description_ru": "", "error": "",
            "output_file": str(source),
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
                    self.log(f"  → разрешение {width}×{height} ниже порога → bed_size")
                    return

            if self.settings.check_faces:
                count, max_rel = self.services.detect_suitable_faces(image)
                record["suitable_faces"] = count
                record["max_face_height_percent"] = round(max_rel * 100, 2)
                if count < 1:
                    dest = self._route("no_or_small_faces", source)
                    record.update(status="no_face", reason="Нет достаточно крупного лица", output_file=str(dest))
                    self.stats["no_face"] += 1
                    self.log(f"  → нет подходящего лица (макс. высота {max_rel * 100:.1f}%) → no_or_small_faces")
                    return
                if self.settings.reject_multiple_faces and count > 1:
                    dest = self._route("moree_faces", source)
                    record.update(status="multiple_faces", reason=f"Подходящих лиц: {count}", output_file=str(dest))
                    self.stats["multiple_faces"] += 1
                    self.log(f"  → подходящих лиц: {count} → moree_faces")
                    return

            if self.settings.check_blur:
                variance = get_blur_variance(image)
                record["blur_variance"] = round(variance, 3)
                if variance < self.settings.blur_threshold:
                    dest = self._route("blure", source)
                    record.update(status="blurry", reason=f"Размыто: Laplacian variance {variance:.2f}", output_file=str(dest))
                    self.stats["blurry"] += 1
                    self.log(f"  → размыто ({variance:.1f} < {self.settings.blur_threshold:g}) → blure")
                    return

            score = None
            if self.settings.check_aesthetic:
                score = float(self.services.load_aesthetic()(image).item())
                record["aesthetic_score"] = round(score, 4)

            if score is not None and score < self.settings.aesthetic_maybe:
                dest = self._route("rejected", source)
                record.update(status="rejected", reason=f"Низкая эстетическая оценка {score:.2f}", output_file=str(dest))
                self.stats["rejected"] += 1
                self.log(f"  → эстетика {score:.2f} → rejected")
                return

            if score is not None and score < self.settings.aesthetic_keep:
                out_name = f"!_{score:.1f}_{source.name}"
                dest = self._route("selected", source, out_name)
                status = "maybe"
                self.stats["maybe"] += 1
                self.log(f"  → эстетика {score:.2f} → пограничное selected/{out_name}")
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
                            russian = translate_en_ru_mymemory(caption, self.settings.translation_email)
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
                    record["description_ru"] = translate_en_ru_mymemory(caption, self.settings.translation_email)
                    self.log(f"  → перевод RU: {record['description_ru']}")
                except Exception as translate_exc:
                    self.log(f"  ! перевод MyMemory не выполнен: {type(translate_exc).__name__}: {translate_exc}")
            self.stats["captions"] += 1
            self.log(f"  → описание EN подготовлено для SQLite: {caption}")
        except Exception as exc:
            self.stats["errors"] += 1
            record.update(status="caption_error", reason="Ошибка генерации подписи", error=f"{type(exc).__name__}: {exc}")
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

        # Скрипт намеренно обрабатывает только изображения непосредственно в исходной папке:
        # папки selected/rejected/bed_size не попадут в повторный обход.
        paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise FileNotFoundError("В выбранной папке не найдено изображений поддерживаемых форматов.")
        self.log(f"Найдено изображений в корне папки: {len(paths)}")
        if self.settings.caption_only:
            self.log("РЕЖИМ «ТОЛЬКО ПОДПИСИ»: фильтры, оценка эстетики, поиск лиц и дубликатов пропускаются.")
            self.log("Файлы изображений не копируются и не перемещаются; подписи записываются рядом с исходниками.")
            if self.settings.skip_existing_captions:
                self.log("Существующие .txt-файлы будут пропущены.")
        else:
            self.log("Исходные изображения не удаляются; результаты копируются в подпапки.")

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

            if self.settings.clean_outputs and not self.settings.caption_only:
                output_names = ("selected", "rejected", "bed_size", "blure", "no_or_small_faces", "moree_faces", "duplicate")
                for dirname in output_names:
                    output_dir = input_dir / dirname
                    if output_dir.is_dir() and not output_dir.is_symlink():
                        shutil.rmtree(output_dir)
                        self.log(f"Очищена предыдущая папка результата: {dirname}")
                old_report = input_dir / "dataset_report.csv"
                if old_report.is_file():
                    old_report.unlink()

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
                    duplicate_count, moved_map = deduplicate_selected(self.settings, self.log, self.stop_event)
                    self.stats["duplicates"] = duplicate_count
                    if moved_map:
                        for row in self.report_rows:
                            moved_to = moved_map.get(str(row.get("output_file", "")))
                            if moved_to:
                                row["status"] = "duplicate"
                                row["reason"] = "Near-duplicate; перемещён после сравнения хешей"
                                row["output_file"] = moved_to
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
            self.log(f"Отчёт: {report}" if report else "CSV-отчёт не сохранён.")
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
        self.review_visible_count: dict[str, int] = {"selected": 0, "rejected": 0}
        self.review_card_widgets: dict[tuple[str, str], dict[str, Any]] = {}
        self.review_caption_drafts: dict[str, str] = {}
        self.review_photo_refs: dict[str, list[Any]] = {"selected": [], "rejected": []}
        self.review_workers: dict[tuple[str, str], threading.Thread] = {}
        self.review_count_vars: dict[str, tk.StringVar] = {}
        self.review_canvases: dict[str, tk.Canvas] = {}
        self.review_list_frames: dict[str, ttk.Frame] = {}
        self.review_load_more_buttons: dict[str, ttk.Button] = {}
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
        ttk.Button(folder_frame, text="Открыть selected", command=lambda: self._open_output("selected")).pack(side="left", padx=(8, 0))

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
        self.stop_button = ttk.Button(action_bar, text="■ Остановить", command=self._stop, state="disabled", style="Danger.TButton")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Открыть CSV-отчёт", command=self._open_report).pack(side="left", padx=(8, 0))
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
        ttk.Label(aesthetic, text="Отправлять в selected от:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(aesthetic, textvariable=self.v["aesthetic_keep"], width=8).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(aesthetic, text="Пограничная зона от:").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(aesthetic, textvariable=self.v["aesthetic_maybe"], width=8).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(aesthetic, text="Ниже пограничной зоны → rejected.\nПограничные файлы получают префикс !_оценка_.", style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        duplicates = ttk.LabelFrame(right, text="Похожие изображения", padding=10)
        duplicates.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(duplicates, text="После фильтрации искать дубликаты в selected", variable=self.v["find_duplicates"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(duplicates, text="Метод хеша:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Combobox(duplicates, textvariable=self.v["duplicate_hash"], values=["phash", "dhash", "whash", "ahash"], state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Label(duplicates, text="Расстояние Хэмминга:").grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(duplicates, textvariable=self.v["duplicate_distance"], width=8).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(7, 0))
        ttk.Checkbutton(duplicates, text="Только показать, не перемещать (безопасно)", variable=self.v["duplicate_dry_run"]).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(duplicates, text="Порог 8 — умеренно строгий для 64-битного хеша.\nРеальное перемещение включайте после проверки тестового режима.", style="Muted.TLabel", wraplength=390).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        warning = ttk.LabelFrame(right, text="Безопасность", padding=10)
        warning.pack(fill="x")
        ttk.Label(warning, text="Исходные файлы не удаляются: они копируются в подпапки.\nСканируются только файлы в корне выбранной папки; ранее созданные подпапки не обрабатываются повторно.", wraplength=410).pack(anchor="w")

    def _on_caption_only_toggle(self):
        # Отдельный режим сам включает генерацию подписей при построении Settings,
        # не меняя настройку создания подписей в обычном режиме фильтрации.
        self._schedule_save_config()

    def _build_models_tab(self):
        tab = self.tab_models
        captions = ttk.LabelFrame(tab, text="Автоматические подписи для датасета", padding=10)
        captions.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(captions, text="Генерировать .txt-подписи для изображений в selected при обычной фильтрации", variable=self.v["generate_captions"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            captions,
            text="ТОЛЬКО ПОДПИСИ — без фильтрации, оценки эстетики, детектора лиц и поиска дубликатов",
            variable=self.v["caption_only"],
            command=self._on_caption_only_toggle,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        ttk.Checkbutton(
            captions,
            text="Пропускать изображения, у которых уже есть одноимённый .txt (не перезаписывать ручные подписи)",
            variable=self.v["skip_existing_captions"],
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(captions, text="Модель:").grid(row=3, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(captions, textvariable=self.v["caption_model"]).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))
        ttk.Label(captions, text="Префикс / trigger token:").grid(row=4, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(captions, textvariable=self.v["caption_prefix"]).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))
        ttk.Label(
            captions,
            text="В режиме «ТОЛЬКО ПОДПИСИ» файлы .txt создаются рядом с исходными изображениями. Уже существующие подписи сохраняются по умолчанию. Модель Salesforce/blip-image-captioning-base загружается с Hugging Face при первом запуске; подписи обычно на английском.",
            style="Muted.TLabel", wraplength=850,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
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
        ttk.Checkbutton(clip, text="Добавлять выбранную CLIP-категорию в конец .txt-подписи BLIP", variable=self.v["append_clip_label_to_caption"]).pack(anchor="w", pady=(7, 0))
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
        """Прокручиваемый просмотр карточек изображений с редактированием sidecar-подписей."""
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, 6))

        ttk.Label(toolbar, textvariable=self._make_review_count_var(group), style="Header.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Обновить список", command=lambda g=group: self._refresh_review_tab(g)).pack(side="right")
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
        folder = base / ("selected" if group == "selected" else "rejected")
        if folder.is_dir():
            self._open_path(folder)
        else:
            messagebox.showinfo(APP_TITLE, f"Папка пока не создана:\n{folder}")

    def _on_review_tab_changed(self, _event=None):
        try:
            selected_tab = self.notebook.select()
            if selected_tab == str(self.tab_selected):
                self._refresh_review_tab("selected")
            elif selected_tab == str(self.tab_rejected):
                self._refresh_review_tab("rejected")
        except tk.TclError:
            pass

    def _review_files(self, group: str) -> list[Path]:
        base = self._review_input_dir()
        if base is None or not base.is_dir():
            return []
        if group == "selected":
            folders = [base / "selected"]
        else:
            # Собираем все типы брака, а не только низкий aesthetic score.
            folders = [
                base / "rejected",
                base / "bed_size",
                base / "blure",
                base / "no_or_small_faces",
                base / "moree_faces",
                base / "duplicate",
            ]
        files: list[Path] = []
        for folder in folders:
            if not folder.is_dir() or folder.is_symlink():
                continue
            try:
                files.extend(
                    path for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            except OSError as exc:
                self._log(f"Не удалось прочитать папку {folder}: {exc}")
        # Файлы с одинаковым именем из разных папок могут быть отдельными изображениями.
        return sorted(files, key=lambda path: (path.parent.name.casefold(), path.name.casefold(), str(path).casefold()))

    def _refresh_all_review_tabs(self):
        self._refresh_review_tab("selected", reset=True)
        self._refresh_review_tab("rejected", reset=True)

    def _refresh_review_tab(self, group: str, reset: bool = True):
        if group not in self.review_list_frames:
            return
        if reset:
            # При обычном обновлении повторно считываем каталоги и Sidecar подписи.
            self.review_file_cache[group] = self._review_files(group)
            self.review_visible_count[group] = 0
            for key in [key for key in self.review_card_widgets if key[0] == group]:
                self.review_card_widgets.pop(key, None)
            self.review_photo_refs[group] = []
            list_frame = self.review_list_frames[group]
            for child in list_frame.winfo_children():
                child.destroy()

        files = self.review_file_cache.get(group, [])
        list_frame = self.review_list_frames[group]
        old_count = self.review_visible_count.get(group, 0)
        end = min(len(files), old_count + self.review_page_size)
        if not files and old_count == 0:
            base = self._review_input_dir()
            if base is None or not base.is_dir():
                text = "Выберите существующую исходную папку выше."
            elif group == "selected":
                text = "Папка selected пока не содержит обработанных изображений."
            else:
                text = "Пока нет отклонённых изображений."
            ttk.Label(list_frame, text=text, style="Muted.TLabel", padding=20).pack(anchor="w")
        else:
            for index in range(old_count, end):
                path = files[index]
                self._build_review_card(list_frame, group, path)
        self.review_visible_count[group] = end

        if group in self.review_count_vars:
            self.review_count_vars[group].set(
                f"{'Выбранные' if group == 'selected' else 'Отбракованные'} — {len(files)} файлов "
                f"(показано {end})"
            )
        button = self.review_load_more_buttons.get(group)
        if button is not None:
            button.configure(state="normal" if end < len(files) else "disabled")
            if end < len(files):
                button.configure(text=f"Показать ещё ({min(self.review_page_size, len(files) - end)})")
            else:
                button.configure(text="Все изображения показаны" if files else "Нет изображений")
        canvas = self.review_canvases.get(group)
        if canvas is not None:
            canvas.configure(scrollregion=canvas.bbox("all"))

    def _build_review_card(self, list_frame: ttk.Frame, group: str, source: Path):
        card = ttk.LabelFrame(list_frame, text=f"{source.name}  ·  {source.parent.name}", padding=8)
        card.pack(fill="x", padx=5, pady=5)

        preview_frame = ttk.Frame(card, width=156, height=160)
        preview_frame.pack(side="left", fill="y", padx=(0, 10))
        preview_frame.pack_propagate(False)
        try:
            from PIL import Image, ImageOps, ImageTk
            with Image.open(source) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                image.thumbnail((144, 144))
                thumbnail = ImageTk.PhotoImage(image.copy())
            self.review_photo_refs[group].append(thumbnail)
            image_label = ttk.Label(preview_frame, image=thumbnail, anchor="center")
            image_label.image = thumbnail
            image_label.pack(fill="both", expand=True)
        except Exception as exc:
            ttk.Label(preview_frame, text=f"Не удалось\nзагрузить фото\n{exc}", wraplength=145).pack(fill="both", expand=True)

        content = ttk.Frame(card)
        content.pack(side="left", fill="both", expand=True)
        caption_path = source.with_suffix(".txt")
        try:
            if str(source) in self.review_caption_drafts:
                caption_value = self.review_caption_drafts[str(source)]
            else:
                caption_value = caption_path.read_text(encoding="utf-8-sig") if caption_path.is_file() else ""
        except OSError as exc:
            caption_value = ""
            self._log(f"Не удалось прочитать подпись {caption_path}: {exc}")

        caption_header = ttk.Frame(content)
        caption_header.pack(fill="x")
        ttk.Label(caption_header, text="Подпись (.txt)", style="Muted.TLabel").pack(side="left")
        caption_state = ttk.Label(
            caption_header,
            text="Есть подпись" if caption_path.is_file() else "Подпись отсутствует",
            style="Muted.TLabel",
        )
        caption_state.pack(side="right")

        caption_text = tk.Text(content, height=4, wrap="word", undo=True, font=("Segoe UI", 9))
        caption_text.pack(fill="x", expand=True, pady=(3, 6))
        if caption_value:
            caption_text.insert("1.0", caption_value.rstrip("\r\n"))
        def remember_caption_draft(_event=None, p=str(source), t=caption_text):
            # after_idle учитывает вставку/вырезание, выполненные стандартными биндингами Tk.
            try:
                self.after_idle(lambda: self.review_caption_drafts.__setitem__(p, t.get("1.0", "end-1c")))
            except tk.TclError:
                pass

        for sequence in ("<KeyRelease>", "<<Paste>>", "<<Cut>>", "<<Undo>>", "<<Redo>>"):
            caption_text.bind(sequence, remember_caption_draft, add="+")

        buttons = ttk.Frame(content)
        buttons.pack(fill="x")
        save_button = ttk.Button(
            buttons,
            text="Сохранить подпись",
            command=lambda p=source, t=caption_text: self._save_review_caption(p, t),
        )
        save_button.pack(side="left")
        generate_button = ttk.Button(
            buttons,
            text="Сгенерировать",
            command=lambda p=source, t=caption_text, g=group: self._generate_review_caption(p, t, g),
        )
        generate_button.pack(side="left", padx=(6, 0))
        move_text = "Забраковать → rejected" if group == "selected" else "Вернуть в selected"
        move_style = "Danger.TButton" if group == "selected" else "Accent.TButton"
        move_button = ttk.Button(
            buttons,
            text=move_text,
            style=move_style,
            command=lambda p=source, g=group, t=caption_text: self._move_review_image(p, g, t),
        )
        move_button.pack(side="left", padx=(6, 0))
        ttk.Button(
            buttons,
            text="Открыть фото",
            command=lambda p=source: self._open_path(p),
        ).pack(side="right")

        self.review_card_widgets[(group, str(source))] = {
            "caption": caption_text,
            "generate": generate_button,
            "save": save_button,
            "move": move_button,
            "caption_state": caption_state,
            "group": group,
            "path": source,
        }

    def _review_is_busy(self) -> bool:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(APP_TITLE, "Основная обработка ещё выполняется. Дождитесь её окончания.")
            return True
        if any(worker.is_alive() for worker in self.review_workers.values()):
            messagebox.showwarning(APP_TITLE, "Дождитесь завершения текущей генерации подписи.")
            return True
        return False

    def _write_caption_for_image(self, source: Path, value: str) -> Path:
        sidecar = source.with_suffix(".txt")
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
            temporary.replace(sidecar)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return sidecar

    def _review_group_for_path(self, source: Path) -> str:
        for group, _path in self.review_card_widgets:
            if _path == str(source):
                return group
        return "selected"

    def _save_review_caption(self, source: Path, caption_widget: tk.Text, show_message: bool = True) -> bool:
        if self._review_is_busy():
            return False
        value = caption_widget.get("1.0", "end-1c").strip()
        sidecar = source.with_suffix(".txt")
        if not value:
            if sidecar.is_file():
                if not messagebox.askyesno(APP_TITLE, f"Подпись пуста. Удалить существующий файл?\n{sidecar}", icon="warning"):
                    return False
                try:
                    sidecar.unlink()
                    self.review_caption_drafts.pop(str(source), None)
                    card = self.review_card_widgets.get((self._review_group_for_path(source), str(source)))
                    if card:
                        card["caption_state"].configure(text="Подпись отсутствует")
                    if show_message:
                        self.v_status.set(f"Подпись удалена: {sidecar.name}")
                    return True
                except OSError as exc:
                    messagebox.showerror(APP_TITLE, f"Не удалось удалить подпись:\n{exc}")
                    return False
            if show_message:
                messagebox.showinfo(APP_TITLE, "Поле подписи пустое. Файл не создан.")
            return False
        try:
            self._write_caption_for_image(source, value)
            self.review_caption_drafts.pop(str(source), None)
            for card in self.review_card_widgets.values():
                if str(card.get("path")) == str(source):
                    card["caption_state"].configure(text="Есть подпись")
            self._log(f"Подпись сохранена: {sidecar}")
            if show_message:
                self.v_status.set(f"Подпись сохранена: {sidecar.name}")
            return True
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось сохранить подпись:\n{exc}")
            return False

    def _safe_move_review_image(self, source: Path, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        caption_source = source.with_suffix(".txt")
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            candidate = target_dir / f"{source.stem}{suffix}{source.suffix}"
            candidate_caption = candidate.with_suffix(".txt")
            if not candidate.exists() and not candidate_caption.exists():
                break
            index += 1

        shutil.move(str(source), str(candidate))
        try:
            if caption_source.is_file():
                shutil.move(str(caption_source), str(candidate_caption))
        except Exception:
            try:
                shutil.move(str(candidate), str(source))
            except Exception:
                pass
            raise
        return candidate

    def _move_review_image(self, source: Path, group: str, caption_widget: tk.Text):
        if self._review_is_busy():
            return
        if not source.is_file():
            messagebox.showwarning(APP_TITLE, f"Изображение уже перемещено или удалено:\n{source}")
            self._refresh_review_tab(group)
            return

        base = self._review_input_dir()
        if base is None or not base.is_dir():
            messagebox.showerror(APP_TITLE, "Выберите существующую исходную папку.")
            return
        # Защита от перемещения карточки, оставшейся на экране после смены исходной папки.
        try:
            if source.parent.parent.resolve() != base.resolve():
                messagebox.showwarning(
                    APP_TITLE,
                    "Карточка относится к другой исходной папке. Обновите список изображений и повторите действие.",
                )
                self._refresh_all_review_tabs()
                return
        except (OSError, RuntimeError):
            messagebox.showwarning(APP_TITLE, "Не удалось проверить исходную папку карточки. Список будет обновлён.")
            self._refresh_all_review_tabs()
            return

        # Сохраняем видимую редакцию перед переносом. Пустое поле не стирает
        # автоматически уже существующую подпись — для удаления есть отдельная кнопка Save.
        caption_value = caption_widget.get("1.0", "end-1c").strip()
        if caption_value:
            try:
                self._write_caption_for_image(source, caption_value)
            except Exception as exc:
                messagebox.showerror(APP_TITLE, f"Не удалось сохранить подпись перед перемещением:\n{exc}")
                return

        target_group = "rejected" if group == "selected" else "selected"
        target_dir = base / target_group
        try:
            target = self._safe_move_review_image(source, target_dir)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Не удалось переместить изображение:\n{exc}")
            return

        self.review_caption_drafts.pop(str(source), None)
        self._log(f"Изображение перемещено: {source} → {target}")
        self.v_status.set(f"Перемещено: {target.name}")
        self._refresh_review_tab(group, reset=True)
        self._refresh_review_tab(target_group, reset=True)

    def _generate_review_caption(self, source: Path, caption_widget: tk.Text, group: str):
        if self._review_is_busy():
            return
        if not source.is_file():
            messagebox.showwarning(APP_TITLE, f"Изображение не найдено:\n{source}")
            self._refresh_review_tab(group)
            return
        current_draft = caption_widget.get("1.0", "end-1c").strip()
        sidecar = source.with_suffix(".txt")
        try:
            saved_caption = sidecar.read_text(encoding="utf-8-sig").strip() if sidecar.is_file() else ""
        except OSError:
            saved_caption = ""
        if current_draft and current_draft != saved_caption:
            if not messagebox.askyesno(
                APP_TITLE,
                "В редакторе есть несохранённый текст. Заменить его новой AI-подписью?\n"
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
            card["generate"].configure(state="disabled")
            card["caption"].configure(state="disabled")
            card["save"].configure(state="disabled")
            card["move"].configure(state="disabled")
        self.v_status.set(f"Генерация подписи: {source.name}")
        self._log(f"Ручная генерация подписи: {source.name}")

        def work():
            services = ModelServices(settings, lambda line: self.event_queue.put(("log", f"[AI] {line}")))
            caption = ""
            error = ""
            try:
                from PIL import Image, ImageOps
                if settings.classify_clip:
                    try:
                        services.load_clip_model()
                    except Exception as exc:
                        self.event_queue.put(("log", f"CLIP недоступен; продолжаю с одной BLIP-подписью: {exc}"))
                services.load_caption_model()
                with Image.open(source) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                caption = services.caption(image)
                prefix = settings.caption_prefix.strip().strip(",;")
                if prefix and not caption.casefold().startswith(prefix.casefold()):
                    caption = f"{prefix}, {caption}".strip().strip(",")
                if settings.classify_clip:
                    try:
                        category, score = services.classify(image)
                        self.event_queue.put(("log", f"CLIP: {category} ({score * 100:.1f}% среди категорий)"))
                        if settings.append_clip_label_to_caption and category and category.casefold() not in caption.casefold():
                            caption = f"{caption.rstrip(' .,;')} , {category}".replace(" ,", ",")
                    except Exception as exc:
                        self.event_queue.put(("log", f"Ошибка классификации CLIP: {exc}"))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                services.close()
                self.event_queue.put(("review_caption_result", group, str(source), caption, error))

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
        report = Path(self.v["input_dir"].get().strip()) / "dataset_report.csv"
        if report.is_file():
            self._open_path(report)
        else:
            messagebox.showinfo(APP_TITLE, "CSV-отчёт появится после запуска обработки.")

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
        if any(worker.is_alive() for worker in self.review_workers.values()):
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
        if settings.clean_outputs and not settings.caption_only and not messagebox.askyesno(
            APP_TITLE,
            "Будут удалены только выходные папки этого приложения: selected, rejected, bed_size, blure, "
            "no_or_small_faces, moree_faces, duplicate, а также старый dataset_report.csv.\n"
            "Исходные изображения в корне папки останутся.\n\nПродолжить?",
            icon="warning",
        ):
            return
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

    def _poll_queue(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._log(event[1])
                elif kind == "review_caption_result":
                    _, group, path_text, caption, error = event
                    key = (group, path_text)
                    self.review_workers.pop(key, None)
                    card = self.review_card_widgets.get(key)
                    if not error:
                        # Сохраняем черновик в памяти, даже если карточка успела скрыться при смене вкладки.
                        self.review_caption_drafts[path_text] = caption
                    if card:
                        card["generate"].configure(state="normal")
                        card["save"].configure(state="normal")
                        card["move"].configure(state="normal")
                        card["caption"].configure(state="normal")
                        if not error:
                            card["caption"].delete("1.0", "end")
                            card["caption"].insert("1.0", caption)
                    if error:
                        self.v_status.set(f"Ошибка генерации: {Path(path_text).name}")
                        self._log(f"Ошибка ручной генерации подписи для {path_text}: {error}")
                        messagebox.showerror(APP_TITLE, f"Не удалось сгенерировать подпись для:\n{path_text}\n\n{error}")
                    else:
                        self.v_status.set(f"Подпись сгенерирована: {Path(path_text).name}. Нажми «Сохранить подпись».")
                        self._log(f"Подпись подготовлена в редакторе: {path_text}")
                elif kind == "progress":
                    _, current, total, filename, stats = event
                    self.progress.configure(maximum=max(total, 1), value=current)
                    self.v_progress.set(f"Обработано: {current} / {total} · {filename}")
                    self.v_summary.set(
                        f"Подписей: {stats['captions']} · пропущено: {stats['captions_skipped']} · ошибки: {stats['errors']}"
                        if self.settings_snapshot and self.settings_snapshot.caption_only
                        else f"selected {stats['selected']} · пограничные {stats['maybe']} · отклонено {stats['rejected']} · ошибки {stats['errors']}"
                    )
                    self.v_status.set("Обработка…")
                elif kind == "done":
                    _, stats, report_path, stopped = event
                    self.last_stats = stats
                    self.start_button.configure(state="normal")
                    self.caption_only_button.configure(state="normal")
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
                        f"CSV-отчёт: {report_path or 'не сохранён'}"
                    )
                    self._log("ИТОГ\n" + summary.replace("\n", " | "))
                    messagebox.showinfo(APP_TITLE, summary)
                    self.after(250, self._refresh_all_review_tabs)
                elif kind == "fatal":
                    _, message, details = event
                    self._log("КРИТИЧЕСКАЯ ОШИБКА: " + message)
                    self._log(details)
                    self.start_button.configure(state="normal")
                    self.caption_only_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.v_status.set("Ошибка запуска")
                    messagebox.showerror(APP_TITLE, message + "\n\nПодробности записаны в журнал.")
                    self.after(250, self._refresh_all_review_tabs)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _log(self, line: str):
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        # Журнал ограничен по размеру, чтобы Tk Text не разрастался на длинном прогоне.
        try:
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > 7000:
                self.log_text.delete("1.0", f"{lines - 6000}.0")
        except Exception:
            pass
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self):
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
