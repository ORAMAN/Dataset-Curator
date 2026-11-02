# -*- coding: utf-8 -*-
"""OpenAI-compatible translator; works with OpenAI, Ollama and compatible gateways."""

from translation.translator.base_tanslator import BaseTranslator
import requests


class OpenAICompatibleTranslator(BaseTranslator):
    def __init__(self):
        super().__init__("openai")

    def translate(self, text):
        if not text:
            return ""

        api_base = str(self.api_config.get("api_base") or "http://localhost:11434/v1").strip().rstrip("/")
        api_key = str(self.api_config.get("api_key") or "").strip()
        model = str(self.api_config.get("model") or "").strip()
        if not model:
            raise ValueError("Укажите название модели в настройках OpenAI-совместимого API.")

        if api_base.endswith("/chat/completions"):
            url = api_base
        else:
            url = api_base + "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key

        system_prompt = (
            "Ты профессиональный переводчик промптов для генерации изображений. "
            "Переводи естественный английский текст на русский, сохраняя смысл и порядок элементов. "
            "Не изменяй числа, имена, названия моделей, имена файлов, URL, идентификаторы, "
            "теги <lora:...>, embedding-теги и другие технические токены. "
            "Сохраняй скобки, квадратные скобки, фигурные скобки, веса, двоеточия, запятые "
            "и структуру исходного промпта. Ничего не добавляй и не удаляй. "
            "Верни только перевод без пояснений и кавычек."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "stream": False,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code >= 400:
            detail = response.text[:1200]
            raise RuntimeError(f"OpenAI-совместимый API вернул HTTP {response.status_code}: {detail}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-совместимый API не вернул choices.")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        translated = str(content).strip()
        if not translated:
            raise RuntimeError("OpenAI-совместимый API вернул пустой перевод.")
        return translated
