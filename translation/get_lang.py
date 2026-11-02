# -*- coding: utf-8 -*-
"""Минимальные сообщения ошибок для адаптеров переводчиков Physton."""

_MESSAGES = {
    "is_required": "Не заполнено обязательное поле: {0}",
    "request_error": "Ошибка запроса к сервису {0}.",
    "response_is_empty": "Сервис {0} вернул пустой ответ.",
    "no_response_from": "Сервис {0} не вернул результат перевода.",
    "response_error": "Не удалось разобрать ответ сервиса {0}.",
    "translate_language_not_support": "Провайдер не поддерживает перевод EN → RU.",
    "model_not_initialized": "Локальная модель перевода ещё не инициализирована.",
}

def get_lang(key, variables=None):
    message = _MESSAGES.get(key, str(key))
    for name, value in (variables or {}).items():
        message = message.replace("{" + str(name) + "}", str(value))
    return message
