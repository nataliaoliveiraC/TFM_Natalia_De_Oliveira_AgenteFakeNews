from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

def detect_language(title, body):
    """
    Detecta el idioma principal de una noticia.

    Devuelve:
    - "en" para inglés
    - "es" para español
    """
    text = f"{title} {body}".strip()

    detected_language = detect(text)

    if detected_language not in ["en", "es"]:
        raise ValueError(
            f"Idioma no soportado: {detected_language}"
        )

    return detected_language