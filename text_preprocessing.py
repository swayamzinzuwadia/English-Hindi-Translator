def preprocess_text(text, language="english"):
    if not isinstance(text, str):
        return text

    if language == "english":
        pattern = re.compile(r"[^a-zA-Z0-9\s]")
        return pattern.sub(r"", text)
    elif language == "hindi":
        pattern = re.compile(r"[^\u0900-\u097F\s]")
        return pattern.sub(r"", text)
    else:
        raise ValueError(
            "Unsupported Language, Supported languages are 'english' and 'hindi'"
        )