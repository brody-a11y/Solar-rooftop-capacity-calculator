"""Keep API keys and tokens out of messages, reasons and CSVs."""

import re

_SECRET = re.compile(r"((?:key|token|api_key|apikey)=)[^&\s'\"]+", re.I)


def redact(text: object) -> str:
    """str(text) with key=/token= query values replaced by '***'."""
    return _SECRET.sub(r"\1***", str(text))
