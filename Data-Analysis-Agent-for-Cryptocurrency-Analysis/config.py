"""Server-side configuration for the Streamlit website."""

import os
import tomllib
from pathlib import Path

import streamlit as st


SECRETS_PATH = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"


def get_secret(name: str) -> str:
    """Read a key from the server environment, app-local file, or host secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        with SECRETS_PATH.open("rb") as stream:
            value = tomllib.load(stream).get(name, "")
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        value = ""
    if value:
        return str(value).strip()

    try:
        return str(st.secrets.get(name, "") or "").strip()
    except FileNotFoundError:
        return ""
