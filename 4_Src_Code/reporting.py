"""Model configuration and explicit failures for the mentoring workflow."""

import os

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class MentorReportError(RuntimeError):
    """The workflow could not produce a complete mentoring report."""


def get_groq_model(override=None):
    """Resolve a model without treating client construction as a health check."""
    if override and override.strip():
        return override.strip()
    try:
        import streamlit as st

        value = st.secrets.get("groq", {}).get("GROQ_MODEL", "")
        if value and value.strip():
            return value.strip()
    except Exception:
        # A CLI run need not have a Streamlit secrets file.
        pass
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL


def provider_failure(error, model):
    """Give actionable feedback without exposing raw provider payloads."""
    status = getattr(error, "status_code", None)
    if status == 404:
        return (
            f"Groq model '{model}' is unavailable to this account. "
            "Set GROQ_MODEL to a model available to the deployment's Groq account."
        )
    if status in (401, 403):
        return "Groq rejected the request. Check the API key and account model permissions."
    if status == 429:
        return "Groq's rate limit was reached. Wait before trying again."
    return "The Groq request failed. Check service availability and try again."
