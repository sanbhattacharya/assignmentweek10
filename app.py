"""Simple Streamlit app that demos a Hugging Face inference call."""

from __future__ import annotations

import json
from typing import Any, Dict

import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")


def main() -> None:
    st.title("My AI Chat")

    token = st.secrets.get("HF_TOKEN", "")
    if not token or not token.strip():
        st.error(
            "The Hugging Face token is missing. Please add `HF_TOKEN` to your `.streamlit/secrets.toml`."
        )
        st.stop()

    model_response = send_test_prompt(token.strip(), "Hello!")
    error = model_response.get("error")

    if error:
        st.error(error)
        return

    st.subheader("Model response")
    st.markdown(model_response.get("text", "No text was returned by the model."))
    extra = model_response.get("details")
    if extra:
        st.caption(f"Raw payload: {extra}")


def send_test_prompt(token: str, prompt: str) -> Dict[str, Any]:
    url = "https://api-inference.huggingface.co/models/openai-community/gpt2"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return {"error": f"Unable to reach Hugging Face inference API: {str(exc)}"}

    try:
        data = response.json()
    except ValueError:
        return {"error": "Hugging Face returned an unexpected response."}

    # FIX STARTS HERE: Split the check into two lines
    if isinstance(data, dict):
        api_error = data.get("error")
        if api_error:
            return {
                "error": f"Hugging Face API error: {api_error}", 
                "details": json.dumps(data, ensure_ascii=False)
            }

    text = parse_response_text(data)
    return {"text": text, "details": json.dumps(data, ensure_ascii=False)}


def parse_response_text(data: Any) -> str:
    if isinstance(data, list) and data:
        candidate = data[0]
        if isinstance(candidate, dict):
            for key in ("generated_text", "text", "output"):
                if key in candidate:
                    return str(candidate[key])
        return json.dumps(candidate, ensure_ascii=False)

    if isinstance(data, dict):
        for key in ("generated_text", "text", "output"):
            if key in data:
                return str(data[key])
        return json.dumps(data, ensure_ascii=False)

    return str(data)


if __name__ == "__main__":
    main()
