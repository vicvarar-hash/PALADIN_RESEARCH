"""Shared sidebar widget for LLM provider/model selection.

Renders once in the sidebar and writes the user's choice to ``st.session_state``
for both the Experiment LLM Lab and the Prediction Lab:

* ``llm_provider`` — canonical key: ``openai`` / ``anthropic`` / ``google`` / ``azure_foundry``
* ``llm_model``    — model id (e.g. ``gpt-4o``, ``claude-sonnet-4-6``)
* ``llm_api_key``  — session-only user-provided key; no server-key fallback
* ``llm_endpoint`` / ``llm_api_version`` — Azure connection settings

Provider-specific inputs survive switching providers within the same session.
Only "Test connection" and inference actions contact the selected provider.
"""
from __future__ import annotations

import streamlit as st

from app.services.llm_provider import (
    PROVIDER_MODELS,
    DEFAULT_MODEL,
    DEFAULT_AZURE_API_VERSION,
    normalize_azure_endpoint,
    test_connection,
)

_PROVIDER_LABELS = ["OpenAI", "Anthropic (Claude)", "Google (Gemini)", "Azure AI Foundry"]
_LABEL_TO_KEY = {
    "OpenAI": "openai",
    "Anthropic (Claude)": "anthropic",
    "Google (Gemini)": "google",
    "Azure AI Foundry": "azure_foundry",
}
_KEY_TO_LABEL = {v: k for k, v in _LABEL_TO_KEY.items()}

_CUSTOM_MODEL = "Custom model..."
_CUSTOM_DEPLOYMENT = "Custom deployment..."


def _save_setting(provider: str, name: str) -> None:
    st.session_state["llm_provider_settings"][provider][name] = st.session_state[
        f"sidebar_llm_{name}_{provider}"
    ]


def render_llm_settings_sidebar() -> dict:
    """Render and publish session-local settings without changing the environment."""
    st.sidebar.subheader("🤖 LLM Settings")

    default_label = _KEY_TO_LABEL.get(
        st.session_state.get("llm_provider", "openai"), "OpenAI"
    )
    provider_label = st.sidebar.selectbox(
        "Provider",
        _PROVIDER_LABELS,
        index=_PROVIDER_LABELS.index(default_label),
        key="sidebar_llm_provider_label",
        help="Shared by Prediction Lab and Experiment LLM Lab.",
    )
    provider = _LABEL_TO_KEY[provider_label]

    settings = st.session_state.setdefault("llm_provider_settings", {}).setdefault(provider, {})
    defaults = {
        "api_key": "",
        "model": DEFAULT_MODEL[provider],
        "custom_model": "",
        "endpoint": "",
        "api_version": DEFAULT_AZURE_API_VERSION,
    }
    # Streamlit removes hidden widgets; retain their values in non-widget state.
    keys = {name: f"sidebar_llm_{name}_{provider}" for name in defaults}
    for name, default in defaults.items():
        settings.setdefault(name, default)
        st.session_state[keys[name]] = settings[name]

    api_key = st.sidebar.text_input(
        "API key", type="password", key=keys["api_key"],
        on_change=_save_setting, args=(provider, "api_key"),
        placeholder="Paste your provider API key",
        help="Kept in this app session, not written to configuration files. "
             "Delete the value to clear your key and disable LLM actions for this provider.",
    ).strip()
    st.sidebar.caption("Your key is kept only in this app session.")

    custom_option = _CUSTOM_DEPLOYMENT if provider == "azure_foundry" else _CUSTOM_MODEL
    models = PROVIDER_MODELS[provider] + [custom_option]
    if st.session_state[keys["model"]] not in models:
        st.session_state[keys["custom_model"]] = st.session_state[keys["model"]]
        st.session_state[keys["model"]] = custom_option
        settings["custom_model"] = st.session_state[keys["custom_model"]]
        settings["model"] = custom_option
    model_label = "Deployment" if provider == "azure_foundry" else "Model"
    model = st.sidebar.selectbox(
        model_label, models, key=keys["model"],
        on_change=_save_setting, args=(provider, "model"),
        help="Choose a preset or enter an exact model ID/deployment name "
             "available to your account. The model must support text/JSON generation.",
    )
    if model == custom_option:
        model = st.sidebar.text_input(
            "Deployment name" if provider == "azure_foundry" else "Model ID",
            key=keys["custom_model"],
            on_change=_save_setting, args=(provider, "custom_model"),
            help="For Azure, use the deployment name, not the underlying model ID.",
        ).strip()

    endpoint = api_version = None
    if provider == "azure_foundry":
        endpoint = st.sidebar.text_input(
            "Azure endpoint", key=keys["endpoint"],
            on_change=_save_setting, args=(provider, "endpoint"),
            placeholder="https://your-resource.cognitiveservices.azure.com",
            help="Azure OpenAI-compatible resource base URL, not a project URL.",
        ).strip()
        api_version = st.sidebar.text_input(
            "Azure API version", key=keys["api_version"],
            on_change=_save_setting, args=(provider, "api_version"),
        ).strip()

    config = {
        "provider": provider, "model": model, "api_key": api_key,
        "endpoint": endpoint, "api_version": api_version,
    }
    for name, value in config.items():
        st.session_state[f"llm_{name}"] = value

    error = get_llm_config_error(config)
    if error:
        st.sidebar.warning(error)

    test_clicked = st.sidebar.button(
        "🔌 Test connection",
        key="sidebar_test_llm_conn",
        use_container_width=True,
        disabled=bool(error),
        help="Sends a small request to the selected model; provider charges may apply.",
    )
    if test_clicked:
        with st.spinner(f"Pinging {provider}/{model}…"):
            ok, msg = test_connection(**config)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

    return config


def get_llm_config() -> dict:
    """Read the current LLM config from session state. Defaults to OpenAI/gpt-4o
    if the sidebar hasn't run yet (defensive — should not happen in normal flow).
    """
    provider = st.session_state.get("llm_provider", "openai")
    settings = st.session_state.get("llm_provider_settings", {}).get(provider, {})
    return {
        "provider": provider,
        "model": st.session_state.get("llm_model", DEFAULT_MODEL.get(provider, "gpt-4o")),
        # Ignore legacy resolved keys, which could have originated on the server.
        "api_key": settings.get("api_key", "").strip(),
        "endpoint": st.session_state.get(
            "llm_endpoint", settings.get("endpoint", "") if provider == "azure_foundry" else None,
        ),
        "api_version": st.session_state.get(
            "llm_api_version",
            settings.get("api_version", DEFAULT_AZURE_API_VERSION)
            if provider == "azure_foundry" else None,
        ),
    }


def get_llm_config_error(config: dict) -> str | None:
    """Explain incomplete settings before a connection test or inference run."""
    if not config["api_key"]:
        return f"Enter your {_KEY_TO_LABEL[config['provider']]} API key in the sidebar."
    if not config["model"].strip():
        return "Select a model or enter a model/deployment name in the sidebar."
    if config["provider"] == "azure_foundry":
        if not config["endpoint"]:
            return "Provide your Azure resource endpoint in the sidebar."
        try:
            normalize_azure_endpoint(config["endpoint"])
        except ValueError as exc:
            return str(exc)
        if not config["api_version"]:
            return "Provide an Azure API version in the sidebar."
    return None
