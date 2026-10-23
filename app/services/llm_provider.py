"""
Unified LLM provider façade.

Supports OpenAI, Anthropic (Claude), Google (Gemini), and Azure AI Foundry
behind a single ``LLMProvider`` interface so the rest of the codebase doesn't
care which backend is in use. All backends are configured to return a JSON
object as the response body.

Usage::

    llm = LLMProvider(provider="anthropic", api_key=..., model="claude-3-5-sonnet-latest")
    raw_json = llm.query(system_prompt, user_prompt)

An explicit user-provided ``api_key`` is required to configure a client.
Missing or empty keys leave it disabled, even if provider keys exist in the
server environment. Non-secret Azure endpoint/API-version environment defaults
remain available to command-line callers that supply their own key; the web app
passes the user's connection settings explicitly.

The legacy single-arg form ``LLMProvider(api_key=...)`` defaults to OpenAI
with ``gpt-4o`` for backwards compatibility.
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlsplit


DEFAULT_AZURE_API_VERSION = "2024-10-21"


# ── Catalog of supported models per provider ─────────────────────────────
PROVIDER_MODELS: Dict[str, List[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    # Suggested Azure deployment names; the sidebar also accepts custom names.
    "azure_foundry": [
        "gpt-5.4",
        "gpt-4o",
        "gpt-35-turbo-16k",
    ],
}

# Default model per provider when none is specified.
DEFAULT_MODEL: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-2.5-flash",
    "azure_foundry": "gpt-5.4",
}

def normalize_provider(name: Optional[str]) -> str:
    """Map UI labels (e.g. 'Claude', 'Gemini', 'Anthropic (Claude)') to canonical provider keys."""
    if not name:
        return "openai"
    n = name.strip().lower()
    # Strip parenthesized suffix like "anthropic (claude)" -> "anthropic"
    if "(" in n:
        n = n.split("(", 1)[0].strip()
    if n in ("openai", "open ai", "gpt"):
        return "openai"
    if n in ("anthropic", "claude"):
        return "anthropic"
    if n in ("google", "gemini", "google ai", "google gemini"):
        return "google"
    if n in ("azure foundry", "azure_foundry", "foundry", "azure", "azure openai",
             "azure ai foundry", "microsoft foundry"):
        return "azure_foundry"
    return n  # let downstream raise if truly unknown


def normalize_azure_endpoint(endpoint: str) -> str:
    """Accept a resource URL or a copied /openai URL, without URL credentials."""
    parsed = urlsplit(endpoint.strip())
    if (parsed.scheme != "https" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("Azure endpoint must be an HTTPS resource URL without credentials.")
    if parsed.path.rstrip("/") and not parsed.path.startswith("/openai"):
        raise ValueError(
            "Use the Azure resource base URL, not a project or model-inference URL."
        )
    return f"{parsed.scheme}://{parsed.netloc}"


# ── Backend interface ────────────────────────────────────────────────────

class _Backend:
    name: str = "<abstract>"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def query(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class _OpenAIBackend(_Backend):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def query(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class _AnthropicBackend(_Backend):
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def query(self, system_prompt: str, user_prompt: str) -> str:
        # Anthropic has no native json_object mode; we strengthen the system
        # prompt to require a single JSON object. The downstream parser
        # already strips markdown fences as a defensive fallback.
        sys = (system_prompt or "") + (
            "\n\nIMPORTANT: Respond with ONLY a single JSON object. "
            "Do not wrap it in markdown code fences. Do not include any prose."
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=sys,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # response.content is a list of content blocks; concatenate text parts.
        parts = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


class _GoogleBackend(_Backend):
    name = "google"

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def query(self, system_prompt: str, user_prompt: str) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=cfg,
        )
        text = getattr(resp, "text", None)
        if text:
            return text
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return ""


class _AzureFoundryBackend(_Backend):
    """Azure AI Foundry / Azure OpenAI backend.

    Uses the Chat Completions API against a deployment hosted on a Foundry
    resource. Accepts an explicit endpoint or reads ``AZURE_FOUNDRY_ENDPOINT``
    (e.g. ``https://<resource>.cognitiveservices.azure.com``) and the
    API version from ``AZURE_FOUNDRY_API_VERSION`` (defaults to a recent
    GA value). ``model`` here is the *deployment name* shown in Foundry,
    not the underlying model id.
    """
    name = "azure_foundry"

    DEFAULT_API_VERSION = DEFAULT_AZURE_API_VERSION

    def __init__(self, api_key: str, model: str, *,
                 endpoint: Optional[str] = None,
                 api_version: Optional[str] = None):
        super().__init__(api_key, model)
        from openai import AzureOpenAI
        endpoint = (os.getenv("AZURE_FOUNDRY_ENDPOINT", "")
                    if endpoint is None else endpoint).strip()
        if not endpoint:
            raise ValueError(
                "Provide an Azure endpoint in the sidebar or set "
                "AZURE_FOUNDRY_ENDPOINT to your Foundry "
                "resource base URL, e.g. "
                "https://<resource>.cognitiveservices.azure.com"
            )
        endpoint = normalize_azure_endpoint(endpoint)
        if api_version is None:
            api_version = os.getenv("AZURE_FOUNDRY_API_VERSION") or self.DEFAULT_API_VERSION
        api_version = api_version.strip()
        if not api_version:
            raise ValueError("Provide an Azure API version.")
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._endpoint = endpoint
        self._api_version = api_version

    def query(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,  # deployment name on Foundry
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


_BACKENDS = {
    "openai": _OpenAIBackend,
    "anthropic": _AnthropicBackend,
    "google": _GoogleBackend,
    "azure_foundry": _AzureFoundryBackend,
}


# ── Public façade ────────────────────────────────────────────────────────

class LLMProvider:
    """Provider-agnostic LLM client.

    Parameters
    ----------
    api_key : str, optional
        User-provided API key for the chosen provider. If omitted or empty,
        the client remains unconfigured. Server keys are never used.
    model : str, optional
        Model name. If omitted, the provider's default model is used.
    provider : str, optional
        ``"openai"``, ``"anthropic"``, ``"google"``, or ``"azure_foundry"``.
        Defaults to ``"openai"`` for backwards compatibility. Also accepts
        common aliases like ``"claude"``, ``"gemini"``, or ``"foundry"``.
    endpoint, api_version : str, optional
        Azure resource endpoint and API version. Omitted values use the
        environment; explicit values are local to this client.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 provider: Optional[str] = None, *,
                 endpoint: Optional[str] = None,
                 api_version: Optional[str] = None):
        self.provider = normalize_provider(provider)
        if self.provider not in _BACKENDS:
            raise ValueError(f"Unknown provider: {self.provider!r}. "
                             f"Supported: {sorted(_BACKENDS)}")
        self.api_key = (api_key or "").strip()
        self.model = DEFAULT_MODEL[self.provider] if model is None else model.strip()
        self._backend: Optional[_Backend] = None
        self.configuration_error = ""
        if not self.api_key:
            self.configuration_error = (
                f"No API key provided for {self.provider}. Enter your key in the sidebar "
                "or pass api_key explicitly."
            )
        elif not self.model:
            self.configuration_error = "Select a model or enter a model/deployment name."
        else:
            try:
                if self.provider == "azure_foundry":
                    self._backend = _BACKENDS[self.provider](
                        self.api_key, self.model, endpoint=endpoint, api_version=api_version,
                    )
                else:
                    self._backend = _BACKENDS[self.provider](self.api_key, self.model)
            except (ImportError, ValueError) as exc:
                self.configuration_error = (
                    f"{type(exc).__name__}: {exc}".replace(self.api_key, "[redacted]")
                )

    def is_configured(self) -> bool:
        return self._backend is not None

    def query(self, system_prompt: str, user_prompt: str) -> str:
        if not self._backend:
            raise ValueError(self.configuration_error)
        return self._backend.query(system_prompt, user_prompt)


# ── Connection test ──────────────────────────────────────────────────────

def test_connection(provider: str, api_key: str, model: Optional[str] = None, *,
                    endpoint: Optional[str] = None,
                    api_version: Optional[str] = None) -> Tuple[bool, str]:
    """Issue a minimal request to verify provider + key + model work.

    Returns (ok, message). The message is suitable for direct display in a UI.
    """
    provider = normalize_provider(provider)
    api_key = api_key.strip()
    if not api_key:
        return False, f"No API key provided for {provider}."
    try:
        llm = LLMProvider(api_key=api_key, model=model, provider=provider,
                          endpoint=endpoint, api_version=api_version)
        if not llm.is_configured():
            return False, llm.configuration_error
        raw = llm.query(
            "You are a connectivity test. Respond with valid JSON only.",
            'Reply with exactly this JSON and nothing else: {"ok": true}',
        )
        if not raw.strip():
            return False, "The model returned an empty response."
        return True, f"OK · {provider}/{llm.model} responded ({len(raw)} chars)."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}".replace(api_key, "[redacted]")
