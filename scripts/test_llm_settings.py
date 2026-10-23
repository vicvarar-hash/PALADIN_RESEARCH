"""Session credentials and provider routing, without network calls.

Run: python -m unittest discover -s scripts -p test_llm_settings.py
"""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from app.services import llm_provider
from app.services.llm_inference_log import save_log
from app.services.llm_inference_producer import produce_inference_log


EMPTY_ENV = {
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "GEMINI_API_KEY": "",
    "AZURE_FOUNDRY_API_KEY": "",
    "AZURE_OPENAI_API_KEY": "",
    "AZURE_FOUNDRY_ENDPOINT": "",
    "AZURE_FOUNDRY_API_VERSION": "",
}
SERVER_KEYS = {name: f"server-test-{name}" for name in EMPTY_ENV if name.endswith("_API_KEY")}
PROVIDERS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic (Claude)",
    "google": "Google (Gemini)",
    "azure_foundry": "Azure AI Foundry",
}
AZURE_ENDPOINT = "https://user-resource.cognitiveservices.azure.com"


def settings_app():
    import streamlit as st
    from app.ui.llm_settings import get_llm_config, render_llm_settings_sidebar

    render_llm_settings_sidebar()
    st.session_state["test_config"] = get_llm_config()


def config_reader_app():
    import streamlit as st
    from app.ui.llm_settings import get_llm_config

    st.session_state["test_config"] = get_llm_config()


def experiment_app():
    from app.ui.llm_settings import render_llm_settings_sidebar
    from app.ui.experiment_lab import render_experiment_lab

    render_llm_settings_sidebar()
    render_experiment_lab([], [])


def prediction_app():
    from app.loaders.astra_loader import load_astra_dataset
    from app.loaders.mcp_loader import load_mcp_personas
    from app.ui.llm_settings import render_llm_settings_sidebar
    from app.ui.prediction_lab import render_prediction_lab

    render_llm_settings_sidebar()
    tasks = load_astra_dataset("datasets/astra_03_tools.json")
    personas, _ = load_mcp_personas("mcp_servers")
    render_prediction_lab(tasks[:1], personas)


class IsolatedEnvironment(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, EMPTY_ENV)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)


class SidebarTests(IsolatedEnvironment):
    def app(self, render=settings_app):
        app = AppTest.from_function(render, default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def configure(self, app, provider, model="my-custom-model"):
        app.selectbox("sidebar_llm_provider_label").select(PROVIDERS[provider]).run()
        app.text_input(f"sidebar_llm_api_key_{provider}").set_value(
            f"test-{provider}-key"
        ).run()
        custom = "Custom deployment..." if provider == "azure_foundry" else "Custom model..."
        app.selectbox(f"sidebar_llm_model_{provider}").select(custom).run()
        app.text_input(f"sidebar_llm_custom_model_{provider}").set_value(model).run()
        if provider == "azure_foundry":
            app.text_input("sidebar_llm_endpoint_azure_foundry").set_value(AZURE_ENDPOINT).run()
            app.text_input("sidebar_llm_api_version_azure_foundry").set_value("2024-10-21").run()
        self.assertFalse(app.exception)
        return app

    def test_custom_keys_and_models_for_every_provider(self):
        before = dict(os.environ)
        with patch("app.ui.llm_settings.test_connection") as connect:
            app = self.app()
            for provider in PROVIDERS:
                with self.subTest(provider=provider):
                    self.configure(app, provider)
                    config = app.session_state["test_config"]
                    self.assertEqual(config["provider"], provider)
                    self.assertEqual(config["api_key"], f"test-{provider}-key")
                    self.assertEqual(config["model"], "my-custom-model")
                    self.assertFalse(app.button("sidebar_test_llm_conn").disabled)
                    self.assertEqual(
                        config["endpoint"], AZURE_ENDPOINT if provider == "azure_foundry" else None
                    )
            connect.assert_not_called()
        self.assertEqual(dict(os.environ), before)

    def test_switching_providers_preserves_private_settings(self):
        app = self.app()
        self.configure(app, "openai", "openai-custom")
        self.configure(app, "anthropic", "anthropic-custom")
        app.selectbox("sidebar_llm_provider_label").select("OpenAI").run()
        self.assertEqual(app.text_input("sidebar_llm_api_key_openai").value, "test-openai-key")
        self.assertEqual(app.session_state["test_config"]["model"], "openai-custom")
        app.text_input("sidebar_llm_api_key_openai").set_value("").run()
        app.selectbox("sidebar_llm_provider_label").select("Anthropic (Claude)").run()
        self.assertEqual(app.session_state["test_config"]["api_key"], "test-anthropic-key")
        app.selectbox("sidebar_llm_provider_label").select("OpenAI").run()
        self.assertEqual(app.session_state["test_config"]["api_key"], "")
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)

    def test_credentials_do_not_cross_sessions(self):
        first = self.configure(self.app(), "openai")
        second = self.app()
        self.assertEqual(second.session_state["test_config"]["api_key"], "")
        self.assertEqual(first.session_state["test_config"]["api_key"], "test-openai-key")

    def test_switching_between_presets_and_custom_models(self):
        app = self.configure(self.app(), "openai", "my-custom-model")
        app.selectbox("sidebar_llm_model_openai").select("gpt-4o-mini").run()
        self.assertEqual(app.session_state["test_config"]["model"], "gpt-4o-mini")
        app.selectbox("sidebar_llm_model_openai").select("Custom model...").run()
        self.assertEqual(app.session_state["test_config"]["model"], "my-custom-model")

    def test_server_credentials_are_not_offered_or_used(self):
        with patch.dict(os.environ, SERVER_KEYS):
            app = self.app()
            for provider, label in PROVIDERS.items():
                with self.subTest(provider=provider):
                    app.selectbox("sidebar_llm_provider_label").select(label).run()
                    self.assertNotIn("Credentials", [box.label for box in app.selectbox])
                    field = app.text_input(f"sidebar_llm_api_key_{provider}")
                    self.assertEqual(field.value, "")
                    self.assertFalse(field.disabled)
                    self.assertEqual(app.session_state["test_config"]["api_key"], "")
                    self.assertTrue(app.button("sidebar_test_llm_conn").disabled)
                    field.set_value("my-test-key").run()
                    self.assertEqual(app.session_state["test_config"]["api_key"], "my-test-key")
                    app.text_input(f"sidebar_llm_api_key_{provider}").set_value("").run()
                    self.assertEqual(app.session_state["test_config"]["api_key"], "")
                    self.assertTrue(app.button("sidebar_test_llm_conn").disabled)

    def test_google_environment_alias_is_ignored(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key"}):
            app = self.app()
            app.selectbox("sidebar_llm_provider_label").select("Google (Gemini)").run()
            self.assertEqual(app.session_state["test_config"]["api_key"], "")
            self.assertEqual(app.text_input("sidebar_llm_api_key_google").value, "")

    def test_legacy_resolved_keys_are_never_reused(self):
        with patch.dict(os.environ, SERVER_KEYS):
            app = AppTest.from_function(config_reader_app, default_timeout=30)
            for provider in PROVIDERS:
                with self.subTest(provider=provider):
                    app.session_state["llm_provider"] = provider
                    app.session_state["llm_api_key"] = "legacy-server-key"
                    app.run()
                    self.assertEqual(app.session_state["test_config"]["api_key"], "")
            app = AppTest.from_function(settings_app, default_timeout=30)
            app.session_state["llm_api_key"] = "legacy-server-key"
            app.session_state["llm_provider_settings"] = {
                "openai": {"credential_source": "Server environment"}
            }
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["test_config"]["api_key"], "")
            self.assertTrue(app.button("sidebar_test_llm_conn").disabled)

    def test_incomplete_settings_disable_connection_test(self):
        app = self.app()
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)
        app.text_input("sidebar_llm_api_key_openai").set_value("test-key").run()
        app.selectbox("sidebar_llm_model_openai").select("Custom model...").run()
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)
        app.text_input("sidebar_llm_custom_model_openai").set_value("   ").run()
        self.assertEqual(app.session_state["test_config"]["model"], "")
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)
        app.selectbox("sidebar_llm_provider_label").select("Azure AI Foundry").run()
        app.text_input("sidebar_llm_api_key_azure_foundry").set_value("test-key").run()
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)
        app.text_input("sidebar_llm_endpoint_azure_foundry").set_value("http://insecure.example").run()
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)
        app.text_input("sidebar_llm_endpoint_azure_foundry").set_value(AZURE_ENDPOINT).run()
        self.assertFalse(app.button("sidebar_test_llm_conn").disabled)
        app.text_input("sidebar_llm_api_version_azure_foundry").set_value("").run()
        self.assertTrue(app.button("sidebar_test_llm_conn").disabled)

    def test_connection_uses_session_configuration_and_surfaces_errors(self):
        app = self.configure(self.app(), "azure_foundry")
        with patch("app.ui.llm_settings.test_connection", return_value=(False, "Access denied")) as connect:
            app.button("sidebar_test_llm_conn").click().run()
        connect.assert_called_once_with(**app.session_state["test_config"])
        self.assertIn("Access denied", [error.value for error in app.error])
        self.assertFalse(app.exception)

    def test_azure_settings_are_user_supplied_despite_server_defaults(self):
        server_endpoint = "https://server-resource.cognitiveservices.azure.com"
        with patch.dict(os.environ, {
            "AZURE_FOUNDRY_API_KEY": "server-azure-key",
            "AZURE_FOUNDRY_ENDPOINT": server_endpoint,
            "AZURE_FOUNDRY_API_VERSION": "server-version",
        }):
            app = self.app()
            app.selectbox("sidebar_llm_provider_label").select("Azure AI Foundry").run()
            self.assertEqual(app.session_state["test_config"]["api_key"], "")
            self.assertEqual(app.session_state["test_config"]["endpoint"], "")
            self.assertEqual(
                app.session_state["test_config"]["api_version"], llm_provider.DEFAULT_AZURE_API_VERSION
            )
            self.assertFalse(app.text_input("sidebar_llm_endpoint_azure_foundry").disabled)
            self.configure(app, "azure_foundry")
            config = app.session_state["test_config"]
            self.assertEqual(config["endpoint"], AZURE_ENDPOINT)
            self.assertEqual(config["api_key"], "test-azure_foundry-key")
            self.assertEqual(os.environ["AZURE_FOUNDRY_ENDPOINT"], server_endpoint)

    def test_both_labs_require_a_user_key_even_with_server_keys(self):
        with patch.dict(os.environ, SERVER_KEYS), patch(
            "app.ui.prediction_lab.SpiffeWorkloadService.fetch_real_identity",
            return_value=(None, "Simulated"),
        ):
            app = self.app(experiment_app)
            button = next(b for b in app.button if b.label.endswith("Run LLM inference"))
            self.assertTrue(button.disabled)
            app = self.app(prediction_app)
            button = next(b for b in app.button if b.label.endswith("Run Execution Pipeline"))
            self.assertTrue(button.disabled)

    def test_experiment_lab_passes_azure_settings(self):
        app = self.app(experiment_app)
        run_button = next(b for b in app.button if b.label.endswith("Run LLM inference"))
        self.assertTrue(run_button.disabled)
        self.configure(app, "azure_foundry", "my-deployment")
        log = {"tasks": [], "mode": "selection", "model": "my-deployment", "retrieval": {}}
        with patch("app.ui.experiment_lab.produce_and_save", return_value=("mock-log.json", log)) as produce:
            next(b for b in app.button if b.label.endswith("Run LLM inference")).click().run()
        self.assertFalse(app.exception)
        self.assertEqual(produce.call_args.kwargs["api_key"], "test-azure_foundry-key")
        self.assertEqual(produce.call_args.kwargs["model"], "my-deployment")
        self.assertEqual(produce.call_args.kwargs["provider"], "azure_foundry")
        self.assertEqual(produce.call_args.kwargs["endpoint"], AZURE_ENDPOINT)
        self.assertEqual(produce.call_args.kwargs["api_version"], "2024-10-21")

    def test_prediction_lab_passes_shared_configuration(self):
        with patch("app.ui.prediction_lab.LLMProvider") as factory, patch(
            "app.ui.prediction_lab.SpiffeWorkloadService.fetch_real_identity",
            return_value=(None, "Simulated"),
        ):
            factory.return_value.configuration_error = ""
            app = self.configure(self.app(prediction_app), "azure_foundry", "my-deployment")
            factory.assert_called_with(
                provider="azure_foundry", model="my-deployment",
                api_key="test-azure_foundry-key", endpoint=AZURE_ENDPOINT,
                api_version="2024-10-21",
            )
            self.assertFalse(app.exception)
            self.assertEqual(
                app.session_state["last_context_key"][-2:], ("azure_foundry", "my-deployment")
            )
            app.session_state["last_run_data"] = {"stale": True}
            app.text_input("sidebar_llm_custom_model_azure_foundry").set_value(
                "different-deployment"
            ).run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["last_run_data"])
            self.assertEqual(app.session_state["last_context_key"][-1], "different-deployment")


class ProviderTests(IsolatedEnvironment):
    def test_missing_or_empty_keys_never_fall_back_to_server_keys(self):
        backends = {provider: Mock() for provider in PROVIDERS}
        with patch.dict(os.environ, SERVER_KEYS), patch.dict(llm_provider._BACKENDS, backends):
            for provider in PROVIDERS:
                for arguments in ({}, {"api_key": None}, {"api_key": ""}, {"api_key": "   "}):
                    with self.subTest(provider=provider, arguments=arguments):
                        llm = llm_provider.LLMProvider(provider=provider, **arguments)
                        self.assertFalse(llm.is_configured())
                        with self.assertRaisesRegex(ValueError, "No API key"):
                            llm.query("system", "user")
                backends[provider].assert_not_called()

    def test_each_public_backend_receives_the_custom_key_and_model(self):
        cases = (
            ("openai", "openai.OpenAI"),
            ("anthropic", "anthropic.Anthropic"),
            ("google", "google.genai.Client"),
        )
        for provider, target in cases:
            with self.subTest(provider=provider), patch(target) as factory:
                client = factory.return_value
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
                )
                client.messages.create.return_value = SimpleNamespace(
                    content=[SimpleNamespace(text='{"ok": true}')]
                )
                client.models.generate_content.return_value = SimpleNamespace(text='{"ok": true}')
                llm = llm_provider.LLMProvider(
                    provider=provider, api_key="my-test-key", model="my-model"
                )
                self.assertTrue(llm.is_configured())
                self.assertEqual(llm.query("system", "user"), '{"ok": true}')
                factory.assert_called_once_with(api_key="my-test-key")
                call = {
                    "openai": client.chat.completions.create,
                    "anthropic": client.messages.create,
                    "google": client.models.generate_content,
                }[provider]
                self.assertEqual(call.call_args.kwargs["model"], "my-model")

    def test_azure_clients_have_independent_credentials_and_endpoints(self):
        with patch.dict(os.environ, {
            "AZURE_FOUNDRY_API_KEY": "server-test-key",
            "AZURE_FOUNDRY_ENDPOINT": "https://server.cognitiveservices.azure.com",
            "AZURE_FOUNDRY_API_VERSION": "server-version",
        }), patch("openai.AzureOpenAI") as factory:
            before = dict(os.environ)
            first = llm_provider.LLMProvider(
                provider="azure", api_key="user-test-key", model="my-deployment",
                endpoint=AZURE_ENDPOINT + "/openai/responses?api-version=ignored",
                api_version="2024-10-21",
            )
            self.assertTrue(first.is_configured())
            factory.assert_called_with(
                api_key="user-test-key", azure_endpoint=AZURE_ENDPOINT, api_version="2024-10-21"
            )
            llm_provider.LLMProvider(provider="azure_foundry", api_key="second-user-key")
            factory.assert_called_with(
                api_key="second-user-key",
                azure_endpoint="https://server.cognitiveservices.azure.com",
                api_version="server-version",
            )
            self.assertEqual(dict(os.environ), before)

    def test_azure_initialization_errors_are_preserved(self):
        llm = llm_provider.LLMProvider(provider="azure_foundry", api_key="test-key", endpoint="")
        self.assertFalse(llm.is_configured())
        self.assertIn("Provide an Azure endpoint", llm.configuration_error)
        with self.assertRaisesRegex(ValueError, "Provide an Azure endpoint"):
            llm.query("system", "user")
        for endpoint in ("http://example.com", "https://user:password@example.com",
                         "https://example.com/api/projects/my-project"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                llm_provider.normalize_azure_endpoint(endpoint)

    def test_connection_errors_redact_keys_and_empty_responses_fail(self):
        backend = Mock()
        backend.query.side_effect = RuntimeError("Rejected key test-private-key")
        with patch.dict(llm_provider._BACKENDS, {"openai": Mock(return_value=backend)}):
            ok, message = llm_provider.test_connection("openai", " test-private-key ")
            self.assertFalse(ok)
            self.assertNotIn("test-private-key", message)
            self.assertIn("[redacted]", message)
            backend.query.side_effect = None
            backend.query.return_value = ""
            ok, message = llm_provider.test_connection("openai", "test-private-key")
            self.assertFalse(ok)
            self.assertIn("empty response", message)

    def test_batch_pipeline_forwards_settings_without_persisting_credentials(self):
        with patch("app.services.llm_provider.LLMProvider") as factory:
            factory.return_value.is_configured.return_value = True
            log = produce_inference_log(
                [], [], provider="azure_foundry", model="my-deployment",
                api_key="test-private-key", endpoint=AZURE_ENDPOINT,
                api_version="2024-10-21", mode="selection",
            )
            factory.assert_called_once_with(
                provider="azure_foundry", model="my-deployment",
                api_key="test-private-key", endpoint=AZURE_ENDPOINT, api_version="2024-10-21",
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.json"
            save_log(str(path), log)
            persisted = path.read_text(encoding="utf-8")
        self.assertNotIn("test-private-key", persisted)
        self.assertNotIn(AZURE_ENDPOINT, persisted)
        self.assertNotIn("api_key", persisted)
        self.assertEqual(json.loads(persisted)["model"], "my-deployment")

    def test_batch_inference_rejects_missing_user_keys(self):
        with patch.dict(os.environ, SERVER_KEYS):
            for provider in PROVIDERS:
                with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, "No API key"):
                    produce_inference_log(
                        [], [], provider=provider, model=llm_provider.DEFAULT_MODEL[provider],
                        api_key="", endpoint=AZURE_ENDPOINT, api_version="2024-10-21",
                        mode="selection",
                    )


if __name__ == "__main__":
    unittest.main()
