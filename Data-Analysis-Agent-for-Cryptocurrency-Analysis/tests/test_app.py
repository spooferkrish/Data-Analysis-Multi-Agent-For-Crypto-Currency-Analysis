import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

import config

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class PriceResponse:
    def raise_for_status(self):
        pass

    def json(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        return {
            "prices": [
                [int((start + timedelta(days=day)).timestamp() * 1000), 100 + day]
                for day in range(60)
            ]
        }


class AnalysisFlowTests(unittest.TestCase):
    def test_one_submit_runs_full_report_with_server_keys(self):
        secrets = {
            "GROQ_API_KEY": "server-groq-test-key",
            "EXA_API_KEY": "server-exa-test-key",
        }
        article = SimpleNamespace(
            title="Ethereum update",
            url="https://example.com/ethereum",
            summary="Market news summary",
            published_date="2026-10-16",
        )
        with tempfile.TemporaryDirectory() as directory:
            secrets_path = Path(directory) / "secrets.toml"
            secrets_path.write_text(
                '\n'.join(f'{name} = "{value}"' for name, value in secrets.items()),
                encoding="utf-8",
            )
            with patch.object(config, "SECRETS_PATH", secrets_path), \
             patch.object(st, "secrets", {}), \
             patch.dict(os.environ, {"GROQ_API_KEY": "", "EXA_API_KEY": ""}), \
             patch("requests.get", return_value=PriceResponse()) as prices, \
             patch("exa_py.Exa") as exa, \
             patch("langchain_groq.ChatGroq") as groq:
                exa.return_value.search_and_contents.return_value.results = [article]
                groq.return_value.invoke.return_value.content = "Market read: neutral."

                app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(prices.call_count, 0)
                self.assertEqual([field.label for field in app.text_input], ["Asset name or symbol"])

                app.text_input[0].set_value("Ethereum")
                app.button[0].click().run()

                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.session_state["symbol"], "ETH")
                self.assertEqual(prices.call_count, 1)
                exa.assert_called_once_with(api_key=secrets["EXA_API_KEY"])
                self.assertEqual(groq.call_args.kwargs["groq_api_key"], secrets["GROQ_API_KEY"])
                self.assertTrue(any("Market read: neutral." in item.value for item in app.markdown))
                self.assertFalse(any(secrets["GROQ_API_KEY"] in item.value for item in app.markdown))

    def test_missing_groq_key_shows_price_and_configuration_message(self):
        with patch.object(config, "SECRETS_PATH", Path(__file__).parent / "no-secrets.toml"), \
             patch.object(st, "secrets", {}), \
             patch.dict(os.environ, {"GROQ_API_KEY": "", "EXA_API_KEY": ""}), \
             patch("requests.get", return_value=PriceResponse()):
            app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
            app.button[0].click().run()

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["symbol"], "BTC")
            self.assertTrue(any("GROQ_API_KEY" in item.value for item in app.warning))
            self.assertFalse(any("API key" in field.label for field in app.text_input))


if __name__ == "__main__":
    unittest.main()
