"""LLM client abstraction layer for the ZCAgent system."""

import json
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class LLMClient:
    """Abstraction layer for LLM API calls with mock support for testing."""

    def __init__(self, config: dict | None = None, mock_response: str | None = None):
        if config is None:
            config = self._load_config()
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)
        self.api_key = config.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = config.get("api_base", "") or os.environ.get("OPENAI_API_BASE", "")
        self._mock_response = mock_response
        self._client = None

    def _load_config(self) -> dict:
        """Load LLM configuration, trying ../../config/config.yaml then config/config.yaml."""
        config_paths = [
            os.path.join(os.path.dirname(__file__), "../../config/config.yaml"),
            "config/config.yaml",
        ]
        for path in config_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    full_config = yaml.safe_load(f)
                return full_config.get("llm", {})
        return {}

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                kwargs = {"api_key": self.api_key}
                if self.api_base:
                    kwargs["base_url"] = self.api_base
                self._client = OpenAI(**kwargs)
            except ImportError:
                logger.warning("openai package not installed, using mock mode")
                self._mock_response = self._mock_response or "Mock response"
        return self._client

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The assistant's response text.
        """
        if self._mock_response is not None:
            logger.debug("Using mock response: %s", self._mock_response)
            return self._mock_response

        client = self._get_client()
        if client is None:
            return self._mock_response or ""

        try:
            response = client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            raise

    def generate_json(self, messages: list[dict[str, str]], **kwargs) -> Any:
        """Generate a JSON response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Parsed JSON object from the response.
        """
        raw = self.generate(messages, **kwargs)
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in raw:
                parts = raw.split("```json", 1)[1]
                raw = parts.split("```", 1)[0].strip() if "```" in parts else parts.strip()
            elif "```" in raw:
                parts = raw.split("```", 1)[1]
                raw = parts.split("```", 1)[0].strip() if "```" in parts else parts.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from LLM response: %s\nResponse preview: %.100s", e, raw)
            raise ValueError(f"LLM response is not valid JSON: {e}") from e
