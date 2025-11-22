# src/llm/backends.py

import os
import json
import re
from typing import List, Dict, Optional

# Azure OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# Mistral
try:
    from mistralai import Mistral
except Exception:
    Mistral = None


def _extract_json_block(text: str) -> Optional[str]:
    """
    Try to robustly extract a top-level JSON object from a text response.
    Returns the JSON string if found, else None.
    """
    if not text:
        return None

    # Fast path: already valid JSON
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Heuristic: find first {...} block
    # This tries to match a top-level balanced JSON object
    stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start != -1:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
    return None


class BaseBackend:
    provider: str
    model_name: str

    def name(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def chat_json(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float = 0.0,
        force_json: bool = True,
    ) -> str:
        """
        Return a JSON string. Backends may add 'response_format' if supported.
        If the model returns extra commentary, try to extract the JSON block.
        """
        text = self._chat_raw(system_message, user_prompt, temperature, force_json)
        if not force_json:
            return text or ""

        block = _extract_json_block(text or "")
        if block is None:
            # As a last resort, wrap empty object
            return "{}"
        return block

    # Implement in subclasses
    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
    ) -> str:
        raise NotImplementedError()


class AzureBackend(BaseBackend):
    provider = "azure"

    def __init__(self):
        if AzureOpenAI is None:
            raise RuntimeError("openai package is not installed")

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-dspy")

        if not api_key or not endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT missing")

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        self.model_name = deployment

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
    ) -> str:
        kwargs = dict(
            model=self.model_name,  # deployment name in Azure
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


class MistralBackend(BaseBackend):
    provider = "mistral"

    def __init__(self):
        if Mistral is None:
            raise RuntimeError("mistralai package is not installed")

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY missing")

        # Default model. Override with MISTRAL_MODEL if you prefer another
        self.model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        self._client = Mistral(api_key=api_key)

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
    ) -> str:
        """
        Mistral does not yet enforce JSON in the same way Azure does.
        We still ask for JSON in the prompt and post-process.
        """
        messages = [
            {"role": "system", "content": system_message + " Always return a single JSON object."},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        # SDK returns choices[0].message.content
        return resp.choices[0].message.content or ""


def make_backend():
    provider = os.getenv("LLM_PROVIDER", "azure").lower()
    if provider == "azure":
        return AzureBackend()
    elif provider == "mistral":
        return MistralBackend()
    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")
