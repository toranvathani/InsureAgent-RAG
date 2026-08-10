"""
Thin wrapper around the LLM provider so agents don't hardcode API details.
Swap the implementation here to move between OpenAI / Azure OpenAI / Anthropic.
"""
import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | openai | azure_openai
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def call_llm(messages: list[dict], temperature: float = 0.0) -> str:
    """
    Sends chat messages to the configured LLM provider and returns raw text
    (expected to be JSON per each agent's system prompt). Includes basic
    latency logging for the /metrics endpoint to consume.
    """
    start = time.time()

    if PROVIDER == "groq":
        text = _call_groq(messages, temperature)

    elif PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content

    elif PROVIDER == "azure_openai":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-06-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content

    else:
        raise NotImplementedError(f"Unsupported LLM_PROVIDER: {PROVIDER}")

    elapsed_ms = (time.time() - start) * 1000
    logger.info("llm_call provider=%s model=%s latency_ms=%.1f", PROVIDER, MODEL, elapsed_ms)

    return text


def _call_groq(messages: list[dict], temperature: float) -> str:
    """
    Groq exposes an OpenAI-compatible /chat/completions endpoint, so we reuse
    the openai SDK and just point it at Groq's base_url. Groq supports JSON
    mode on most current models (llama-3.x, mixtral, etc.) via response_format.

    NOTE: Groq does not offer an embeddings endpoint. Embeddings for RAG
    (see app/rag/retriever.py) still go through OpenAI regardless of which
    LLM_PROVIDER is set here.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        # Some Groq models/versions reject response_format; retry without it
        # and rely on the system prompt's "return ONLY JSON" instruction.
        logger.warning("Groq call with response_format failed (%s), retrying without it", e)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )

    return resp.choices[0].message.content
