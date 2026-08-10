"""
Agent 1 — Extraction Agent

Takes raw unstructured claim text (FNOL form, adjuster notes, etc.) and returns
a Pydantic-validated ExtractedClaim. Uses a strict system prompt + JSON schema
enforcement, with a repair-retry loop if the model returns invalid JSON.
"""
import json
import logging

from pydantic import ValidationError

from app.models.schemas import ExtractedClaim
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an insurance claims data-extraction specialist.

Extract structured fields from the raw claim document below. Return ONLY valid JSON
matching this schema, with no markdown fences, no preamble, no explanation:

{
  "claim_id": string,
  "policyholder_name": string,
  "policy_number": string,
  "loss_date": string (YYYY-MM-DD) or null,
  "claim_amount": number or null,
  "loss_description": string,
  "loss_type": string or null,
  "missing_fields": array of field names you could not confidently populate
}

Rules:
- If a field is not present in the text, set it to null and add its name to missing_fields.
- Never invent values for factual fields like claim_amount, policy_number, or dates. Do not
  guess numbers or IDs that aren't stated.
- EXCEPTION: loss_type is a categorization field, not a factual one — infer it from the
  loss_description even if not stated as a single word (e.g. "burst pipe flooded the floor"
  -> "water damage"; "smoke damage to walls" -> "fire"). Only set loss_type to null and add
  it to missing_fields if the description gives no usable signal at all.
- loss_description should be a concise 1-2 sentence summary in your own words.
"""


def extract_claim(raw_text: str, max_retries: int = 2) -> ExtractedClaim:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]

    last_error = None
    for attempt in range(max_retries + 1):
        response_text = call_llm(messages)
        try:
            data = json.loads(response_text)
            return ExtractedClaim(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning("Extraction attempt %s failed validation: %s", attempt, e)
            # repair prompt: ask the model to fix its own output
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That response was invalid JSON or failed schema validation: {e}. "
                        "Return corrected JSON only, matching the schema exactly."
                    ),
                }
            )

    raise ValueError(f"Extraction failed after {max_retries + 1} attempts: {last_error}")
