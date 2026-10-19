"""
Claim extraction and classification agent module.
Uses google-genai SDK with real function calling against the brand facts catalogue.
Enforces:
  - Max 3 model turns per answer
  - 60s provider request timeout
  - Exponential backoff for 429/transient errors
  - Strict tool validation (executor validates tool name & brand)
  - Raw quote extraction by model -> Python character offset computation
  - Strict validation of bounds, quote equality, no overlap, labels, fact IDs
  - Never passes reference benchmark data into model/tool payload
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.facts_tool import (
    execute_tool_call,
    get_gemini_tool_declaration,
    ToolExecutionError,
    UnknownBrandError,
    IllegalToolError,
)
from agent.loader import get_valid_fact_ids, get_valid_brand_ids
from agent.spans import locate_claims, SpanError
from agent.validate import validate_claims, ClaimValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TURNS = 3


class AgentError(Exception):
    """Base exception for agent-level errors."""
    pass


class TurnExhaustionError(AgentError):
    """Raised when model reaches turn limit without completing extraction."""
    pass


class InvalidOutputError(AgentError):
    """Raised when model response is not valid JSON or structure."""
    pass


def load_prompt_template(version: str = "v1") -> str:
    """Load system prompt from agent/prompts/{version}.txt."""
    prompt_file = Path("agent") / "prompts" / f"{version}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if present."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def parse_model_claims(raw_text: str, expected_answer_id: str) -> List[Dict[str, Any]]:
    """Parse JSON claims from model text output."""
    cleaned = clean_json_text(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise InvalidOutputError(f"Model returned invalid JSON: {e}. Raw: {raw_text[:200]}")

    if not isinstance(data, dict):
        raise InvalidOutputError(f"Model JSON root must be an object, got {type(data)}")

    claims = data.get("claims")
    if not isinstance(claims, list):
        raise InvalidOutputError(f"Model JSON missing 'claims' list or wrong type: {type(claims)}")

    return claims


def call_with_retry(
    callable_fn,
    max_retries: int = 4,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0
):
    """
    Execute callable with exponential backoff for transient provider errors (e.g. 429, 503).
    """
    delay = initial_delay
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return callable_fn()
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_transient = (
                "429" in err_str
                or "resource_exhausted" in err_str
                or "503" in err_str
                or "unavailable" in err_str
                or "timeout" in err_str
                or "deadline" in err_str
            )
            if attempt < max_retries and is_transient:
                logger.warning(f"Transient error on attempt {attempt + 1}: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise last_exc


class ClaimAgent:
    """Agent that performs claim extraction and classification using Gemini function calling."""

    def __init__(
        self,
        prompt_version: str = "v1",
        model_name: str = DEFAULT_MODEL,
        catalogue_path: str | Path = "data/brand_facts.csv",
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        load_dotenv()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env")

        self.prompt_version = prompt_version
        self.model_name = model_name
        self.catalogue_path = Path(catalogue_path)
        self.timeout = timeout
        self.valid_fact_ids = get_valid_fact_ids(self.catalogue_path)
        self.valid_brand_ids = get_valid_brand_ids(self.catalogue_path)

        self.system_instruction = load_prompt_template(prompt_version)
        self.tool_declaration = get_gemini_tool_declaration()

        # Initialize GenAI Client with configured timeout (in milliseconds for google-genai SDK)
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
        )

    def process_answer(
        self,
        answer_id: str,
        brand_id: str,
        question: str,
        ai_answer: str,
    ) -> Dict[str, Any]:
        """
        Process a single answer through the agent.
        Enforces 3-turn limit, real tool execution, span location, and strict validation.
        Returns a dict with:
          answer_id, status ('success' or 'error'), claims, error, latency_ms, tool_calls_log
        """
        start_time = time.time()
        tool_calls_log: List[Dict[str, Any]] = []

        try:
            # Validate input brand
            brand_id_clean = brand_id.strip().lower()
            if brand_id_clean not in self.valid_brand_ids:
                raise UnknownBrandError(
                    f"Unknown primary brand_id '{brand_id}'. Known brands: {sorted(self.valid_brand_ids)}"
                )

            # Build generation config
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=[self.tool_declaration],
                temperature=0.0,
            )

            # Initialize multi-turn chat session
            chat = self.client.chats.create(model=self.model_name, config=config)

            # Turn 1: Initial user message (input question and answer context)
            initial_prompt = (
                f"Please analyze the following answer and extract and classify its factual claims:\n\n"
                f"answer_id: {answer_id}\n"
                f"brand_id: {brand_id_clean}\n"
                f"question: {question}\n\n"
                f"ai_answer:\n{ai_answer}"
            )

            turn_count = 1
            response = call_with_retry(lambda: chat.send_message(initial_prompt))

            # Multi-turn tool execution loop
            while turn_count <= MAX_TURNS:
                function_calls = getattr(response, "function_calls", None) or []

                if not function_calls:
                    # Model provided final text response
                    break

                # If we have reached MAX_TURNS and model is still calling tools, turn limit is exhausted
                if turn_count >= MAX_TURNS:
                    raise TurnExhaustionError(
                        f"Turn limit of {MAX_TURNS} exhausted with pending tool calls: {[fc.name for fc in function_calls]}"
                    )

                # Execute all requested tool calls in this turn
                tool_responses = []
                for fc in function_calls:
                    tool_name = fc.name
                    args = dict(fc.args or {})
                    tool_calls_log.append({
                        "turn": turn_count,
                        "tool_name": tool_name,
                        "args": args,
                    })

                    # Executor validates tool name and brand argument
                    call_result = execute_tool_call(
                        tool_name=tool_name,
                        args=args,
                        catalogue_path=self.catalogue_path,
                    )

                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": call_result},
                        )
                    )

                # Send tool responses back to model in next turn
                turn_count += 1
                response = call_with_retry(lambda: chat.send_message(tool_responses))

            # Check if response has text
            raw_text = response.text
            if not raw_text or not raw_text.strip():
                # Inspect candidate parts
                parts_text = []
                for cand in getattr(response, "candidates", []) or []:
                    for p in getattr(cand.content, "parts", []) or []:
                        if getattr(p, "text", None):
                            parts_text.append(p.text)
                raw_text = "".join(parts_text)

            if not raw_text or not raw_text.strip():
                raise InvalidOutputError("Model returned empty text response after conversation turns.")

            # Parse candidate claims from JSON
            raw_claims = parse_model_claims(raw_text, expected_answer_id=answer_id)

            # Python computes exact start_char and end_char Unicode offsets
            located_claims = locate_claims(
                answer_id=answer_id,
                ai_answer=ai_answer,
                raw_claims=raw_claims,
            )

            # Strict validation
            validated_claims = validate_claims(
                answer_id=answer_id,
                ai_answer=ai_answer,
                claims=located_claims,
                valid_fact_ids=self.valid_fact_ids,
            )

            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "answer_id": answer_id,
                "status": "success",
                "claims": validated_claims,
                "error": None,
                "latency_ms": latency_ms,
                "tool_calls": tool_calls_log,
                "raw_model_output": raw_text,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_type = type(e).__name__
            error_message = str(e)
            logger.error(f"Error processing answer {answer_id}: [{error_type}] {error_message}")

            # Answer-level error with NO accepted predictions
            return {
                "answer_id": answer_id,
                "status": "error",
                "claims": [],
                "error": {
                    "error_type": error_type,
                    "message": error_message,
                },
                "latency_ms": latency_ms,
                "tool_calls": tool_calls_log,
                "raw_model_output": None,
            }
