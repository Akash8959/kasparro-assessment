"""
Tests using a mock/fake Gemini provider to verify error handling, edge cases, and zero benchmark data leakage.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from agent.agent import (
    ClaimAgent,
    InvalidOutputError,
    TurnExhaustionError,
    load_prompt_template,
)
from agent.facts_tool import UnknownBrandError, IllegalToolError
from agent.validate import ClaimValidationError


class FakeChat:
    """Mock Gemini chat session allowing step-by-step scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent_messages = []

    def send_message(self, message):
        self.sent_messages.append(message)
        if not self.responses:
            raise RuntimeError("No more scripted responses in FakeChat.")
        return self.responses.pop(0)


class FakeResponse:
    """Mock Gemini response with text and optional function_calls."""

    def __init__(self, text="", function_calls=None):
        self.text = text
        self.function_calls = function_calls or []
        self.candidates = []


class FakeFunctionCall:
    """Mock Gemini function call object."""

    def __init__(self, name, args):
        self.name = name
        self.args = args


@pytest.fixture
def mock_agent():
    """Create a ClaimAgent with a mocked GenAI client."""
    with patch("agent.agent.genai.Client"):
        agent = ClaimAgent(
            prompt_version="v1",
            api_key="fake-test-key-never-sent",
        )
        return agent


def test_provider_malformed_json_returns_answer_error(mock_agent):
    """Verify malformed JSON from model causes an answer-level error with empty claims."""
    bad_json_response = FakeResponse(text="Here are the claims: {not valid json at all")
    fake_chat = FakeChat([bad_json_response])
    mock_agent.client.chats.create.return_value = fake_chat

    result = mock_agent.process_answer(
        answer_id="A99",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="ClearTrail Mini is $120.",
    )

    assert result["status"] == "error"
    assert result["claims"] == []
    assert result["error"]["error_type"] == "InvalidOutputError"
    assert "invalid json" in result["error"]["message"].lower()


def test_provider_unsupported_label_rejected(mock_agent):
    """Verify unsupported claim label (e.g., 'verified') causes answer-level error."""
    payload = {
        "claims": [
            {
                "quote": "ClearTrail Mini is $120.",
                "label": "verified",  # Invalid! Must be supported|contradicted|unknown
                "fact_ids": [],
                "reason": "Verified by catalogue.",
            }
        ]
    }
    response = FakeResponse(text=json.dumps(payload))
    fake_chat = FakeChat([response])
    mock_agent.client.chats.create.return_value = fake_chat

    result = mock_agent.process_answer(
        answer_id="A99",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="ClearTrail Mini is $120.",
    )

    assert result["status"] == "error"
    assert result["claims"] == []
    assert result["error"]["error_type"] == "ClaimValidationError"
    assert "label" in result["error"]["message"].lower()


def test_provider_invented_fact_id_rejected(mock_agent):
    """Verify model citing a non-existent fact ID causes answer-level error."""
    payload = {
        "claims": [
            {
                "quote": "ClearTrail Mini is $120.",
                "label": "supported",
                "fact_ids": ["nonexistent.catalogue.fact.id"],
                "reason": "Found it.",
            }
        ]
    }
    response = FakeResponse(text=json.dumps(payload))
    fake_chat = FakeChat([response])
    mock_agent.client.chats.create.return_value = fake_chat

    result = mock_agent.process_answer(
        answer_id="A99",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="ClearTrail Mini is $120.",
    )

    assert result["status"] == "error"
    assert result["claims"] == []
    assert result["error"]["error_type"] == "ClaimValidationError"
    assert "catalogue" in result["error"]["message"].lower()


def test_provider_unknown_brand_rejected(mock_agent):
    """Verify processing an unknown primary brand fails immediately with UnknownBrandError."""
    result = mock_agent.process_answer(
        answer_id="A99",
        brand_id="unknown_fictional_brand",
        question="What is the price?",
        ai_answer="The price is $10.",
    )

    assert result["status"] == "error"
    assert result["claims"] == []
    assert result["error"]["error_type"] == "UnknownBrandError"


def test_provider_illegal_tool_call_rejected(mock_agent):
    """Verify model attempting to call an illegal tool (e.g. 'web_search') causes answer error."""
    illegal_call = FakeFunctionCall(name="web_search", args={"query": "ClearTrail specs"})
    call_response = FakeResponse(function_calls=[illegal_call])
    fake_chat = FakeChat([call_response])
    mock_agent.client.chats.create.return_value = fake_chat

    result = mock_agent.process_answer(
        answer_id="A99",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="ClearTrail Mini is $120.",
    )

    assert result["status"] == "error"
    assert result["claims"] == []
    assert result["error"]["error_type"] == "IllegalToolError"


def test_provider_turn_exhaustion(mock_agent):
    """Verify agent stops when model exceeds 3 conversational turns and marks answer error."""
    tool_call = FakeFunctionCall(name="get_brand_facts", args={"brand_id": "cleartrail"})
    # Chat keeps responding with function calls infinitely
    fake_chat = FakeChat([
        FakeResponse(function_calls=[tool_call]),
        FakeResponse(function_calls=[tool_call]),
        FakeResponse(function_calls=[tool_call]),
        FakeResponse(function_calls=[tool_call]),
    ])
    mock_agent.client.chats.create.return_value = fake_chat

    result = mock_agent.process_answer(
        answer_id="A99",
        brand_id="cleartrail",
        question="What is the price?",
        ai_answer="ClearTrail Mini is $120.",
    )

    assert result["status"] == "error"
    assert result["claims"] == []
    assert result["error"]["error_type"] == "TurnExhaustionError"


def test_zero_benchmark_data_leakage(mock_agent):
    """
    CRITICAL TEST: Verify that reference gold benchmark terms and values
    NEVER enter the prompt, system instructions, or messages sent to the provider.
    """
    # 1. System instructions check
    for version in ("v1", "v2"):
        prompt_text = load_prompt_template(version)
        assert "development_gold" not in prompt_text
        assert "expected_label" not in prompt_text
        assert "evidence_fact_ids" not in prompt_text
        assert "difficulty_tags" not in prompt_text
        assert "gold" not in prompt_text.lower() or "gold" in prompt_text.lower() and "gold standard" not in prompt_text.lower()

    # 2. Sent messages check
    valid_payload = {
        "claims": [
            {
                "quote": "ClearTrail Mini is $120.",
                "label": "supported",
                "fact_ids": ["ct.mini.price"],
                "reason": "Verified from catalogue.",
            }
        ]
    }
    tool_call = FakeFunctionCall(name="get_brand_facts", args={"brand_id": "cleartrail"})
    final_response = FakeResponse(text=json.dumps(valid_payload))
    fake_chat = FakeChat([FakeResponse(function_calls=[tool_call]), final_response])
    mock_agent.client.chats.create.return_value = fake_chat

    mock_agent.process_answer(
        answer_id="A01",
        brand_id="cleartrail",
        question="Sample question?",
        ai_answer="ClearTrail Mini is $120.",
    )

    for msg in fake_chat.sent_messages:
        msg_str = str(msg)
        assert "expected_label" not in msg_str
        assert "evidence_fact_ids" not in msg_str
        assert "difficulty_tags" not in msg_str
        assert "gold" not in msg_str.lower()
