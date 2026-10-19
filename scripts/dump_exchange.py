"""
Script to dump the exact sanitized request/response exchange for answer A01.
Saves to reports/sample_exchange_A01.json.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import types
from agent.loader import load_answers
from agent.facts_tool import get_gemini_tool_declaration, execute_tool_call
from agent.agent import load_prompt_template

load_dotenv()


def main():
    answers = load_answers("data/answers.csv", split="development")
    a01 = next(a for a in answers if a["answer_id"] == "A01")

    system_prompt = load_prompt_template("v1")
    tool = get_gemini_tool_declaration()

    client = genai.Client(http_options=types.HttpOptions(timeout=60000))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tool],
        temperature=0.0,
    )

    chat = client.chats.create(model="gemini-3.5-flash", config=config)

    user_message = (
        f"Please analyze the following answer and extract and classify its factual claims:\n\n"
        f"answer_id: {a01['answer_id']}\n"
        f"brand_id: {a01['brand_id']}\n"
        f"question: {a01['question']}\n\n"
        f"ai_answer:\n{a01['ai_answer']}"
    )

    resp1 = chat.send_message(user_message)

    tool_calls_data = []
    for fc in resp1.function_calls or []:
        call_res = execute_tool_call(fc.name, fc.args)
        tool_calls_data.append({
            "name": fc.name,
            "args": dict(fc.args or {}),
            "result": call_res,
        })

    tool_responses = [
        types.Part.from_function_response(name=tc["name"], response={"result": tc["result"]})
        for tc in tool_calls_data
    ]

    resp2 = chat.send_message(tool_responses)

    exchange = {
        "answer_id": "A01",
        "model": "gemini-3.5-flash",
        "system_prompt": system_prompt,
        "turn_1_user_message": user_message,
        "turn_1_model_tool_calls": [
            {"name": tc["name"], "args": tc["args"]} for tc in tool_calls_data
        ],
        "turn_2_tool_results": [
            {"name": tc["name"], "result": tc["result"]} for tc in tool_calls_data
        ],
        "turn_2_model_response": resp2.text,
    }

    Path("reports").mkdir(exist_ok=True)
    out_file = Path("reports/sample_exchange_A01.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(exchange, f, indent=2)

    print(f"Saved {out_file} successfully ({out_file.stat().st_size} bytes).")


if __name__ == "__main__":
    main()
