"""
Brand facts tool module for live function calling.
Validates tool name, brand argument, and returns brand fact rows with fact_version 'speaker-catalogue-v4'.
"""

from pathlib import Path
from typing import Any, Dict, List, Set
from google.genai import types

from agent.loader import load_brand_facts, get_valid_brand_ids, FACT_VERSION


class ToolExecutionError(Exception):
    """Raised when an illegal tool name or unknown brand is requested."""
    pass


class UnknownBrandError(ToolExecutionError):
    """Raised when brand_id is unknown."""
    pass


class IllegalToolError(ToolExecutionError):
    """Raised when an unrecognized tool name is called."""
    pass


def get_brand_facts(brand_id: str, catalogue_path: str | Path = "data/brand_facts.csv") -> Dict[str, Any]:
    """
    Look up catalogue facts for a given brand_id.
    Validates that brand_id is in the known catalogue brands.
    Returns the brand's rows along with fact_version 'speaker-catalogue-v4'.
    """
    known_brands = get_valid_brand_ids(catalogue_path)
    # Check exact match
    if not brand_id or brand_id not in known_brands:
        raise UnknownBrandError(f"Unknown brand_id '{brand_id}'. Known brands: {sorted(known_brands)}")

    all_facts = load_brand_facts(catalogue_path)
    brand_rows = [f for f in all_facts if f["brand_id"] == brand_id]

    return {
        "fact_version": FACT_VERSION,
        "brand_id": brand_id,
        "facts": brand_rows,
    }


def execute_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    catalogue_path: str | Path = "data/brand_facts.csv"
) -> Dict[str, Any]:
    """
    Validate tool name and arguments, and execute the tool.
    Raises IllegalToolError if tool_name is not 'get_brand_facts'.
    Raises UnknownBrandError or ToolExecutionError if arguments are invalid.
    """
    if tool_name != "get_brand_facts":
        raise IllegalToolError(f"Illegal tool name requested: '{tool_name}'. Only 'get_brand_facts' is allowed.")

    if not isinstance(args, dict) or "brand_id" not in args:
        raise ToolExecutionError(f"Missing required parameter 'brand_id' in tool call arguments: {args}")

    brand_id = str(args["brand_id"]).strip().lower()
    return get_brand_facts(brand_id, catalogue_path=catalogue_path)


def get_gemini_tool_declaration() -> types.Tool:
    """Return the Gemini SDK Tool declaration for get_brand_facts."""
    func_decl = types.FunctionDeclaration(
        name="get_brand_facts",
        description="Retrieve catalogue facts for a specific speaker brand. Returns catalogue fact rows and fact_version 'speaker-catalogue-v4'.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "brand_id": types.Schema(
                    type=types.Type.STRING,
                    description="The brand identifier to look up (e.g. 'cleartrail', 'cloudnote', 'stonewave')."
                )
            },
            required=["brand_id"]
        )
    )
    return types.Tool(function_declarations=[func_decl])
