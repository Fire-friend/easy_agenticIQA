"""
JSON parsing utilities for handling VLM responses.
"""

import json
import re
from typing import Any, Dict


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON content from VLM response.

    Handles common patterns:
    - JSON wrapped in markdown code blocks (```json ... ``` or ``` ... ```)
    - JSON with surrounding text
    - Plain JSON

    Args:
        response: Raw response string from VLM

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no JSON content can be extracted
    """
    # Try to extract JSON from ```json ... ``` or ``` ... ``` blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # If no code block, try to find JSON object directly
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()

    # If still no match, try array format
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()

    # Return original response stripped
    return response.strip()


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from VLM response with automatic extraction.

    Args:
        response: Raw response string from VLM

    Returns:
        Parsed JSON as dictionary

    Raises:
        ValueError: If JSON cannot be parsed
    """
    try:
        # Extract JSON content
        json_str = extract_json_from_response(response)

        # Parse JSON
        data = json.loads(json_str)

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}. Response preview: {response[:200]}")
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
